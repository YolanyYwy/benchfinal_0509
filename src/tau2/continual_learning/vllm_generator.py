"""
vLLM rollout generator — per-rank tp=1, persistent engine with weight offload.

Requires vLLM <= 0.6.x (V0 engine, UniProcExecutor, no subprocess).
vLLM 0.14+ forces V1 engine which spawns EngineCore subprocesses that
deadlock when 8 DeepSpeed ranks all initialise NCCL simultaneously.

Install: pip install vllm==0.6.6

Architecture:
  - Each DeepSpeed rank owns one GPU (local_rank == GPU index)
  - One vLLM engine (tp=1) is created per rank, kept alive across steps
  - Before rollout: DS model params moved to CPU; vLLM weights synced from DS
  - After rollout:  vLLM weights moved to CPU; DS model reloaded to GPU
"""

from __future__ import annotations

import gc
import os
import pickle
import uuid
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


class VLLMRolloutEngine:

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 3072,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
    ):
        self.model_path             = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len          = max_model_len
        self.dtype                  = dtype
        self.enforce_eager          = enforce_eager

        self.rank      = int(os.environ.get("LOCAL_RANK", 0))
        self.local_gpu = self.rank

        self._engine       = None
        self._tokenizer    = None
        self._ds_model     = None
        self._ref_model    = None
        self._extra_models = []
        self._cpu_cache: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, ds_model: torch.nn.Module, world_size: int = 1,
             ref_model: torch.nn.Module = None,
             extra_models_to_offload: list = None):
        from vllm import LLM
        from transformers import AutoTokenizer

        self._ds_model  = ds_model
        self._ref_model = ref_model
        self._extra_models = []
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        import time as _time
        import os as _os
        import torch.distributed as _tdist

        _single_gpu = (world_size == 1)

        # ── Multi-GPU: offload DS model to CPU so vLLM can use the GPU ──────
        if not _single_gpu:
            print(f"[VLLMRolloutEngine rank={self.rank}] Offloading DS model to CPU...",
                  flush=True)
            for p in ds_model.parameters():
                p.data = p.data.cpu()
            torch.cuda.empty_cache()

        _time.sleep(self.rank * 2)
        print(f"[VLLMRolloutEngine rank={self.rank}] Creating vLLM engine "
              f"(tp=1, gpu={self.local_gpu}, mem={self.gpu_memory_utilization})...",
              flush=True)

        _dist_keys = ["RANK", "LOCAL_RANK", "WORLD_SIZE",
                      "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE"]

        if _single_gpu:
            # ── Single-GPU: init a file-based gloo pg for vLLM ──────────────
            _saved_env = {k: _os.environ.pop(k) for k in _dist_keys if k in _os.environ}
            try:
                if not _tdist.is_initialized():
                    import tempfile as _tempfile
                    _store_file = _tempfile.mktemp(prefix="vllm_dist_")
                    _store = _tdist.FileStore(_store_file, 1)
                    _tdist.init_process_group(backend="gloo", store=_store,
                                              rank=0, world_size=1)
                self._engine = LLM(
                    model=self.model_path,
                    tokenizer=self.model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    dtype=self.dtype,
                    enforce_eager=self.enforce_eager,
                    trust_remote_code=True,
                    disable_log_stats=True,
                    device=f"cuda:{self.local_gpu}",
                )
            finally:
                _os.environ.update(_saved_env)
            # Single-GPU: vLLM loaded weights from disk (same as DS model).
            # Offload vLLM weights to CPU now; generate() will sync+reload each step.
            self._offload_vllm_weights()

        else:
            # ── Multi-GPU: destroy DS pg, let vLLM init its own, then rebuild ─
            _ds_rank       = _tdist.get_rank()
            _ds_world_size = _tdist.get_world_size()
            _master_addr   = _os.environ.get("MASTER_ADDR", "127.0.0.1")
            _master_port   = int(_os.environ.get("MASTER_PORT", "29500"))
            _tdist.destroy_process_group()

            _saved_env = {k: _os.environ.pop(k) for k in _dist_keys if k in _os.environ}
            try:
                import tempfile as _tempfile
                _store_file = _tempfile.mktemp(prefix="vllm_dist_")
                _store_obj = _tdist.FileStore(_store_file, 1)
                _tdist.init_process_group(backend="gloo", store=_store_obj,
                                          rank=0, world_size=1)
                self._engine = LLM(
                    model=self.model_path,
                    tokenizer=self.model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    dtype=self.dtype,
                    enforce_eager=self.enforce_eager,
                    trust_remote_code=True,
                    disable_log_stats=True,
                    device=f"cuda:{self.local_gpu}",
                )
            finally:
                _os.environ.update(_saved_env)
                # Destroy vLLM's gloo pg and rebuild DS nccl pg
                if _tdist.is_initialized():
                    _tdist.destroy_process_group()
                _store_ds = _tdist.TCPStore(
                    host_name=_master_addr,
                    port=_master_port + 100,
                    world_size=_ds_world_size,
                    is_master=(_ds_rank == 0),
                    timeout=_tdist.default_pg_timeout,
                )
                _tdist.init_process_group(
                    backend="nccl",
                    store=_store_ds,
                    rank=_ds_rank,
                    world_size=_ds_world_size,
                )
            # Multi-GPU: offload vLLM weights to CPU (DS model will reload later)
            self._offload_vllm_weights()

        print(f"[VLLMRolloutEngine rank={self.rank}] Engine ready.", flush=True)
        if world_size > 1:
            dist.barrier()

    def destroy(self):
        if self._engine is None:
            return
        try:
            self._engine.llm_engine.model_executor.shutdown()
        except Exception:
            pass
        del self._engine
        self._engine = None
        self._cpu_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _destroy_engine(self):
        """Destroy vLLM engine to free KV cache GPU memory after generation.
        The engine will be recreated on the next generate() call."""
        if self._engine is None:
            return
        try:
            self._engine.llm_engine.model_executor.shutdown()
        except Exception:
            pass
        del self._engine
        self._engine = None
        self._cpu_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _free_kv_cache(self):
        """Free KV cache GPU tensors to reclaim memory for backward pass."""
        try:
            executor = self._engine.llm_engine.model_executor
            if hasattr(executor, "driver_worker"):
                worker = executor.driver_worker
            elif hasattr(executor, "workers") and executor.workers:
                worker = executor.workers[0]
            else:
                print(f"[VLLMRolloutEngine rank={self.rank}] _free_kv_cache: "
                      f"cannot find worker, executor={type(executor).__name__} "
                      f"attrs={[a for a in dir(executor) if not a.startswith('_')]}",
                      flush=True)
                return

            # Unwrap WorkerWrapperBase if needed
            inner_worker = worker
            if hasattr(inner_worker, "worker"):
                inner_worker = inner_worker.worker

            if hasattr(inner_worker, "cache_engine") and inner_worker.cache_engine:
                for ce in inner_worker.cache_engine:
                    if hasattr(ce, "gpu_cache"):
                        del ce.gpu_cache
                        ce.gpu_cache = []
                inner_worker.cache_engine = []
                print(f"[VLLMRolloutEngine rank={self.rank}] KV cache freed.",
                      flush=True)
            else:
                print(f"[VLLMRolloutEngine rank={self.rank}] _free_kv_cache: "
                      f"worker type={type(inner_worker).__name__} "
                      f"attrs={[a for a in dir(inner_worker) if not a.startswith('_') and 'cache' in a.lower()]}",
                      flush=True)
        except Exception as e:
            print(f"[VLLMRolloutEngine rank={self.rank}] _free_kv_cache warning: {e}",
                  flush=True)

    def _get_vllm_model(self):
        executor = self._engine.llm_engine.model_executor
        if hasattr(executor, "driver_worker"):
            return executor.driver_worker.model_runner.model
        if hasattr(executor, "workers") and executor.workers:
            return executor.workers[0].model_runner.model
        raise RuntimeError(
            f"Cannot locate vLLM model. executor type={type(executor).__name__}, "
            f"attrs={[a for a in dir(executor) if not a.startswith('_')]}"
        )

    def _offload_vllm_weights(self):
        """Move vLLM model weights to CPU, saving them in _cpu_cache."""
        model = self._get_vllm_model()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self._cpu_cache:
                    self._cpu_cache[name] = param.data.cpu().clone()
                else:
                    self._cpu_cache[name].copy_(param.data.cpu())
                param.data = self._cpu_cache[name]
        torch.cuda.empty_cache()

    def _sync_weights_from_ds(self):
        """Copy current DS model weights into vLLM model in-place."""
        gpu = torch.device(f"cuda:{self.local_gpu}")

        unwrapped = self._ds_model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        vllm_model = self._get_vllm_model()
        ds_params  = dict(unwrapped.named_parameters())

        with torch.no_grad():
            for name, vp in vllm_model.named_parameters():
                if name in ds_params:
                    # Move vLLM param to GPU first, then copy DS weights in-place
                    if vp.data.device.type == "cpu":
                        vp.data = vp.data.to(gpu)
                    vp.data.copy_(ds_params[name].data.to(gpu))
                elif vp.data.device.type == "cpu":
                    vp.data = vp.data.to(gpu)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def generate(
        self,
        samples: list,
        G: int,
        temperature: float,
        max_new_tokens: int,
        world_size: int,
        device: torch.device,
    ) -> Tuple[list, list, list]:
        from vllm import SamplingParams
        from .policy_model import APIBankTrajectory

        B          = len(samples)
        my_indices = list(range(self.rank, B, world_size))
        _single_gpu = (world_size == 1)

        # 1. Offload DS model to CPU (both single and multi GPU),
        #    then sync latest DS weights into vLLM (which is on GPU via _cpu_cache).
        for p in self._ds_model.parameters():
            p.data = p.data.cpu()
        torch.cuda.empty_cache()
        self._sync_weights_from_ds()  # loads vLLM weights back to GPU from _cpu_cache

        # 3. Build prompts
        prompts:    List[str] = []
        sample_idx: List[int] = []
        uids:       List[str] = []
        uid_per = {i: str(uuid.uuid4()) for i in my_indices}

        for i in my_indices:
            sample   = samples[i]
            messages = sample.to_prompt_messages()
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

            # Skip samples whose prompt exceeds max_model_len.
            token_ids = self._tokenizer.encode(prompt)
            if len(token_ids) > self.max_model_len:
                print(f"[VLLMRolloutEngine rank={self.rank}] prompt idx={i} "
                      f"skipped ({len(token_ids)} tokens > max_model_len={self.max_model_len})",
                      flush=True)
                continue

            for _ in range(G):
                prompts.append(prompt)
                sample_idx.append(i)
                uids.append(uid_per[i])

        # 4. Generate
        local_trajs: list = []
        local_olps:  list = []
        local_uids:  list = []

        if prompts:
            # stop on </tool_call> so generation never gets truncated mid-block
            stop_str = "</tool_call>"
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                logprobs=1,          # must be >= 1 to get per-token logprobs for PPO
                stop=[stop_str],
                stop_token_ids=[self._tokenizer.eos_token_id],
                include_stop_str_in_output=True,  # keep </tool_call> in output_text
            )
            import time
            t0 = time.time()
            outputs = self._engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=(self.rank == 0),
            )
            print(f"[VLLMRolloutEngine rank={self.rank}] "
                  f"{len(outputs)} seqs in {time.time()-t0:.1f}s", flush=True)

            for j, (out, s_idx) in enumerate(zip(outputs, sample_idx)):
                sample      = samples[s_idx]
                output_text = out.outputs[0].text   # raw text, keep for PPO (must match resp_ids)
                resp_ids    = list(out.outputs[0].token_ids)

                # reward_text is what the reward function sees:
                # - append </tool_call> if stop fired but tag was stripped
                # - strip <think>...</think> so reward only scores the tool call
                import re as _re
                reward_text = output_text
                if "<tool_call>" in reward_text and "</tool_call>" not in reward_text:
                    reward_text = reward_text + "</tool_call>"
                reward_text = _re.sub(r"<think>.*?</think>\s*", "", reward_text, flags=_re.DOTALL)

                lp_list = []
                if out.outputs[0].logprobs:
                    for tok_id, lp_dict in zip(resp_ids, out.outputs[0].logprobs):
                        if lp_dict is None:
                            lp_list.append(0.0)
                        elif tok_id in lp_dict:
                            lp_list.append(lp_dict[tok_id].logprob)
                        else:
                            lp_list.append(next(iter(lp_dict.values())).logprob)

                old_lp = (torch.tensor(lp_list, dtype=torch.float32)
                          if lp_list else torch.zeros(1))

                local_trajs.append(APIBankTrajectory(
                    sample_id=sample.id,
                    prompt=prompts[j],
                    output_text=output_text,
                    reward_text=reward_text,
                    prompt_ids=self._tokenizer.encode(
                        prompts[j], return_tensors="pt")[0].cpu(),
                    response_ids=torch.tensor(resp_ids, dtype=torch.long),
                    gold_tool_calls=getattr(sample, "ground_truth",
                                    getattr(sample, "gold_tool_calls", [])),
                    ref_log_probs=None,  # filled in batch below
                ))
                local_olps.append(old_lp)
                local_uids.append(uids[j])

        # 5. Sync all ranks before touching shared NCCL state.
        # Fast ranks must not start DS reload (which may trigger DS collectives)
        # while slow ranks are still inside vLLM generate().
        print(f"[VLLMRolloutEngine rank={self.rank}] reaching pre-offload barrier", flush=True)
        if world_size > 1:
            dist.barrier()
        print(f"[VLLMRolloutEngine rank={self.rank}] passed pre-offload barrier", flush=True)

        # 6. Offload vLLM weights to CPU, reload DS model to GPU for backward pass.
        # Both single-GPU and multi-GPU: same flow.
        target = torch.device(f"cuda:{self.local_gpu}")
        if not _single_gpu:
            self._free_kv_cache()
        self._offload_vllm_weights()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[VLLMRolloutEngine rank={self.rank}] reloading DS model to GPU", flush=True)
        for p in self._ds_model.parameters():
            p.data = p.data.to(target)
        print(f"[VLLMRolloutEngine rank={self.rank}] DS model reload done", flush=True)

        # 7. Recompute old_log_probs using DS actor model (ToolRL style).
        # DS model now has the same weights as when vLLM generated — this gives
        # accurate old_log_probs for the PPO ratio computation.
        if local_trajs:
            print(f"[VLLMRolloutEngine rank={self.rank}] recomputing old_log_probs "
                  f"via actor forward ({len(local_trajs)} trajs)...", flush=True)
            ds_inner = self._ds_model
            while hasattr(ds_inner, "module"):
                ds_inner = ds_inner.module
            ds_inner.eval()
            with torch.no_grad():
                for idx_t, traj in enumerate(local_trajs):
                    if traj.response_ids is None or len(traj.response_ids) == 0:
                        continue
                    try:
                        prompt_t   = traj.prompt_ids.unsqueeze(0).to(target)
                        resp_t     = traj.response_ids.unsqueeze(0).to(target)
                        full_t     = torch.cat([prompt_t, resp_t], dim=1)
                        n_p        = prompt_t.shape[1]
                        act_out    = ds_inner(full_t)
                        act_logits = act_out.logits[
                            :, n_p - 1: n_p - 1 + resp_t.shape[1], :]
                        act_lp     = (F.log_softmax(act_logits, dim=-1)
                                      .gather(2, resp_t.unsqueeze(-1))
                                      .squeeze(-1).squeeze(0).cpu())
                        if torch.isnan(act_lp).any() or torch.isinf(act_lp).any():
                            print(f"[VLLMRolloutEngine rank={self.rank}] "
                                  f"NaN/inf in act_lp idx={idx_t}, keeping vLLM logprob",
                                  flush=True)
                        else:
                            local_olps[idx_t] = act_lp
                        del act_out, act_logits, full_t, prompt_t, resp_t
                    except Exception as e:
                        print(f"[VLLMRolloutEngine rank={self.rank}] "
                              f"actor old_lp failed idx={idx_t}: {e}", flush=True)
            ds_inner.train()
            torch.cuda.empty_cache()
            print(f"[VLLMRolloutEngine rank={self.rank}] old_log_probs recomputed.", flush=True)

        # 8. Compute ref_log_probs in batch (ToolRL style: after rollout, before training).
        # Move ref model to GPU transiently, compute all ref log_probs, move back to CPU.
        if self._ref_model is not None and local_trajs:
            print(f"[VLLMRolloutEngine rank={self.rank}] computing ref_log_probs "
                  f"({len(local_trajs)} trajs)...", flush=True)
            self._ref_model.to(target)
            self._ref_model.eval()
            with torch.no_grad():
                for idx_t, traj in enumerate(local_trajs):
                    if traj.response_ids is None or len(traj.response_ids) == 0:
                        continue
                    try:
                        prompt_t = traj.prompt_ids.unsqueeze(0).to(target)
                        resp_t   = traj.response_ids.unsqueeze(0).to(target)
                        full_t   = torch.cat([prompt_t, resp_t], dim=1)
                        n_p      = prompt_t.shape[1]
                        ref_out  = self._ref_model(full_t)
                        ref_logits = ref_out.logits[
                            :, n_p - 1: n_p - 1 + resp_t.shape[1], :]
                        ref_lp   = (F.log_softmax(ref_logits, dim=-1)
                                    .gather(2, resp_t.unsqueeze(-1))
                                    .squeeze(-1).squeeze(0).cpu())
                        traj.ref_log_probs = ref_lp
                        del ref_out, ref_logits, full_t, prompt_t, resp_t
                    except Exception as e:
                        print(f"[VLLMRolloutEngine rank={self.rank}] "
                              f"ref_lp failed idx={idx_t}: {e}", flush=True)
            self._ref_model.cpu()
            torch.cuda.empty_cache()
            print(f"[VLLMRolloutEngine rank={self.rank}] ref_log_probs done.", flush=True)

        # Wait for all ranks to finish reload before all_gather
        print(f"[VLLMRolloutEngine rank={self.rank}] reaching pre-allgather barrier", flush=True)
        if world_size > 1:
            dist.barrier()
        print(f"[VLLMRolloutEngine rank={self.rank}] passed pre-allgather barrier", flush=True)

        # 7. all_gather
        if world_size > 1:
            payload  = pickle.dumps((local_trajs, local_olps, local_uids))
            length_t = torch.tensor([len(payload)], dtype=torch.long, device=device)
            all_lens = [torch.zeros(1, dtype=torch.long, device=device)
                        for _ in range(world_size)]
            dist.all_gather(all_lens, length_t)

            max_len  = max(int(l.item()) for l in all_lens)
            # Use torch.tensor (copies data) instead of frombuffer (zero-copy but
            # risks Bus error if the bytearray is GC'd before async transfer completes)
            padded_bytes = bytes(payload) + bytes(max_len - len(payload))
            pay_t    = torch.tensor(
                list(padded_bytes), dtype=torch.uint8, device=device)
            all_pays = [torch.zeros(max_len, dtype=torch.uint8, device=device)
                        for _ in range(world_size)]
            dist.all_gather(all_pays, pay_t)

            all_trajs, all_olps, all_uids = [], [], []
            for pt, pl in zip(all_pays, all_lens):
                t, o, u = pickle.loads(
                    pt[:int(pl.item())].cpu().numpy().tobytes())
                all_trajs.extend(t)
                all_olps.extend(o)
                all_uids.extend(u)
        else:
            all_trajs, all_olps, all_uids = local_trajs, local_olps, local_uids

        if self.rank == 0:
            print(f"[VLLMRolloutEngine] {len(all_trajs)} trajectories gathered.",
                  flush=True)

        return all_trajs, all_olps, all_uids


# ---------------------------------------------------------------------------
# Singleton + functional wrapper
# ---------------------------------------------------------------------------

_engine: Optional[VLLMRolloutEngine] = None


def get_engine(
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enforce_eager: bool,
) -> VLLMRolloutEngine:
    global _engine
    if _engine is None:
        _engine = VLLMRolloutEngine(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=enforce_eager,
        )
    return _engine


def vllm_generate(
    samples: list,
    model_path: str,
    G: int,
    temperature: float,
    max_new_tokens: int,
    max_model_len: int,
    tensor_parallel_size: int,    # ignored, always tp=1 per rank
    gpu_memory_utilization: float,
    dtype: str,
    enforce_eager: bool,
    device: torch.device,
    world_size: int,
    ds_model: Optional[torch.nn.Module] = None,
    ref_model: Optional[torch.nn.Module] = None,
) -> Tuple[list, list, list]:
    engine = get_engine(
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=enforce_eager,
    )
    if engine._engine is None:
        assert ds_model is not None, \
            "ds_model must be provided on the first vllm_generate call"
        engine.init(ds_model, world_size=world_size, ref_model=ref_model)

    return engine.generate(
        samples=samples,
        G=G,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        world_size=world_size,
        device=device,
    )
