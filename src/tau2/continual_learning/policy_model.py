"""Policy model wrapper for GRPO training with open-source LLMs."""

import copy
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from AGentCL.agent.base import LocalAgent
from AGentCL.agent.llm_agent import LLMAgent, LLMAgentState
from AGentCL.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage, ToolCall
from AGentCL.data_model.tasks import Task
from AGentCL.environment.environment import Environment
from AGentCL.orchestrator.orchestrator import Orchestrator
from AGentCL.registry import registry
from AGentCL.user.user_simulator import UserSimulator
from AGentCL.user.base import ValidUserInputMessage, UserState

from .config import GRPOConfig
from .reward_oracle import Trajectory


def strip_thinking_content(text: str) -> str:
    """Remove all <think>...</think> content from generated text."""
    if not text:
        return ""

    cleaned = text.strip()

    # Remove complete think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)

    # Handle incomplete think tag
    if '<think>' in cleaned:
        think_start = cleaned.find('<think>')
        if '</think>' not in cleaned[think_start:]:
            cleaned = cleaned[:think_start]

    # Remove any remaining artifacts
    cleaned = re.sub(r'</think>\s*', '', cleaned)
    cleaned = re.sub(r'<think>\s*', '', cleaned)

    return cleaned.strip()


class LocalUserSimulator(UserSimulator):
    """User simulator that uses a local model instead of API."""

    def __init__(
        self,
        tools,
        instructions,
        model,
        tokenizer,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ):
        super().__init__(tools=tools, instructions=instructions, llm=None, llm_args={})
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _generate_next_message(self, message: ValidUserInputMessage, state: UserState) -> Tuple[UserMessage, UserState]:
        from AGentCL.data_model.message import MultiToolMessage

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()
        prompt = self._messages_to_prompt(messages)

        # Always unwrap for generation
        model = self.model
        while hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            # Limit max_new_tokens for faster generation
            max_tokens = min(self.max_new_tokens, 256)

            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if self.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
            else:
                generation_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **generation_kwargs)

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

        # Strip thinking content
        cleaned_text = strip_thinking_content(generated_text)

        # Default response if nothing left
        if not cleaned_text:
            cleaned_text = "Please continue."

        user_message = UserMessage(
            role="user",
            content=cleaned_text,
            cost=0.0,
            usage=None,
            raw_data=None,
        )

        state.messages.append(user_message)
        return user_message, state

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to prompt format.

        Note: flip_roles() has already been called, so we just need to
        convert the message types to their role strings without flipping again.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    # After flip_roles(), AssistantMessage = what user sim said
                    chat_messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, UserMessage):
                    # After flip_roles(), UserMessage = what agent said
                    chat_messages.append({"role": "user", "content": msg.content or ""})
                else:
                    chat_messages.append({"role": "user", "content": msg.content or ""})

            # Try to use enable_thinking=False for Qwen3 models
            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    prompt_parts.append(f"Assistant: {msg.content or ''}")
                elif isinstance(msg, UserMessage):
                    prompt_parts.append(f"User: {msg.content or ''}")
                else:
                    prompt_parts.append(f"User: {msg.content or ''}")
            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)


class PolicyModel:
    """Wrapper around open-source LLM for policy learning with gradient access."""

    def __init__(self, config: GRPOConfig, device: torch.device):
        self.config = config
        self.device = device

        # Check if using DeepSpeed
        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if local_rank == 0:
            print(f"Loading model: {config.model_name_or_path}")
            print(f"Using DeepSpeed: {use_deepspeed}")

        # When using DeepSpeed, don't use device_map - let DeepSpeed handle it
        # Use low_cpu_mem_usage to reduce memory footprint during loading
        if use_deepspeed:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                device_map={"": device},
                trust_remote_code=True,
            )

        self.model.train()
        if local_rank == 0:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Agent model: {trainable_params:,} / {total_params:,} parameters are trainable")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Note: gradient_checkpointing is incompatible with DeepSpeed ZeRO
        # when compute_grpo_loss does multiple forward passes per backward.
        # We disable it here and let DeepSpeed handle memory optimization.
        # If you need gradient checkpointing, set world_size=1 or use FSDP.
        self._gradient_checkpointing_enabled = False
        if config.gradient_checkpointing and config.world_size == 1:
            self.model.gradient_checkpointing_enable()
            self._gradient_checkpointing_enabled = True
            if local_rank == 0:
                print("[Warning] Gradient checkpointing enabled (single GPU mode)")
        elif config.gradient_checkpointing:
            if local_rank == 0:
                print("[Warning] Gradient checkpointing disabled for DeepSpeed compatibility")

        # Reference model — keep on CPU to save GPU memory.
        # KL is computed using old_log_probs (from generation) as the reference,
        # so we don't need a GPU forward pass through the ref model during training.
        if local_rank == 0:
            print("Loading reference model to CPU (no GPU memory used)...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=getattr(torch, config.model_dtype),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False
        # Keep on CPU — only moved to GPU transiently inside compute_log_probs_single_turn

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        if config.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=config.warmup_steps)
        else:
            self.scheduler = None

        if config.use_local_user_model:
            if local_rank == 0:
                print(f"Loading user simulator model: {config.user_model}")

            # User model - also don't use device_map with DeepSpeed
            if use_deepspeed:
                self.user_model = AutoModelForCausalLM.from_pretrained(
                    config.user_model,
                    torch_dtype=getattr(torch, config.model_dtype),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.user_model = self.user_model.to(device)
            else:
                self.user_model = AutoModelForCausalLM.from_pretrained(
                    config.user_model,
                    torch_dtype=getattr(torch, config.model_dtype),
                    device_map={"": device},
                    trust_remote_code=True,
                )
            self.user_model.eval()
            for p in self.user_model.parameters():
                p.requires_grad = False

            self.user_tokenizer = AutoTokenizer.from_pretrained(config.user_model, trust_remote_code=True)
            if self.user_tokenizer.pad_token is None:
                self.user_tokenizer.pad_token = self.user_tokenizer.eos_token

            if getattr(config, "local_rank", 0) == 0:
                print(f"User simulator model loaded on {device}")
        else:
            self.user_model = None
            self.user_tokenizer = None
            if getattr(config, "local_rank", 0) == 0:
                print(f"Using API for user simulator: {config.user_model}")

        if getattr(config, "local_rank", 0) == 0:
            print(f"Model loaded on {device}")

    def generate_responses(self, task: Task, environment: Environment, num_samples: int, domain: str, timeout_per_sample: float = 300.0) -> list[Trajectory]:
        """
        Generate trajectories on rank 0 only, then broadcast to all ranks.

        DeepSpeed ZeRO-2 shards parameters across ranks, so unwrapped_model.generate()
        on non-zero ranks uses an incomplete parameter set and produces garbage output.
        The fix: only rank 0 runs the orchestrator (with the full consolidated model),
        then we pickle+broadcast the trajectories so every rank has identical data for
        the backward pass.
        """
        import os
        import pickle
        import time as time_module
        import torch.distributed as dist

        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.model.eval()

        trajectories = []

        if rank == 0:
            print(f"\n  → Rank 0 generating {num_samples} trajectories (will broadcast to all ranks)...", flush=True)
            for sample_idx in range(num_samples):
                sample_start = time_module.time()
                try:
                    agent = self._create_agent(environment, domain)
                    user = self._create_user(task, domain)
                    orchestrator = Orchestrator(
                        domain=domain,
                        agent=agent,
                        user=user,
                        environment=environment,
                        task=task,
                        max_steps=50,
                    )
                    orchestrator.initialize()
                    simulation_run = orchestrator.run()
                    elapsed = time_module.time() - sample_start
                    print(f"  [Rank 0] Sample {sample_idx+1}/{num_samples} done in {elapsed:.1f}s, {len(simulation_run.messages)} messages", flush=True)
                    trajectories.append(Trajectory(
                        task_id=task.id,
                        messages=simulation_run.messages,
                        termination_reason=simulation_run.termination_reason,
                        cost=0.0,
                    ))
                except Exception as e:
                    elapsed = time_module.time() - sample_start
                    print(f"  [Rank 0] Sample {sample_idx+1} failed in {elapsed:.1f}s: {e}", flush=True)

            print(f"  [Rank 0] Generation done, got {len(trajectories)} trajectories. Broadcasting...", flush=True)

        # Broadcast trajectories from rank 0 to all other ranks
        if world_size > 1:
            # Serialize on rank 0, broadcast byte length, then broadcast payload
            if rank == 0:
                payload = pickle.dumps(trajectories)
                length_tensor = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
            else:
                length_tensor = torch.tensor([0], dtype=torch.long, device=self.device)

            dist.broadcast(length_tensor, src=0)
            payload_len = length_tensor.item()

            if rank == 0:
                padded_bytes = bytes(payload) + bytes(payload_len - len(payload))
                payload_tensor = torch.tensor(list(padded_bytes), dtype=torch.uint8, device=self.device)
            else:
                payload_tensor = torch.zeros(payload_len, dtype=torch.uint8, device=self.device)

            dist.broadcast(payload_tensor, src=0)

            if rank != 0:
                trajectories = pickle.loads(payload_tensor.cpu().numpy().tobytes())
                print(f"  [Rank {rank}] Received {len(trajectories)} trajectories from rank 0.", flush=True)

        return trajectories

    def _create_agent(self, environment: Environment, domain: str) -> LLMAgent:
        """Create an LLMAgent that uses the local model via the registry."""
        from AGentCL.utils.llm_utils import register_local_model

        tools_obj = environment.get_tools()
        tools = list(tools_obj.values()) if isinstance(tools_obj, dict) else list(tools_obj)
        domain_policy = environment.policy

        # Register the local model so LLMAgent can use it via "local:policy_model"
        register_local_model("policy_model", self.model, self.tokenizer)

        return LLMAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm="local:policy_model",
            llm_args={
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
            },
        )

    def _create_user(self, task: Task, domain: str):
        """Create a user simulator using GPT41UserSimulator.

        Uses GPT41UserSimulator which calls the zidongtaichu API via requests SSE streaming.
        """
        from AGentCL.user.user_simulator import GPT41UserSimulator

        # Get user tools for specific domains (e.g., telecom)
        user_tools = None
        if domain == "telecom":
            from AGentCL.domains.telecom.user_tools import TelecomUserTools
            from AGentCL.domains.telecom.user_data_model import TelecomUserDB
            user_db = TelecomUserDB()
            user_toolkit = TelecomUserTools(user_db)
            user_tools_obj = user_toolkit.get_tools()
            user_tools = list(user_tools_obj.values()) if isinstance(user_tools_obj, dict) else list(user_tools_obj)

        if self.config.use_local_user_model:
            return LocalUserSimulator(
                instructions=task.user_scenario,
                tools=user_tools,
                model=self.user_model,
                tokenizer=self.user_tokenizer,
                temperature=self.config.user_model_temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
        else:
            # 根据 rank 选择 API key
            api_key = self._get_api_key_for_rank()

            # Use GPT41UserSimulator which calls zidongtaichu API via SSE streaming
            user = GPT41UserSimulator(
                instructions=task.user_scenario,
                tools=user_tools,
                api_key=api_key,
            )
            return user

    def _get_api_key_for_rank(self) -> str:
        """根据当前 rank 获取对应的 API key."""
        import os
        rank = int(os.environ.get("LOCAL_RANK", 0))

        # 如果配置了多个 API keys，则按 rank 分配
        if self.config.user_api_keys is not None and len(self.config.user_api_keys) > 0:
            # 循环使用 keys（如果 rank 数量超过 key 数量）
            key_index = rank % len(self.config.user_api_keys)
            api_key = self.config.user_api_keys[key_index]
            if rank == 0:
                print(f"[API Key] Rank {rank} using key index {key_index} (total {len(self.config.user_api_keys)} keys)")
            return api_key
        else:
            # 使用默认的单个 key
            return self.config.user_api_key

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert message history to a prompt string for the policy model."""
        if not messages:
            # Empty history — return empty string; caller will handle this
            return ""
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content or ""})
                elif isinstance(msg, AssistantMessage):
                    content = strip_thinking_content(msg.content or "")
                    chat_messages.append({"role": "assistant", "content": content})
                else:
                    chat_messages.append({"role": "user", "content": msg.content or ""})
            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    parts.append(f"System: {msg.content or ''}")
                elif isinstance(msg, AssistantMessage):
                    parts.append(f"Assistant: {strip_thinking_content(msg.content or '')}")
                else:
                    parts.append(f"User: {msg.content or ''}")
            parts.append("Assistant:")
            return "\n\n".join(parts)

    def compute_log_probs(
        self,
        trajectory: Trajectory,
        use_reference: bool = False,
        max_messages: int = 5,
        max_seq_len: int = 1024,
        grad_messages: int = 3,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute token-level log-probs for assistant messages in trajectory.

        Returns token-level tensors (like ToolRL) instead of a single scalar,
        so the loss can be computed as a proper token-masked mean.

        Returns:
            log_probs  : (total_response_tokens,) — token log-probs with grad
            ref_log_probs: (total_response_tokens,) — ref token log-probs, no grad
            n_tokens   : int — number of valid response tokens
        """
        rank = getattr(self.config, "local_rank", 0)

        assistant_messages = [m for m in trajectory.messages if isinstance(m, AssistantMessage) and m.content]
        if len(assistant_messages) > max_messages:
            assistant_messages = assistant_messages[-max_messages:]

        grad_start_idx = max(0, len(assistant_messages) - grad_messages)

        all_log_probs     = []   # token-level, with grad for grad_messages
        all_ref_log_probs = []   # token-level, no grad

        for msg_idx, msg in enumerate(assistant_messages):
            need_grad = (msg_idx >= grad_start_idx)
            try:
                history = []
                for m in trajectory.messages:
                    if m == msg:
                        break
                    history.append(m)

                prompt      = self._messages_to_prompt(history)
                target_text = (msg.content or "")[:1024]
                if not target_text:
                    continue

                prompt_ids = self.tokenizer.encode(
                    prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
                ).to(self.device)
                full_ids = self.tokenizer.encode(
                    prompt + target_text, return_tensors="pt", truncation=True, max_length=max_seq_len
                ).to(self.device)

                target_ids = full_ids[:, prompt_ids.shape[1]:]
                if prompt_ids.shape[1] >= full_ids.shape[1] or target_ids.shape[1] == 0:
                    continue

                start_idx = prompt_ids.shape[1] - 1
                end_idx   = full_ids.shape[1] - 1
                if start_idx < 0 or end_idx <= start_idx:
                    continue

                # Policy forward
                with torch.set_grad_enabled(need_grad):
                    out = self.model(full_ids)
                tgt_logits = out.logits[:, start_idx:end_idx, :]
                if tgt_logits.shape[1] != target_ids.shape[1]:
                    continue
                tok_lp = F.log_softmax(tgt_logits, dim=-1) \
                          .gather(2, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T,)
                if not need_grad:
                    tok_lp = tok_lp.detach()
                all_log_probs.append(tok_lp)

                # Reference forward (always no_grad)
                ref_model = self.reference_model
                if hasattr(ref_model, "module"):
                    ref_model = ref_model.module
                with torch.no_grad():
                    ref_out = ref_model(full_ids)
                ref_tgt_logits = ref_out.logits[:, start_idx:end_idx, :]
                ref_tok_lp = F.log_softmax(ref_tgt_logits, dim=-1) \
                              .gather(2, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T,)
                all_ref_log_probs.append(ref_tok_lp.detach())

                del out, tgt_logits, ref_out, ref_tgt_logits
                del prompt_ids, full_ids, target_ids

            except Exception as e:
                if debug:
                    import traceback
                    print(f"      [Rank {rank}] msg_idx={msg_idx} exception: {e}")
                    traceback.print_exc()
                continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not all_log_probs:
            zero = torch.zeros(1, device=self.device)
            return zero, zero.detach(), 0

        log_probs     = torch.cat(all_log_probs,     dim=0)  # (total_T,)
        ref_log_probs = torch.cat(all_ref_log_probs, dim=0)  # (total_T,)
        return log_probs, ref_log_probs, log_probs.shape[0]

    def compute_grpo_loss(
        self,
        trajectories: list[Trajectory],
        advantages: torch.Tensor,
        num_forwards: int = None,  # kept for API compat
    ) -> Tuple[torch.Tensor, int]:
        """
        Token-level GRPO loss, aligned with ToolRL's implementation.

        For each trajectory:
          policy_loss = -adv * mean(token_log_probs)   [response tokens only]
          kl_loss     = mean(low_var_kl(log_π, log_π_ref))
          loss        = policy_loss + kl_coef * kl_loss

        Aggregated as masked mean across trajectories (mask=0 if no valid tokens).
        An anchor forward ensures DeepSpeed always has a parameter to reduce.
        """
        rank = getattr(self.config, "local_rank", 0)

        traj_losses = []
        traj_masks  = []

        for traj_idx, (traj, adv) in enumerate(zip(trajectories, advantages)):
            assistant_messages = [m for m in traj.messages if isinstance(m, AssistantMessage) and m.content]

            log_probs, ref_log_probs, n_tokens = self.compute_log_probs(traj, debug=True)

            adv_val = adv.detach().float()

            if n_tokens == 0:
                # No valid tokens — contribute zero loss (parameter-free), mask=0
                traj_losses.append(torch.zeros(1, device=self.device).squeeze())
                traj_masks.append(torch.zeros(1, device=self.device).squeeze())
                print(f"    [Rank {rank}] Traj {traj_idx}: MASKED (no tokens), "
                      f"assistant_msgs={len(assistant_messages)}")
                continue

            # Token-level policy loss: -adv * mean(log_π)
            policy_loss = -adv_val * log_probs.mean()

            # low_var_kl: exp(log_π_ref - log_π) - (log_π_ref - log_π) - 1
            # clamped to [-10, 10] as in ToolRL
            kl_diff = ref_log_probs - log_probs.detach()
            low_var_kl = torch.exp(kl_diff) - kl_diff - 1.0
            low_var_kl = torch.clamp(low_var_kl, -10.0, 10.0)
            kl_loss = low_var_kl.mean()

            traj_loss = policy_loss + self.config.kl_coef * kl_loss
            traj_loss = torch.clamp(traj_loss, -50.0, 50.0)

            is_valid = torch.isfinite(traj_loss)
            mask_val = 1.0 if is_valid else 0.0

            traj_losses.append(traj_loss if is_valid else torch.zeros(1, device=self.device).squeeze())
            traj_masks.append(torch.tensor(mask_val, device=self.device))

            print(f"    [Rank {rank}] Traj {traj_idx}: {'VALID' if is_valid else 'MASKED'} "
                  f"loss={traj_loss.item():.4f}, n_tokens={n_tokens}, "
                  f"policy={policy_loss.item():.4f}, kl={kl_loss.item():.4f}, "
                  f"adv={adv_val.item():.4f}, assistant_msgs={len(assistant_messages)}")

        stacked_loss = torch.stack(traj_losses)
        stacked_mask = torch.stack(traj_masks)
        num_valid    = int(stacked_mask.sum().item())

        print(f"  [Rank {rank}] num_valid={num_valid}/{len(trajectories)}, "
              f"losses={[f'{l.item():.4f}' for l in stacked_loss]}")

        # Anchor: ensures at least one model parameter is in the graph so
        # DeepSpeed ZeRO-2 always has something to reduce.
        anchor = self._dummy_forward()
        loss = stacked_loss.sum() / (stacked_mask.sum() + 1e-6) + anchor

        return loss, num_valid


    def _dummy_forward(self) -> torch.Tensor:
        """做一个 dummy forward，保持所有 rank 同步"""
        dummy_input = torch.tensor([[self.tokenizer.pad_token_id or 0]], device=self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.logits.sum() * 0.0

    def count_forwards_needed(self, trajectories: list[Trajectory], max_messages: int = 5) -> int:
        """计算处理这些 trajectories 需要的 forward 次数"""
        total = 0
        for traj in trajectories:
            assistant_messages = [m for m in traj.messages if isinstance(m, AssistantMessage) and m.content]
            # compute_log_probs 最多处理 max_messages 条消息
            total += min(len(assistant_messages), max_messages) if assistant_messages else 0
        return total

    def reset_optimizer(self):
        """
        重置 optimizer 状态。

        在切换 domain 时调用，清除之前的动量（momentum），
        避免旧 domain 的梯度方向影响新 domain 的训练。
        """
        local_rank = getattr(self.config, "local_rank", 0)
        if local_rank == 0:
            print(f"[PolicyModel] Resetting optimizer state...")

        # 重新创建 optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # 重新创建 scheduler（如果有的话）
        if self.config.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            self.scheduler = None

        if local_rank == 0:
            print(f"[PolicyModel] Optimizer reset complete.")

    def save_checkpoint(self, path: str, model_override=None, save_optimizer: bool = True):
        """
        保存 checkpoint。

        注意：DeepSpeed ZeRO 的 optimizer state 是按 rank 分片的，
        每个 rank 需要保存自己的 optimizer state 到不同文件。
        """
        os.makedirs(path, exist_ok=True)
        rank = getattr(self.config, "local_rank", 0)

        # 模型只需要 rank 0 保存（所有 rank 的模型权重是一样的）
        if rank == 0:
            if model_override is not None:
                model_to_save = model_override
            else:
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            model_to_save.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        # Optimizer state 每个 rank 都要保存（因为 ZeRO 分片）
        if save_optimizer:
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                },
                os.path.join(path, f"optimizer_rank{rank}.pt"),
            )

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        加载 checkpoint。

        注意：DeepSpeed ZeRO 的 optimizer state 是按 rank 分片的，
        每个 rank 需要加载自己的 optimizer state。
        """
        import glob

        local_rank = getattr(self.config, "local_rank", 0)

        # 获取目标模型（unwrap DeepSpeed wrapper）
        target = self.model.module if hasattr(self.model, "module") else self.model

        # 查找权重文件
        safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        bin_files = sorted(glob.glob(os.path.join(path, "model*.bin"))) or sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))

        if local_rank == 0:
            print(f"[load_checkpoint] path={path}")
            print(f"[load_checkpoint] safetensor_files={safetensor_files}")
            print(f"[load_checkpoint] bin_files={bin_files}")

        if safetensor_files:
            # 加载 safetensors 格式
            if local_rank == 0:
                print(f"[load_checkpoint] Loading {len(safetensor_files)} safetensor files...")
            from safetensors.torch import load_file
            state_dict = {}
            for sf in safetensor_files:
                state_dict.update(load_file(sf, device="cpu"))
            # 使用 strict=False 处理 tied weights（如 lm_head.weight）
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
            if local_rank == 0 and missing:
                # 过滤掉 tied weights 的警告
                real_missing = [k for k in missing if not k.endswith('.weight') or 'lm_head' not in k]
                if real_missing:
                    print(f"[load_checkpoint] Warning: Missing keys (non-tied): {real_missing}")
            del state_dict
        elif bin_files:
            # 加载 bin 格式
            if local_rank == 0:
                print(f"[load_checkpoint] Loading {len(bin_files)} bin files...")
            state_dict = {}
            for bf in bin_files:
                state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))
            # 使用 strict=False 处理 tied weights（如 lm_head.weight）
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
            if local_rank == 0 and missing:
                # 过滤掉 tied weights 的警告
                real_missing = [k for k in missing if not k.endswith('.weight') or 'lm_head' not in k]
                if real_missing:
                    print(f"[load_checkpoint] Warning: Missing keys (non-tied): {real_missing}")
            del state_dict
        else:
            # 没有找到权重文件，报错而不是回退到 from_pretrained
            raise FileNotFoundError(
                f"No safetensors or bin files found in {path}. "
                f"Please check the checkpoint directory structure."
            )

        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if local_rank == 0:
            print(f"[load_checkpoint] Checkpoint loaded successfully from {path}")

        if not load_optimizer:
            return

        rank = getattr(self.config, "local_rank", 0)

        # 优先加载按 rank 分片的 optimizer state
        optimizer_path_rank = os.path.join(path, f"optimizer_rank{rank}.pt")
        optimizer_path_legacy = os.path.join(path, "optimizer.pt")

        # 调试信息
        print(f"[load_checkpoint] rank={rank}, optimizer_path_rank={optimizer_path_rank}, exists={os.path.exists(optimizer_path_rank)}")

        if os.path.exists(optimizer_path_rank):
            # 新格式：每个 rank 有自己的 optimizer state
            print(f"[load_checkpoint] rank={rank} loading optimizer from {optimizer_path_rank}")
            ckpt = torch.load(optimizer_path_rank, map_location="cpu", weights_only=False)

            # DeepSpeed ZeRO optimizer 需要特殊处理
            # 检查是否是 DeepSpeed wrapped optimizer
            if hasattr(self.optimizer, 'optimizer') and hasattr(self.optimizer.optimizer, 'param_groups'):
                # 这是 Accelerate wrapped DeepSpeed optimizer
                # 直接加载到底层 optimizer 的 state
                try:
                    # 尝试直接设置 optimizer state（绕过 DeepSpeed 的 load_state_dict）
                    base_optimizer = self.optimizer.optimizer
                    if hasattr(base_optimizer, 'optimizer'):
                        # DeepSpeed ZeRO optimizer
                        base_optimizer = base_optimizer.optimizer

                    saved_state = ckpt["optimizer_state_dict"]
                    # 如果保存的是完整的 DeepSpeed state，提取当前 rank 的部分
                    if isinstance(saved_state, dict) and "optimizer_state_dict" in saved_state:
                        saved_state = saved_state["optimizer_state_dict"]

                    base_optimizer.load_state_dict(saved_state)
                    print(f"[load_checkpoint] rank={rank} optimizer loaded successfully (direct)")
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")
            else:
                # 普通 optimizer
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print(f"[load_checkpoint] rank={rank} optimizer loaded successfully")
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")

            if self.scheduler and ckpt.get("scheduler_state_dict") is not None:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load scheduler state: {e}")
        elif os.path.exists(optimizer_path_legacy):
            # 旧格式：尝试加载，如果失败则跳过
            try:
                ckpt = torch.load(optimizer_path_legacy, map_location=self.device, weights_only=False)
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self.scheduler and ckpt.get("scheduler_state_dict") is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (KeyError, RuntimeError) as e:
                if rank == 0:
                    print(f"[Warning] Could not load legacy optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Single-turn API-Bank rollout
    # ------------------------------------------------------------------

    def generate_single_turn(
        self,
        sample,                      # APIBankSample
        num_samples: int = 1,
        temperature: float = None,
        max_new_tokens: int = None,
    ) -> List["APIBankTrajectory"]:
        """
        Single-turn rollout for API-Bank: system+user → LLM → output_text.

        Runs only on rank 0 (DeepSpeed ZeRO-2 safe), then broadcasts
        the serialised trajectories to all other ranks.

        Returns a list of APIBankTrajectory (length == num_samples).
        """
        import torch.distributed as dist

        rank       = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        temperature = temperature if temperature is not None else self.config.temperature
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

        self.model.eval()
        trajectories: List[APIBankTrajectory] = []

        if rank == 0:
            # Unwrap DeepSpeed / DDP wrapper for generation
            model = self.model
            while hasattr(model, "module"):
                model = model.module

            messages = sample.to_prompt_messages()

            # Build prompt string
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).input_ids.to(self.device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            if temperature > 0:
                gen_kwargs["do_sample"]    = True
                gen_kwargs["temperature"]  = temperature
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                for _ in range(num_samples):
                    out_ids = model.generate(prompt_ids, **gen_kwargs)
                    response_ids = out_ids[0, prompt_ids.shape[1]:]
                    output_text  = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    trajectories.append(APIBankTrajectory(
                        sample_id=sample.id,
                        prompt=prompt,
                        output_text=output_text,
                        prompt_ids=prompt_ids[0].cpu(),
                        response_ids=response_ids.cpu(),
                        gold_tool_calls=sample.gold_tool_calls,
                    ))

            print(f"  [Rank 0] Generated {len(trajectories)} single-turn trajectories for {sample.id}", flush=True)

        # Broadcast from rank 0 to all other ranks
        if world_size > 1:
            payload = pickle.dumps(trajectories) if rank == 0 else b""
            length_t = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
            dist.broadcast(length_t, src=0)
            if rank != 0:
                payload = bytes(length_t.item())
            max_len = length_t.item()
            if rank == 0:
                padded_bytes = bytes(payload) + bytes(max_len - len(payload))
                payload_t = torch.tensor(list(padded_bytes), dtype=torch.uint8, device=self.device)
            else:
                payload_t = torch.zeros(max_len, dtype=torch.uint8, device=self.device)
            dist.broadcast(payload_t, src=0)
            if rank != 0:
                trajectories = pickle.loads(payload_t.cpu().numpy().tobytes())

        return trajectories

    def compute_log_probs_single_turn(
        self,
        traj: "APIBankTrajectory",
        max_seq_len: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Token-level log-probs for a single-turn APIBankTrajectory.
        Only computes policy log_probs (ref is handled separately in
        compute_grpo_loss_toolrl via load/offload pattern).

        Returns:
            log_probs     : (T,) with grad
            dummy         : (T,) zeros (ref handled externally)
            n_tokens      : int
        """
        prompt_text   = traj.prompt
        response_text = traj.output_text

        if not response_text:
            z = torch.zeros(1, device=self.device)
            return z, z.detach(), 0

        prompt_ids = self.tokenizer.encode(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(self.device)
        full_ids = self.tokenizer.encode(
            prompt_text + response_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).to(self.device)

        n_prompt = prompt_ids.shape[1]
        n_full   = full_ids.shape[1]
        if n_prompt >= n_full:
            z = torch.zeros(1, device=self.device)
            return z, z.detach(), 0

        target_ids = full_ids[:, n_prompt:]
        start_idx  = n_prompt - 1
        end_idx    = n_full   - 1

        # Unwrap DeepSpeed engine to call HF model directly.
        inner_model = self.model
        while hasattr(inner_model, "module"):
            inner_model = inner_model.module

        # Debug: check where params actually are
        first_param = next(inner_model.parameters())
        if first_param.device.type == "cpu":
            print(f"[WARNING rank={getattr(self.config,'local_rank',0)}] "
                  f"policy model params are on CPU! Moving to {self.device}...",
                  flush=True)
            inner_model.to(self.device)

        out = inner_model(full_ids)
        tgt_logits = out.logits[:, start_idx:end_idx, :]
        log_probs  = (
            F.log_softmax(tgt_logits, dim=-1)
            .gather(2, target_ids.unsqueeze(-1))
            .squeeze(-1)
            .squeeze(0)
        )  # (T,)

        del out, tgt_logits, full_ids, prompt_ids, target_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return log_probs, torch.zeros_like(log_probs).detach(), log_probs.shape[0]

    def compute_grpo_loss_single_turn(
        self,
        trajectories: List["APIBankTrajectory"],
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Token-level GRPO loss for single-turn trajectories.
        Identical math to compute_grpo_loss() but uses compute_log_probs_single_turn().
        """
        rank = getattr(self.config, "local_rank", 0)
        traj_losses, traj_masks = [], []

        for i, (traj, adv) in enumerate(zip(trajectories, advantages)):
            log_probs, ref_log_probs, n_tokens = self.compute_log_probs_single_turn(traj)
            adv_val = adv.detach().float()

            if n_tokens == 0:
                traj_losses.append(torch.zeros(1, device=self.device).squeeze())
                traj_masks.append(torch.zeros(1, device=self.device).squeeze())
                print(f"    [Rank {rank}] Traj {i}: MASKED (no tokens)")
                continue

            policy_loss = -adv_val * log_probs.mean()

            kl_diff   = ref_log_probs - log_probs.detach()
            low_var_kl = torch.exp(kl_diff) - kl_diff - 1.0
            kl_loss    = torch.clamp(low_var_kl, -10.0, 10.0).mean()

            traj_loss = policy_loss + self.config.kl_coef * kl_loss
            traj_loss = torch.clamp(traj_loss, -50.0, 50.0)

            is_valid = torch.isfinite(traj_loss)
            traj_losses.append(traj_loss if is_valid else torch.zeros(1, device=self.device).squeeze())
            traj_masks.append(torch.tensor(1.0 if is_valid else 0.0, device=self.device))

            print(f"    [Rank {rank}] Traj {i}: loss={traj_loss.item():.4f} "
                  f"policy={policy_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"adv={adv_val.item():.4f} n_tok={n_tokens}")

        stacked_loss = torch.stack(traj_losses)
        stacked_mask = torch.stack(traj_masks)
        num_valid    = int(stacked_mask.sum().item())

        anchor = self._dummy_forward()
        loss   = stacked_loss.sum() / (stacked_mask.sum() + 1e-6) + anchor
        return loss, num_valid

    # ------------------------------------------------------------------
    # ToolRL-style multi-prompt parallel generation
    # ------------------------------------------------------------------

    def generate_group(
        self,
        tasks: list,                 # List[Task], length = batch_size
        environments: list,          # List[Environment], same length
        domain: str,
        G: int = None,               # responses per prompt (num_samples_per_prompt)
        temperature: float = None,
    ) -> tuple:
        """
        Generate G trajectories for each of B prompts, with each GPU
        handling its own slice of prompts in parallel.

        Returns:
            trajectories : List[Trajectory], length = B * G, ordered as
                           [task0_g0, task0_g1, ..., task1_g0, task1_g1, ...]
            old_log_probs: List[Tensor(T,)], one per trajectory (CPU, no grad)
            prompt_uids  : List[str], length = B * G — same uid for all G
                           responses of the same prompt (used for GRPO grouping)
        """
        import pickle as _pickle
        import uuid
        import torch.distributed as dist

        rank       = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        G          = G if G is not None else self.config.num_samples_per_prompt
        temperature = temperature if temperature is not None else self.config.temperature
        B          = len(tasks)

        self.model.eval()

        # Each rank handles its own slice of prompts
        my_indices = list(range(rank, B, world_size))

        local_trajs: list = []
        local_olps:  list = []   # old_log_probs per trajectory
        local_uids:  list = []

        model = self.model
        while hasattr(model, "module"):
            model = model.module

        for idx in my_indices:
            task = tasks[idx]
            env  = environments[idx]
            uid  = str(uuid.uuid4())

            for _ in range(G):
                try:
                    agent = self._create_agent(env, domain)
                    user  = self._create_user(task, domain)
                    from AGentCL.orchestrator.orchestrator import Orchestrator
                    orch = Orchestrator(
                        domain=domain, agent=agent, user=user,
                        environment=env, task=task, max_steps=50,
                    )
                    orch.initialize()
                    sim = orch.run()
                    from .reward_oracle import Trajectory
                    traj = Trajectory(
                        task_id=task.id,
                        messages=sim.messages,
                        termination_reason=sim.termination_reason,
                        cost=0.0,
                    )
                    # Compute old_log_probs immediately (no grad)
                    with torch.no_grad():
                        lp, _, n = self.compute_log_probs(traj)
                    local_trajs.append(traj)
                    local_olps.append(lp.cpu() if n > 0 else torch.zeros(1))
                    local_uids.append(uid)
                except Exception as e:
                    print(f"  [Rank {rank}] generate_group idx={idx} failed: {e}", flush=True)

        # All-gather across ranks
        if world_size > 1:
            payload = _pickle.dumps((local_trajs, local_olps, local_uids))
            length_t = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
            all_lengths = [torch.zeros(1, dtype=torch.long, device=self.device)
                           for _ in range(world_size)]
            dist.all_gather(all_lengths, length_t)

            max_len = max(l.item() for l in all_lengths)
            padded_bytes = bytes(payload) + bytes(max_len - len(payload))
            payload_t = torch.tensor(list(padded_bytes), dtype=torch.uint8, device=self.device)
            all_payloads = [torch.zeros(max_len, dtype=torch.uint8, device=self.device)
                            for _ in range(world_size)]
            dist.all_gather(all_payloads, payload_t)

            all_trajs, all_olps, all_uids = [], [], []
            for r, (pl_t, pl_len) in enumerate(zip(all_payloads, all_lengths)):
                t, o, u = _pickle.loads(pl_t[:pl_len.item()].cpu().numpy().tobytes())
                all_trajs.extend(t)
                all_olps.extend(o)
                all_uids.extend(u)
        else:
            all_trajs, all_olps, all_uids = local_trajs, local_olps, local_uids

        if rank == 0:
            print(f"  [generate_group] {B} prompts × {G} = {B*G} expected, "
                  f"got {len(all_trajs)} trajectories", flush=True)

        return all_trajs, all_olps, all_uids

    def compute_grpo_loss_toolrl(
        self,
        trajectories: list,
        advantages: "torch.Tensor",        # (N,) scalar per trajectory
        old_log_probs: list,               # List[Tensor(T_i,)] from vLLM
        token_level_rewards: list,         # List[Tensor(T_i,)] token-level rewards
        accelerator=None,
    ) -> dict:
        """
        Exact ToolRL GRPO loss (core_algos.compute_policy_loss).

        Token-level pipeline (mirrors ToolRL exactly):
          1. Forward → new_log_prob (T,), logits (T, V)
          2. eos_mask: 1 for all real response tokens
          3. advantage broadcast: scalar → (T,) * eos_mask
          4. ratio = exp(new_lp - old_lp)  [NO CLAMP]
          5. pg_loss = masked_mean(max(-A*r, -A*clip(r, 1-ε, 1+ε)))
          6. entropy = masked_mean(-sum(p*log_p))
          7. ppo_kl = masked_mean(log_ratio)  [ToolRL style]
          8. loss = pg_loss - ent_coeff * entropy
        """
        clip_range = getattr(self.config, "clip_range", 0.2)
        ent_coeff  = getattr(self.config, "entropy_coeff", 0.001)
        mini_bsz   = getattr(self.config, "ppo_mini_batch_size", 128)
        micro_bsz  = getattr(self.config, "ppo_micro_batch_size", 32)
        ppo_epochs = getattr(self.config, "ppo_epochs", 1)
        max_grad   = getattr(self.config, "max_grad_norm", 1.0)

        N = len(trajectories)
        if N == 0:
            return {"pg_loss": 0.0, "pg_clipfrac": 0.0, "kl": 0.0,
                    "entropy": 0.0, "grad_norm": 0.0}

        # Build (full_ids, n_prompt) from vLLM token ids — no re-tokenization
        traj_inputs: list = []
        inner_model = self.model
        while hasattr(inner_model, "module"):
            inner_model = inner_model.module

        for i in range(N):
            traj = trajectories[i]
            if not traj.output_text or traj.response_ids is None or len(traj.response_ids) == 0:
                traj_inputs.append(None)
                continue
            prompt_ids_1d = traj.prompt_ids.long()         # (P,)
            resp_ids_1d   = traj.response_ids.long()       # (R,)
            full_ids_1d   = torch.cat([prompt_ids_1d, resp_ids_1d], dim=0)  # (P+R,)
            n_prompt      = prompt_ids_1d.shape[0]
            if n_prompt >= full_ids_1d.shape[0]:
                traj_inputs.append(None)
            else:
                traj_inputs.append((full_ids_1d.unsqueeze(0), n_prompt))

        import random as _random
        indices = list(range(N))
        all_metrics = {"pg_loss": [], "pg_clipfrac": [], "kl": [],
                       "entropy": [], "grad_norm": []}

        for _epoch in range(ppo_epochs):
            _random.shuffle(indices)

            for mb_start in range(0, N, mini_bsz):
                mb_idx = indices[mb_start: mb_start + mini_bsz]
                if not mb_idx:
                    continue

                grad_accum = max(1, len(mb_idx) // micro_bsz)
                self.policy_optimizer_zero_grad()

                for mc_start in range(0, len(mb_idx), micro_bsz):
                    mc_idx = mb_idx[mc_start: mc_start + micro_bsz]
                    mc_pg, mc_clip, mc_kl, mc_ent = [], [], [], []

                    for i in mc_idx:
                        if traj_inputs[i] is None:
                            continue

                        full_ids, n_prompt = traj_inputs[i]
                        full_ids_gpu = full_ids.to(self.device)
                        target_ids   = full_ids_gpu[:, n_prompt:]   # (1, R)

                        # ── Forward pass ──────────────────────────────────────
                        self.model.train()
                        out        = inner_model(full_ids_gpu)
                        tgt_logits = out.logits[
                            :, n_prompt - 1: full_ids_gpu.shape[1] - 1, :]  # (1, R, V)

                        # new log_probs: (R,)
                        new_lp = (F.log_softmax(tgt_logits, dim=-1)
                                  .gather(2, target_ids.unsqueeze(-1))
                                  .squeeze(-1).squeeze(0))

                        # old log_probs from actor re-forward: align length
                        old_lp  = old_log_probs[i].to(self.device)
                        min_len = min(new_lp.shape[0], old_lp.shape[0])
                        new_lp_a = new_lp[:min_len]
                        old_lp_a = old_lp[:min_len].detach()

                        # eos_mask: 1 for all real response tokens
                        eos_mask = torch.ones(min_len, device=self.device)

                        # advantage broadcast: scalar → (T,) * eos_mask (ToolRL style)
                        adv_tok = advantages[i].float().to(self.device).expand(min_len) * eos_mask

                        # ── PPO clip loss (ToolRL compute_policy_loss) ────────
                        # NO CLAMP on log_ratio (ToolRL original)
                        log_ratio = new_lp_a - old_lp_a
                        ratio     = torch.exp(log_ratio)
                        pg_loss1  = -adv_tok * ratio
                        pg_loss2  = -adv_tok * torch.clamp(
                            ratio, 1.0 - clip_range, 1.0 + clip_range)
                        pg_loss_i  = (torch.max(pg_loss1, pg_loss2) * eos_mask
                                      ).sum() / eos_mask.sum()
                        clipfrac_i = ((pg_loss2 > pg_loss1).float() * eos_mask
                                      ).sum() / eos_mask.sum()

                        # ── Entropy (ToolRL compute_entropy_loss) ─────────────
                        probs     = torch.softmax(tgt_logits.squeeze(0), dim=-1)   # (R, V)
                        log_probs = F.log_softmax(tgt_logits.squeeze(0), dim=-1)
                        entropy_i = -(probs * log_probs).sum(dim=-1)               # (R,)
                        entropy_i = (entropy_i[:min_len] * eos_mask).sum() / eos_mask.sum()

                        # ── KL (ToolRL ppo_kl = masked_mean(log_ratio)) ───────
                        with torch.no_grad():
                            ppo_kl_i = (log_ratio * eos_mask).sum() / eos_mask.sum()

                        # ── loss = pg_loss - ent_coeff * entropy ──────────────
                        loss_i = (pg_loss_i - ent_coeff * entropy_i) / grad_accum

                        # Skip NaN/inf loss to prevent weight corruption
                        if torch.isnan(loss_i) or torch.isinf(loss_i):
                            print(f"[compute_grpo_loss] NaN/inf loss at traj {i}, skipping",
                                  flush=True)
                            continue

                        if accelerator is not None:
                            accelerator.backward(loss_i)
                        else:
                            loss_i.backward()

                        mc_pg.append(pg_loss_i.detach().item())
                        mc_clip.append(clipfrac_i.detach().item())
                        mc_kl.append(ppo_kl_i.detach().item())
                        mc_ent.append(entropy_i.detach().item())

                        del out, tgt_logits, new_lp, full_ids_gpu, target_ids
                        del probs, log_probs, ratio, pg_loss1, pg_loss2

                    if mc_pg:
                        all_metrics["pg_loss"].append(float(sum(mc_pg)    / len(mc_pg)))
                        all_metrics["pg_clipfrac"].append(float(sum(mc_clip) / len(mc_clip)))
                        all_metrics["kl"].append(float(sum(mc_kl)    / len(mc_kl)))
                        all_metrics["entropy"].append(float(sum(mc_ent)   / len(mc_ent)))

                # Optimizer step after mini-batch
                if accelerator is not None:
                    grad_norm = accelerator.clip_grad_norm_(
                        self.model.parameters(), max_grad)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad)
                self.policy_optimizer_step()
                all_metrics["grad_norm"].append(
                    float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))

        return {k: float(sum(v) / len(v)) if v else 0.0
                for k, v in all_metrics.items()}

    # ------------------------------------------------------------------
    # ToolRL-aligned GRPO loss (replaces compute_grpo_loss_toolrl above)
    # ------------------------------------------------------------------

    def policy_optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def policy_optimizer_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

@dataclass
class APIBankTrajectory:
    """Holds everything produced by one single-turn rollout."""
    sample_id:       str
    prompt:          str                  # full prompt string fed to the model
    output_text:     str                  # raw model output — used for PPO re-forward (matches resp_ids)
    prompt_ids:      torch.Tensor         # CPU tensor, shape (prompt_len,)
    response_ids:    torch.Tensor         # CPU tensor, shape (response_len,)
    gold_tool_calls: List[dict]           # ground-truth tool calls
    reward_text:     str = ""             # stripped output for reward computation (no <think>)
    ref_log_probs:   Optional[torch.Tensor] = None  # CPU tensor (T,), pre-computed
    reward:          float = 0.0          # filled in after reward computation
    reward_info:     object = None        # RewardInfo object
