"""
vLLM worker subprocess — runs vLLM generation in a clean process
with no DeepSpeed/torch.distributed interference.

Called by vllm_generator.vllm_generate() via subprocess.
"""

import argparse
import os
import pickle
import sys
import uuid
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path",   required=True)
    parser.add_argument("--output_path",    required=True)
    parser.add_argument("--model_path",     required=True)
    parser.add_argument("--G",              type=int,   default=4)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int,   default=1024)
    parser.add_argument("--max_model_len",  type=int,   default=3072)
    parser.add_argument("--tp",             type=int,   default=8)
    parser.add_argument("--mem",            type=float, default=0.9)
    parser.add_argument("--dtype",          type=str,   default="bfloat16")
    parser.add_argument("--enforce_eager",  action="store_true", default=False)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load samples
    with open(args.samples_path, "rb") as f:
        samples = pickle.load(f)

    print(f"[vllm_worker] {len(samples)} samples × G={args.G} = {len(samples)*args.G} seqs",
          flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # vLLM engine — runs in this clean subprocess, no distributed conflict
    engine = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    # Sampling params — logprobs=0 returns the chosen token's logprob
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=0,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Build prompts: each sample repeated G times
    prompts:    list = []
    sample_idx: list = []
    uids:       list = []
    uid_per_sample = [str(uuid.uuid4()) for _ in samples]

    for i, sample in enumerate(samples):
        messages = sample.to_prompt_messages()
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        for _ in range(args.G):
            prompts.append(prompt)
            sample_idx.append(i)
            uids.append(uid_per_sample[i])

    # Generate
    print(f"[vllm_worker] Starting generation ({len(prompts)} seqs)...", flush=True)
    import time
    t0 = time.time()
    outputs = engine.generate(prompts=prompts, sampling_params=sampling_params,
                               use_tqdm=True)
    print(f"[vllm_worker] Generation done in {time.time()-t0:.1f}s", flush=True)

    # Build results
    import torch
    from tau2.continual_learning.policy_model import APIBankTrajectory

    print(f"[vllm_worker] Parsing {len(outputs)} outputs...", flush=True)
    trajectories = []
    old_log_probs = []

    for j, (out, s_idx) in enumerate(zip(outputs, sample_idx)):
        sample      = samples[s_idx]
        output_text = out.outputs[0].text
        resp_ids    = list(out.outputs[0].token_ids)

        if j % 50 == 0:
            print(f"[vllm_worker] Parsing {j}/{len(outputs)} "
                  f"| sample={sample.id} "
                  f"| resp_len={len(resp_ids)} "
                  f"| text[:60]={output_text[:60]!r}", flush=True)

        # old_log_probs from vLLM logprobs
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

        prompt_ids = tokenizer.encode(prompts[j], return_tensors="pt")[0]

        traj = APIBankTrajectory(
            sample_id=sample.id,
            prompt=prompts[j],
            output_text=output_text,
            prompt_ids=prompt_ids.cpu(),
            response_ids=torch.tensor(resp_ids, dtype=torch.long),
            gold_tool_calls=sample.gold_tool_calls,
        )
        trajectories.append(traj)
        old_log_probs.append(old_lp)

    print(f"[vllm_worker] Done. {len(trajectories)} trajectories.", flush=True)

    # Save results
    with open(args.output_path, "wb") as f:
        pickle.dump((trajectories, old_log_probs, uids), f)

    print(f"[vllm_worker] Results saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
