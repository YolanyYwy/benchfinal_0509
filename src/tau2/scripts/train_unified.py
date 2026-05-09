"""
Unified GRPO training driver for rlla_by_domain data.

One entrypoint, one process, one accelerate launch — supports all modes:
  --mode single      每个 domain 前 reload 原始模型 + reset optimizer
  --mode sequential  domain 间权重自然接力（baseline）
  --mode replay      启用 ReplayCL（从 trajectory buffer 采 old domain 样本混入）
  --mode adaptive_replay
  --mode ewc         启用 EWCCL（Fisher 信息矩阵做正则）
  --mode online_ewc
  --mode ewc_pp
  --mode fusion      启用 ModelFusionCL（每个 task 完权重融合）
  --mode adaptive_fusion
  --mode progressive 启用 ProgressiveNetsCL
  --mode dynamic_expansion

DS model + vLLM engine 只初始化一次，所有 domain 共用；避免每个 domain
都重新启动 accelerate 的昂贵开销（加载模型 + vLLM profiling 可能 1-2 分钟/domain）。

Usage:
    accelerate launch --config_file accelerate_1gpu_bf16.yaml \\
        src/tau2/scripts/train_unified.py \\
        --mode sequential \\
        --domains bamboogle api_bank airline \\
        --data_root data/rlla_by_domain \\
        --model_name_or_path /path/to/base/model \\
        --log_dir /9950backfile/ywy/logs/unified
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning.config import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


SUPPORTED_MODES = [
    "single",
    "sequential",
    "replay", "adaptive_replay",
    "ewc", "online_ewc", "ewc_pp",
    "fusion", "adaptive_fusion",
    "progressive", "dynamic_expansion",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified GRPO training driver (single process across all domains)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", type=str, required=True, choices=SUPPORTED_MODES,
                   help="Training mode: single (independent per-domain), sequential (接力), "
                        "or any CL algorithm name")

    # Data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--domains", type=str, nargs="+", required=True)
    p.add_argument("--max_samples", type=int, default=None)

    # Model
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Base model path. For --mode single, this is reloaded before each domain.")
    p.add_argument("--model_dtype", type=str, default="bfloat16")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int,   default=1024)

    # GRPO
    p.add_argument("--num_samples_per_prompt", type=int,   default=4)
    p.add_argument("--batch_size_per_gpu",     type=int,   default=16)
    p.add_argument("--kl_coef",                type=float, default=0.001)
    p.add_argument("--clip_range",             type=float, default=0.2)
    p.add_argument("--entropy_coeff",          type=float, default=0.001)
    p.add_argument("--ppo_mini_batch_size",    type=int,   default=32)
    p.add_argument("--ppo_micro_batch_size",   type=int,   default=8)
    p.add_argument("--ppo_epochs",             type=int,   default=1)
    p.add_argument("--num_steps_per_task",     type=int,   default=500)
    p.add_argument("--learning_rate",          type=float, default=1e-6)
    p.add_argument("--warmup_steps",           type=int,   default=0)
    p.add_argument("--max_grad_norm",          type=float, default=1.0)

    # Early stopping
    p.add_argument("--early_stopping_patience",  type=int,   default=0)
    p.add_argument("--early_stopping_threshold", type=float, default=3.5)

    # Logging
    p.add_argument("--log_dir",              type=str, default="logs/unified")
    p.add_argument("--eval_interval",        type=int, default=50)
    p.add_argument("--checkpoint_interval",  type=int, default=0,
                   help="0 = disable intra-domain checkpoints, only save final per-domain model")
    p.add_argument("--skip_intermediate_eval", action="store_true")

    # vLLM
    p.add_argument("--vllm_tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.4)
    p.add_argument("--vllm_enforce_eager",          action="store_true")

    # W&B
    p.add_argument("--wandb_project",  type=str, default="rlla_unified")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode",     type=str, default=None)

    # Resume (skip first N domains — useful if first K already finished)
    p.add_argument("--skip_first_n", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mode_to_cl_algorithm(mode: str) -> str:
    """Map --mode value to config.cl_algorithm name understood by GRPOTrainer."""
    if mode == "single":
        # single 模式下我们自己手动 reload 权重，CL 算法用 sequential 即可（不做任何事）
        return "sequential"
    return mode  # sequential / replay / ewc / fusion / progressive / ... 都直接透传


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    missing = [d for d in args.domains if not (data_root / d).is_dir()]
    if missing:
        print(f"[ERROR] Data dirs not found: {missing}")
        sys.exit(1)

    domains_to_run = args.domains[args.skip_first_n:]
    if args.skip_first_n > 0:
        print(f"[Resume] Skipping first {args.skip_first_n} domain(s): "
              f"{args.domains[:args.skip_first_n]}")

    print(f"[Unified] mode={args.mode}  cl_algorithm={mode_to_cl_algorithm(args.mode)}")
    print(f"[Unified] Will train {len(domains_to_run)} domains: {domains_to_run}")

    # ── Build trainer ONCE ───────────────────────────────────────────────────
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        model_dtype=args.model_dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        num_samples_per_prompt=args.num_samples_per_prompt,
        batch_size_per_gpu=args.batch_size_per_gpu,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        entropy_coeff=args.entropy_coeff,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size=args.ppo_micro_batch_size,
        ppo_epochs=args.ppo_epochs,
        num_steps_per_task=args.num_steps_per_task,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        log_dir=args.log_dir,  # 每个 domain 会覆盖
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        skip_intermediate_eval=args.skip_intermediate_eval,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        task_order=domains_to_run,
        train_split=0.9,
        cl_algorithm=mode_to_cl_algorithm(args.mode),
    )

    trainer = GRPOTrainer(config)

    # 记住原始模型路径，single 模式每个 domain 前 reload 用
    base_model_path = args.model_name_or_path

    try:
        for i, domain in enumerate(domains_to_run):
            data_dir = str(data_root / domain)

            # Per-domain log dir
            domain_log_dir = Path(args.log_dir) / domain
            domain_log_dir.mkdir(parents=True, exist_ok=True)
            trainer.config.log_dir = str(domain_log_dir)
            trainer.config.task_order = [domain]

            # Per-domain wandb run name（每个 domain 一个独立 run，指标不混淆）
            if trainer.is_main_process() and trainer.metrics is not None:
                # 如果上一个 domain 结束时没关 wandb，就先关掉再起新 run
                if trainer.metrics._wandb_inited:
                    trainer.metrics.wandb.finish()
                    trainer.metrics._wandb_inited = False
                # 更新 run name 并重新 init
                per_domain_run = f"{args.mode}_{domain}_{i}"
                trainer.config.wandb_run_name = per_domain_run
                trainer.metrics.init_wandb()

            if trainer.is_main_process():
                print(f"\n{'='*70}")
                print(f"[{args.mode.upper()} {i+1}/{len(domains_to_run)}] Domain: {domain}")
                print(f"  data_dir: {data_dir}")
                print(f"  log_dir:  {domain_log_dir}")
                print(f"{'='*70}\n")

            # ── single 模式：每个 domain 前重置权重和 optimizer ──
            if args.mode == "single" and i > 0:
                if trainer.is_main_process():
                    print(f"[single] Reloading base weights from {base_model_path}")
                trainer.policy.load_checkpoint(base_model_path, load_optimizer=False)
                trainer.policy.reset_optimizer()
                trainer.accelerator.wait_for_everyone()

            # train_rlla 里我们让它自己关 wandb（close_metrics=True），
            # 下一轮循环顶部会再 init 一个新 run
            trainer.train_rlla(
                data_dir=data_dir,
                max_samples=args.max_samples,
                close_metrics=True,
                domain=domain,
            )

            if trainer.is_main_process():
                print(f"\n[{args.mode.upper()}] Finished domain {domain}.")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
