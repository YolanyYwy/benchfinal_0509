"""
Training script for ToolRL-style GRPO on rlla_by_domain data.

Usage (single domain):
    accelerate launch --config_file accelerate_8gpu_bf16.yaml \\
        src/tau2/scripts/train_rlla.py \\
        --data_dir data/rlla_by_domain/airline

Usage (all domains sequentially):
    for domain in airline retail api_bank bamboogle; do
        accelerate launch ... train_rlla.py --data_dir data/rlla_by_domain/$domain
    done
"""

import argparse
import os
import sys
from pathlib import Path

# Force vLLM V0 engine before any vLLM import — V1 spawns EngineCore subprocesses
# that deadlock when 8 DeepSpeed ranks all initialise NCCL simultaneously.
os.environ.setdefault("VLLM_USE_V1", "0")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning.config import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with GRPO on rlla_by_domain (ToolRL style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to one domain dir, e.g. data/rlla_by_domain/airline")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training samples (debug only)")

    # Model
    parser.add_argument("--model_name_or_path", type=str,
                        default="/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int,   default=1024)

    # GRPO — ToolRL defaults
    parser.add_argument("--num_samples_per_prompt", type=int,   default=4)
    parser.add_argument("--batch_size_per_gpu",     type=int,   default=64)
    parser.add_argument("--kl_coef",                type=float, default=0.001)
    parser.add_argument("--clip_range",             type=float, default=0.2)
    parser.add_argument("--entropy_coeff",          type=float, default=0.001)
    parser.add_argument("--ppo_mini_batch_size",    type=int,   default=128)
    parser.add_argument("--ppo_micro_batch_size",   type=int,   default=32)
    parser.add_argument("--ppo_epochs",             type=int,   default=1)
    parser.add_argument("--num_steps_per_task",     type=int,   default=500)
    parser.add_argument("--learning_rate",          type=float, default=1e-6)
    parser.add_argument("--warmup_steps",           type=int,   default=0)
    parser.add_argument("--max_grad_norm",          type=float, default=1.0)

    # Early stopping
    parser.add_argument("--early_stopping_patience",  type=int,   default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=3.5)

    # Logging
    parser.add_argument("--log_dir",              type=str, default="logs/rlla")
    parser.add_argument("--eval_interval",        type=int, default=50)
    parser.add_argument("--checkpoint_interval",  type=int, default=100)
    parser.add_argument("--skip_intermediate_eval", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # vLLM
    parser.add_argument("--vllm_tensor_parallel_size",   type=int,   default=8)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_enforce_eager",          action="store_true")

    # W&B
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode",     type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    domain_name = Path(args.data_dir).name
    run_name = args.wandb_run_name or f"rlla_{domain_name}"

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
        log_dir=args.log_dir,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        skip_intermediate_eval=args.skip_intermediate_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        wandb_mode=args.wandb_mode,
        task_order=[domain_name],
        train_split=0.9,
    )

    trainer = GRPOTrainer(config)
    try:
        trainer.train_rlla(
            data_dir=args.data_dir,
            max_samples=args.max_samples,
        )
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
