"""
Training script for GRPO on API-Bank single-turn tool-use tasks (ToolRL style).

Usage (single GPU):
    python src/tau2/scripts/train_apibank.py \
        --model_name_or_path /path/to/model \
        --num_steps_per_task 500

Usage (multi-GPU with DeepSpeed):
    accelerate launch --config_file accelerate_config.yaml \
        src/tau2/scripts/train_apibank.py \
        --model_name_or_path /path/to/model
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning.config import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train with GRPO on API-Bank (ToolRL style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--temperature",    type=float, default=1.0)    # ToolRL: 1.0
    parser.add_argument("--max_new_tokens", type=int,   default=1024)   # ToolRL: 1024

    # API-Bank data
    parser.add_argument("--levels",      type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (debug only)")

    # GRPO — ToolRL defaults
    parser.add_argument("--num_samples_per_prompt", type=int,   default=4,      # rollout.n=4
                        help="G: responses per prompt")
    parser.add_argument("--batch_size_per_gpu",     type=int,   default=64,     # 512/8GPUs
                        help="Prompts per step per GPU")
    parser.add_argument("--kl_coef",                type=float, default=0.001)  # kl_ctrl.kl_coef
    parser.add_argument("--clip_range",             type=float, default=0.2)    # clip_ratio
    parser.add_argument("--entropy_coeff",          type=float, default=0.001)  # entropy_coeff
    parser.add_argument("--ppo_mini_batch_size",    type=int,   default=128)    # ppo_mini_batch_size
    parser.add_argument("--ppo_micro_batch_size",   type=int,   default=32)
    parser.add_argument("--ppo_epochs",             type=int,   default=1)      # ppo_epochs
    parser.add_argument("--num_steps_per_task",     type=int,   default=500)
    parser.add_argument("--learning_rate",          type=float, default=1e-6)   # actor.optim.lr
    parser.add_argument("--warmup_steps",           type=int,   default=10)
    parser.add_argument("--max_grad_norm",          type=float, default=1.0)

    # Early stopping
    parser.add_argument("--early_stopping_patience",  type=int,   default=20)
    parser.add_argument("--early_stopping_threshold", type=float, default=3.5)

    # Logging
    parser.add_argument("--log_dir",               type=str,  default="logs/apibank")
    parser.add_argument("--eval_interval",         type=int,  default=50)
    parser.add_argument("--checkpoint_interval",   type=int,  default=100)
    parser.add_argument("--skip_intermediate_eval", action="store_true", default=False)

    # W&B
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode",     type=str, default=None)

    # vLLM rollout (H20 96GB: vLLM 40% + DeepSpeed 60%)
    parser.add_argument("--vllm_tensor_parallel_size",    type=int,   default=8)
    parser.add_argument("--vllm_gpu_memory_utilization",  type=float, default=0.4)
    parser.add_argument("--vllm_enforce_eager",           action="store_true", default=False)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Misc
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.verbose:
        import logging
        from loguru import logger
        logger.disable("AGentCL")
        logging.basicConfig(level=logging.WARNING)

    config = GRPOConfig(
        # Model
        model_name_or_path=args.model_name_or_path,
        model_dtype=args.model_dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        # GRPO — ToolRL aligned
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
        # Early stopping
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        # Logging
        log_dir=args.log_dir,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        skip_intermediate_eval=args.skip_intermediate_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        # vLLM
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        # W&B
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        # Placeholder (not used in API-Bank mode)
        task_order=["airline"],
        train_split=0.9,
    )

    trainer = GRPOTrainer(config)
    try:
        trainer.train_apibank(
            levels=args.levels,
            max_samples=args.max_samples,
        )
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
