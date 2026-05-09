"""
单任务训练脚本：在每个 domain 上单独训练模型并保存权重。

用于计算 FWT 的 baseline：
FWT_j = 顺序训练完任务 1~j 后在任务 j 上的性能 - 只在任务 j 上单独训练后的性能

支持多卡训练（DeepSpeed ZeRO-2）

使用方法：
    # 单卡
    python train_single_domain.py --domain airline --output_dir ./single_domain_checkpoints

    # 多卡（通过 accelerate）
    accelerate launch --config_file accelerate_config.yaml train_single_domain.py \
        --domain airline --output_dir ./single_domain_checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train single domain model for FWT baseline (supports multi-GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # User model configuration
    parser.add_argument("--user_model", type=str, default="gpt_oss_120b")
    parser.add_argument("--use_local_user_model", action="store_true", default=False)
    parser.add_argument("--user_model_temperature", type=float, default=0.0)
    parser.add_argument("--user_api_base", type=str, default="https://cloud.zidongtaichu.com/maas/v1")
    parser.add_argument("--user_api_key", type=str, default="vp8ggmuy102xmtpcyf9enr3g")
    parser.add_argument("--user_api_keys", type=str, default=None, help="Multiple API keys separated by space")

    # Training
    parser.add_argument("--domain", type=str, required=True,
                        help="Single domain to train (e.g., airline, retail, telecom)")
    parser.add_argument("--output_dir", type=str, default="./single_domain_checkpoints",
                        help="Directory to save single-domain checkpoints")
    parser.add_argument("--num_steps_per_task", type=int, default=100)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # GRPO
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--num_eval_tasks", type=int, default=5)
    parser.add_argument("--num_eval_samples", type=int, default=1)
    parser.add_argument("--pass_at_k", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=50)

    # W&B
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tasks_per_domain", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)

    # Optimization
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--no_flash_attention", action="store_false", dest="use_flash_attention")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")

    # Checkpoint and resume
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Save checkpoint every N steps")
    parser.add_argument("--skip_intermediate_eval", action="store_true", default=False,
                        help="Skip intermediate evaluation during training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from step checkpoint")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop if reward >= threshold for this many consecutive steps (0=disabled)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.8,
                        help="Reward threshold for early stopping")

    return parser.parse_args()


def main():
    args = parse_args()

    if not getattr(args, "verbose", False):
        import logging
        from loguru import logger
        logger.disable("AGentCL")
        logging.basicConfig(level=logging.WARNING)
        for logger_name in ["AGentCL", "tau2", "litellm"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    output_dir = Path(args.output_dir) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理多 API keys
    user_api_key = args.user_api_key
    user_api_keys = None
    if args.user_api_keys:
        user_api_keys = args.user_api_keys.split()
        print(f"Using {len(user_api_keys)} API keys for multi-GPU training")

    # 创建配置 - 只训练单个 domain，不需要泛化测试
    # task_order 只包含一个 domain
    config = GRPOConfig(
        # Model
        model_name_or_path=args.model_name_or_path,
        model_dtype=args.model_dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        user_model=args.user_model,
        use_local_user_model=args.use_local_user_model,
        user_model_temperature=args.user_model_temperature,
        user_api_base=args.user_api_base,
        user_api_key=user_api_key,
        user_api_keys=user_api_keys,
        # Random seed
        seed=args.seed,
        # GRPO
        num_samples_per_prompt=args.num_samples_per_prompt,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        gamma=args.gamma,
        # Training
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps_per_task=args.num_steps_per_task,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        # Continual learning - 使用 sequential，单任务不需要特殊算法
        cl_algorithm="sequential",
        # Tasks - 只训练单个 domain
        task_order=[args.domain],
        max_tasks_per_domain=args.max_tasks_per_domain,
        # Logging
        log_dir=str(output_dir / "logs"),
        eval_interval=args.eval_interval,
        verbose=args.verbose,
        save_trajectory_logs=True,
        trajectory_log_interval=1,
        checkpoint_interval=args.checkpoint_interval,
        skip_intermediate_eval=args.skip_intermediate_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        # Evaluation
        pass_at_k=args.pass_at_k,
        num_eval_samples=args.num_eval_samples,
        num_eval_tasks=args.num_eval_tasks,
        # Optimization
        use_flash_attention=args.use_flash_attention,
        gradient_checkpointing=args.gradient_checkpointing,
        # W&B
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or f"single_{args.domain}",
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
    )

    print(f"\n{'='*60}")
    print(f"Single Domain Training for FWT Baseline")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")
    print(f"Steps: {args.num_steps_per_task}")
    print(f"{'='*60}\n")

    trainer = GRPOTrainer(config)

    try:
        # 使用简化的训练流程 - 只训练单个任务
        trainer.train_single_domain_only()
    finally:
        trainer.cleanup()

    print(f"\n{'='*60}")
    print(f"Single domain training completed!")
    print(f"Model saved to: {output_dir}/model")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
