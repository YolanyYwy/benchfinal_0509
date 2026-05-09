"""
Training script for GRPO-based continual learning on agent tool-use tasks.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tau2.continual_learning import GRPOConfig
from tau2.continual_learning.grpo_trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train agents with GRPO for continual learning on tool-use tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, default="/home/houzhiyan/Qwen3-4B")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--user_model", type=str, default="gpt_oss_120b")
    parser.add_argument("--use_local_user_model", action="store_true", default=False)
    parser.add_argument("--no_local_user_model", action="store_false", dest="use_local_user_model")
    parser.add_argument("--user_model_temperature", type=float, default=0.0)

    # User API configuration (中转 API)
    parser.add_argument("--user_api_base", type=str, default="https://cloud.zidongtaichu.com/maas/v1",
                        help="Base URL for user model API")
    parser.add_argument("--user_api_key", type=str, default="vp8ggmuy102xmtpcyf9enr3g",
                        help="API key for user model")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # GRPO hyperparameters
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)

    # Training configuration
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_steps_per_task", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Continual learning configuration
    parser.add_argument(
        "--cl_algorithm",
        type=str,
        default="sequential",
        choices=[
            "sequential", "replay", "adaptive_replay",
            "ewc", "online_ewc", "ewc_pp",
            "progressive", "dynamic_expansion",
            "fusion", "adaptive_fusion",
        ],
    )
    parser.add_argument("--replay_buffer_size", type=int, default=1000)
    parser.add_argument("--replay_ratio", type=float, default=0.2)

    # Task configuration
    parser.add_argument(
        "--task_order",
        type=str,
        nargs="+",
        default=["airline", "retail", "telecom"],
        choices=["airline", "retail", "telecom", "delivery", "instore", "ota"],
    )
    parser.add_argument("--max_tasks_per_domain", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=0.8)

    # Logging and checkpointing
    parser.add_argument("--log_dir", type=str, default="logs/grpo_cl")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_progress_bar", action="store_true", default=True)
    parser.add_argument("--no_progress_bar", action="store_false", dest="use_progress_bar")
    parser.add_argument("--save_trajectory_logs", action="store_true", default=True)
    parser.add_argument("--no_trajectory_logs", action="store_false", dest="save_trajectory_logs")
    parser.add_argument("--trajectory_log_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Save checkpoint every N steps")
    parser.add_argument("--skip_intermediate_eval", action="store_true", default=False,
                        help="Skip intermediate evaluation during training")

    # Evaluation configuration
    parser.add_argument("--pass_at_k", type=int, default=1, help="k value for pass@k metric")
    parser.add_argument("--num_eval_samples", type=int, default=5, help="Number of samples per task for evaluation")
    parser.add_argument("--num_eval_tasks", type=int, default=20, help="Number of tasks to evaluate")

    # Single-domain baseline for FWT calculation
    parser.add_argument("--single_domain_checkpoint_dir", type=str, default=None,
                        help="Directory containing single-domain trained checkpoints for FWT baseline")

    # Optimization flags
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--no_flash_attention", action="store_false", dest="use_flash_attention")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")

    # Resuming
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint directory")
    parser.add_argument("--resume_from_task", type=int, default=0, help="Resume from task index (for continual learning)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from step checkpoint")

    # ✅ W&B
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project (None disables wandb)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="W&B tags (optional)")
    parser.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"],
                        help="Override wandb mode (optional)")

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

    config = GRPOConfig(
        # Model
        model_name_or_path=args.model_name_or_path,
        model_dtype=args.model_dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        user_model=args.user_model,
        use_local_user_model=args.use_local_user_model,
        user_model_temperature=args.user_model_temperature,
        # User API configuration
        user_api_base=args.user_api_base,
        user_api_key=args.user_api_key,
        # Random seed
        seed=args.seed,
        # GRPO
        num_samples_per_prompt=args.num_samples_per_prompt,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
        # Training
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps_per_task=args.num_steps_per_task,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        # Continual learning
        cl_algorithm=args.cl_algorithm,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        # Tasks
        task_order=args.task_order,
        max_tasks_per_domain=args.max_tasks_per_domain,
        train_split=args.train_split,
        # Logging
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        verbose=args.verbose,
        log_interval=args.log_interval,
        use_progress_bar=args.use_progress_bar,
        save_trajectory_logs=args.save_trajectory_logs,
        trajectory_log_interval=args.trajectory_log_interval,
        checkpoint_interval=args.checkpoint_interval,
        skip_intermediate_eval=args.skip_intermediate_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_from_task=args.resume_from_task,
        # Evaluation
        pass_at_k=args.pass_at_k,
        num_eval_samples=args.num_eval_samples,
        num_eval_tasks=args.num_eval_tasks,
        # Single-domain baseline
        single_domain_checkpoint_dir=args.single_domain_checkpoint_dir,
        # Optimization
        use_flash_attention=args.use_flash_attention,
        gradient_checkpointing=args.gradient_checkpointing,
        # ✅ W&B
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
    )

    trainer = GRPOTrainer(config)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
