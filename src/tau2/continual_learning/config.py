"""Configuration for GRPO-based continual learning training."""

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class GRPOConfig:
    # Model configuration
    model_name_or_path: str = "/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct"
    model_dtype: str = "bfloat16"

    # User model configuration
    user_model: str = "gpt_oss_120b"  # 使用紫东太初 API
    use_local_user_model: bool = False  # 默认使用 API
    user_model_temperature: float = 0.0

    # 紫东太初 API 配置
    user_api_base: str = "https://cloud.zidongtaichu.com/maas/v1"
    user_api_key: str = "vp8ggmuy102xmtpcyf9enr3g"
    # 多卡训练时，每个卡使用不同的 API key（可选）
    user_api_keys: Optional[list[str]] = None  # 如果提供，则按 rank 分配 key

    # Random seed for reproducibility
    seed: int = 42

    # GRPO hyperparameters — aligned with ToolRL run_grpo.sh
    num_samples_per_prompt: int = 4      # rollout.n=4
    kl_coef: float = 0.001               # algorithm.kl_ctrl.kl_coef=0.001
    clip_range: float = 0.2              # actor.clip_ratio=0.2 (ppo_trainer.yaml)
    entropy_coeff: float = 0.001         # actor.entropy_coeff=0.001
    gamma: float = 1.0                   # algorithm.gamma=1.0

    # ToolRL-style PPO mini-batch update
    ppo_mini_batch_size: int = 128       # actor.ppo_mini_batch_size=128
    ppo_micro_batch_size: int = 32       # actor.ppo_micro_batch_size (128/4 ranks)
    ppo_epochs: int = 1                  # actor.ppo_epochs=1

    # Training configuration — aligned with ToolRL run_grpo.sh
    batch_size_per_gpu: int = 64         # data.train_batch_size=512 / 8 GPUs = 64
    gradient_accumulation_steps: int = 1 # handled inside ppo mini-batch loop
    num_steps_per_task: int = 100
    learning_rate: float = 1e-6          # actor.optim.lr=1e-6
    warmup_steps: int = 10
    max_grad_norm: float = 1.0

    # Generation
    temperature: float = 1.0             # rollout.temperature=1.0
    max_new_tokens: int = 1024           # data.max_response_length=1024
    max_prompt_length: int = 2048        # data.max_prompt_length=2048

    # vLLM rollout configuration (H20 96GB: vLLM 40% + DeepSpeed 60%)
    # tp=1 per rank (persistent engine with weight offload/sync)
    vllm_tensor_parallel_size: int = 1     # ignored — always tp=1 per rank
    vllm_gpu_memory_utilization: float = 0.5   # KV-cache budget per GPU after DS offload
    vllm_enforce_eager: bool = True        # must be True with weight offload (no CUDA graphs)

    # Continual Learning configuration
    cl_algorithm: str = "sequential"
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.2

    # Task configuration
    task_order: list[str] = field(default_factory=lambda: ["airline", "retail", "telecom"])
    max_tasks_per_domain: Optional[int] = None
    train_split: float = 0.8

    # Distributed (env-first)
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", "1")))
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", "0")))

    # Logging/checkpointing
    log_dir: str = "logs/grpo_cl"
    save_interval: int = 10
    eval_interval: int = 5
    skip_intermediate_eval: bool = False  # 跳过训练中的中间评估，只在任务结束时评估
    checkpoint_interval: int = 5  # 每隔多少 step 保存一次 checkpoint，用于断点续训
    resume_from_checkpoint: Optional[str] = None  # 从指定 checkpoint 恢复训练
    resume_from_task: int = 0  # 从第几个 task 开始训练（用于持续学习断点续训）
    resume_from_eval_domain: Optional[str] = None  # 从指定 domain 的评估开始恢复（用于评估阶段中断恢复）
    skip_training_for_resume_task: bool = False  # 跳过 resume_from_task 的训练，直接从评估开始

    # 早停配置（基于训练 reward 饱和）
    early_stopping_patience: int = 10  # 连续多少步 reward >= threshold 后停止，0 表示不启用
    early_stopping_threshold: float = 0.8  # reward 达到此阈值视为饱和

    # API configuration
    user_api_timeout: int = 60  # User API 调用超时时间（秒）
    user_api_max_retries: int = 3  # User API 调用最大重试次数
    verbose: bool = False
    log_interval: int = 10
    use_progress_bar: bool = True
    save_trajectory_logs: bool = True
    trajectory_log_interval: int = 1

    # Evaluation configuration
    pass_at_k: int = 1  # pass@k 中的 k 值
    num_eval_samples: int = 5  # 每个任务评估时生成的样本数
    num_eval_tasks: int = 20  # 评估的任务数量

    # Single-domain baseline checkpoints (for FWT calculation)
    # FWT_j = 顺序训练后性能 - 单任务训练后性能
    single_domain_checkpoint_dir: Optional[str] = None  # 单任务训练的 checkpoint 目录

    # Optimization flags
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # ✅ Weights & Biases
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    wandb_mode: Optional[str] = None  # online/offline/disabled

    def __post_init__(self):
        valid_dtypes = ["bfloat16", "float16", "float32"]
        if self.model_dtype not in valid_dtypes:
            raise ValueError(f"model_dtype must be one of {valid_dtypes}, got {self.model_dtype}")

        valid_algorithms = [
            "sequential", "replay", "adaptive_replay",
            "ewc", "online_ewc", "ewc_pp",
            "progressive", "dynamic_expansion",
            "fusion", "adaptive_fusion",
        ]
        if self.cl_algorithm not in valid_algorithms:
            raise ValueError(f"cl_algorithm must be one of {valid_algorithms}, got {self.cl_algorithm}")

        valid_domains = ["airline", "retail", "telecom", "delivery", "instore", "ota",
                         "api_bank", "api_bank_level1", "api_bank_level2", "api_bank_level3",
                         "bamboogle"]
        for domain in self.task_order:
            if domain not in valid_domains:
                # rlla_by_domain may use arbitrary domain names — just warn, don't crash
                import warnings
                warnings.warn(f"Unknown domain in task_order: {domain}. "
                              f"Known domains: {valid_domains}")

        if self.batch_size_per_gpu < 1:
            raise ValueError(f"batch_size_per_gpu must be >= 1, got {self.batch_size_per_gpu}")

        if self.num_samples_per_prompt < 2:
            raise ValueError(f"num_samples_per_prompt must be >= 2 for GRPO, got {self.num_samples_per_prompt}")

        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be between 0 and 1, got {self.train_split}")

    @property
    def global_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.world_size

    @property
    def effective_batch_size(self) -> int:
        return self.global_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
