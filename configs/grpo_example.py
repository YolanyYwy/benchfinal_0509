# Example Configuration for GRPO Continual Learning

# This is an example Python script showing how to configure and run GRPO training
# You can modify these parameters and run this script directly

from tau2.continual_learning import GRPOConfig, GRPOTrainer

# Create configuration
config = GRPOConfig(
    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",  # or "meta-llama/Llama-3-8B-Instruct"
    model_dtype="bfloat16",  # bfloat16, float16, or float32
    temperature=0.7,  # Sampling temperature for generation
    max_new_tokens=2048,  # Maximum tokens to generate per response

    # ============================================================================
    # GRPO Hyperparameters
    # ============================================================================
    num_samples_per_prompt=4,  # Number of responses to generate per task
    kl_coef=0.1,  # KL divergence penalty coefficient
    gamma=1.0,  # Discount factor for rewards

    # ============================================================================
    # Training Configuration
    # ============================================================================
    batch_size_per_gpu=4,  # Tasks per GPU per step (reduce if OOM)
    gradient_accumulation_steps=2,  # Accumulate gradients over N steps
    num_steps_per_task=100,  # Training steps per domain
    learning_rate=1e-6,  # Learning rate (start small for stability)
    warmup_steps=10,  # Warmup steps for learning rate
    max_grad_norm=1.0,  # Gradient clipping threshold

    # ============================================================================
    # Continual Learning Configuration
    # ============================================================================
    cl_algorithm="sequential",  # Only "sequential" implemented currently
    replay_buffer_size=1000,  # Max trajectories to store per domain
    replay_ratio=0.2,  # Ratio of replay samples (for future algorithms)

    # ============================================================================
    # Task Configuration
    # ============================================================================
    task_order=["airline", "retail", "telecom"],  # Order of domains
    max_tasks_per_domain=None,  # Limit tasks per domain (None = use all)
    train_split=0.8,  # Fraction for training (rest for eval)

    # ============================================================================
    # Logging and Checkpointing
    # ============================================================================
    log_dir="logs/grpo_cl_example",  # Directory for logs and checkpoints
    save_interval=10,  # Save checkpoint every N steps
    eval_interval=5,  # Evaluate every N steps
    wandb_project=None,  # Set to project name to enable wandb logging

    # ============================================================================
    # Optimization Flags
    # ============================================================================
    use_flash_attention=True,  # Use Flash Attention 2 (requires installation)
    gradient_checkpointing=True,  # Save memory at cost of speed
)

# Create trainer
trainer = GRPOTrainer(config)

# Train
trainer.train()
