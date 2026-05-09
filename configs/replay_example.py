"""Example: Training with Experience Replay

This script demonstrates how to use Experience Replay for continual learning.
Experience Replay helps prevent catastrophic forgetting by mixing old task
samples with new task samples during training.
"""

from tau2.continual_learning import GRPOConfig, GRPOTrainer
from tau2.continual_learning.continual_learning import ReplayCL, AdaptiveReplayCL


def train_with_basic_replay():
    """Train with basic experience replay."""
    print("\n" + "="*80)
    print("Training with Basic Experience Replay")
    print("="*80 + "\n")

    # Create configuration
    config = GRPOConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        task_order=["airline", "retail", "telecom"],
        num_steps_per_task=100,
        batch_size_per_gpu=4,
        num_samples_per_prompt=4,
        learning_rate=1e-6,

        # Replay configuration
        cl_algorithm="replay",
        replay_ratio=0.3,  # 30% of batch will be replay samples
        replay_buffer_size=1000,

        log_dir="logs/replay_example",
    )

    # Create trainer
    trainer = GRPOTrainer(config)

    # Optionally customize replay algorithm
    trainer.cl_algorithm = ReplayCL(
        replay_ratio=0.3,
        replay_strategy="high_reward",  # Prefer high-reward trajectories
        min_buffer_size=20,  # Start replay after 20 samples
        replay_all_domains=True,  # Replay from all previous domains
    )

    # Train
    trainer.train()

    print("\nTraining complete!")
    print(f"Replay statistics: {trainer.cl_algorithm.get_statistics()}")


def train_with_adaptive_replay():
    """Train with adaptive experience replay."""
    print("\n" + "="*80)
    print("Training with Adaptive Experience Replay")
    print("="*80 + "\n")

    # Create configuration
    config = GRPOConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        task_order=["airline", "retail", "telecom"],
        num_steps_per_task=100,
        batch_size_per_gpu=4,
        num_samples_per_prompt=4,
        learning_rate=1e-6,

        # Adaptive replay configuration
        cl_algorithm="adaptive_replay",
        replay_ratio=0.2,  # Initial replay ratio
        replay_buffer_size=1000,

        log_dir="logs/adaptive_replay_example",
    )

    # Create trainer
    trainer = GRPOTrainer(config)

    # Optionally customize adaptive replay
    trainer.cl_algorithm = AdaptiveReplayCL(
        initial_replay_ratio=0.2,
        max_replay_ratio=0.5,  # Can increase up to 50%
        min_replay_ratio=0.1,  # Can decrease down to 10%
        adaptation_rate=0.1,  # How quickly to adapt
        forgetting_threshold=0.1,  # Performance drop threshold
        replay_strategy="high_reward",
    )

    # Train
    trainer.train()

    print("\nTraining complete!")
    print(f"Final replay ratio: {trainer.cl_algorithm.replay_ratio:.3f}")
    print(f"Replay statistics: {trainer.cl_algorithm.get_statistics()}")


def compare_strategies():
    """Compare different replay strategies."""
    print("\n" + "="*80)
    print("Comparing Replay Strategies")
    print("="*80 + "\n")

    strategies = ["random", "high_reward", "recent", "balanced"]

    for strategy in strategies:
        print(f"\nTraining with {strategy} strategy...")

        config = GRPOConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            task_order=["airline", "retail"],
            num_steps_per_task=50,
            batch_size_per_gpu=4,
            max_tasks_per_domain=10,  # Limit for faster comparison

            cl_algorithm="replay",
            replay_ratio=0.3,

            log_dir=f"logs/replay_{strategy}",
        )

        trainer = GRPOTrainer(config)
        trainer.cl_algorithm = ReplayCL(
            replay_ratio=0.3,
            replay_strategy=strategy,
        )

        trainer.train()

        # Get final metrics
        stats = trainer.cl_algorithm.get_statistics()
        print(f"  Total replay samples: {stats['total_replay_samples']}")


def custom_replay_configuration():
    """Example of custom replay configuration."""
    print("\n" + "="*80)
    print("Custom Replay Configuration")
    print("="*80 + "\n")

    config = GRPOConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        task_order=["airline", "retail", "telecom"],
        num_steps_per_task=100,

        cl_algorithm="replay",
        log_dir="logs/custom_replay",
    )

    trainer = GRPOTrainer(config)

    # Custom replay: only replay from most recent domain
    trainer.cl_algorithm = ReplayCL(
        replay_ratio=0.4,  # Higher replay ratio
        replay_strategy="high_reward",  # Focus on successful examples
        min_buffer_size=50,  # Wait for more samples before replaying
        replay_all_domains=False,  # Only replay from most recent domain
    )

    print("Configuration:")
    print(f"  Replay ratio: {trainer.cl_algorithm.replay_ratio}")
    print(f"  Strategy: {trainer.cl_algorithm.replay_strategy}")
    print(f"  Min buffer size: {trainer.cl_algorithm.min_buffer_size}")
    print(f"  Replay all domains: {trainer.cl_algorithm.replay_all_domains}")

    trainer.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "basic"

    if mode == "basic":
        train_with_basic_replay()
    elif mode == "adaptive":
        train_with_adaptive_replay()
    elif mode == "compare":
        compare_strategies()
    elif mode == "custom":
        custom_replay_configuration()
    else:
        print("Usage: python replay_example.py [basic|adaptive|compare|custom]")
        print("\nModes:")
        print("  basic    - Train with basic experience replay")
        print("  adaptive - Train with adaptive experience replay")
        print("  compare  - Compare different replay strategies")
        print("  custom   - Custom replay configuration")
