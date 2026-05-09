"""Experience Replay continual learning algorithm.

Experience Replay is a simple but effective continual learning method that
stores trajectories from previous tasks and replays them during training on
new tasks. This helps prevent catastrophic forgetting by maintaining exposure
to old task distributions.

References:
    - Ratcliff, R. (1990). Connectionist models of recognition memory:
      Constraints imposed by learning and forgetting functions.
    - Rolnick, D., et al. (2019). Experience Replay for Continual Learning.
"""

from typing import TYPE_CHECKING

from AGentCL.data_model.tasks import Task

from .base import CLAlgorithm

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class ReplayCL(CLAlgorithm):
    """Experience Replay continual learning algorithm.

    This algorithm maintains a buffer of trajectories from previous tasks
    and mixes them with new task samples during training. This helps prevent
    catastrophic forgetting by ensuring the model continues to see examples
    from old tasks.

    The algorithm supports multiple replay strategies:
    - Random: Sample uniformly from buffer
    - High-reward: Prefer high-reward trajectories
    - Recent: Prefer recently added trajectories
    - Balanced: Balance across domains

    Args:
        replay_ratio: Ratio of replay samples to new samples (default: 0.2)
            - 0.0 = no replay (equivalent to sequential)
            - 0.5 = equal mix of new and replay samples
            - 1.0 = equal number of replay and new samples
        replay_strategy: Strategy for sampling from buffer
            - "random": Uniform random sampling
            - "high_reward": Prefer high-reward trajectories
            - "recent": Prefer recently added trajectories
            - "balanced": Balance across previous domains
        min_buffer_size: Minimum buffer size before starting replay (default: 10)
        replay_all_domains: Whether to replay from all previous domains or just the most recent (default: True)

    Example:
        >>> config = GRPOConfig(cl_algorithm="replay")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = ReplayCL(replay_ratio=0.3, replay_strategy="high_reward")
        >>> trainer.train()
    """

    def __init__(
        self,
        replay_ratio: float = 0.2,
        replay_strategy: str = "random",
        min_buffer_size: int = 10,
        replay_all_domains: bool = True,
    ):
        """Initialize Experience Replay algorithm.

        Args:
            replay_ratio: Ratio of replay samples to new samples
            replay_strategy: Strategy for sampling ("random", "high_reward", "recent", "balanced")
            min_buffer_size: Minimum buffer size before starting replay
            replay_all_domains: Whether to replay from all previous domains
        """
        self.replay_ratio = replay_ratio
        self.replay_strategy = replay_strategy
        self.min_buffer_size = min_buffer_size
        self.replay_all_domains = replay_all_domains

        # Validate parameters
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError(f"replay_ratio must be between 0 and 1, got {replay_ratio}")

        valid_strategies = ["random", "high_reward", "recent", "balanced"]
        if replay_strategy not in valid_strategies:
            raise ValueError(
                f"replay_strategy must be one of {valid_strategies}, got {replay_strategy}"
            )

        # Track which domains we've seen
        self.seen_domains = []

        # Statistics
        self.total_replay_samples = 0
        self.replay_samples_per_domain = {}

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """Augment batch with replay samples from previous tasks.

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Augmented list of tasks (new tasks + replay samples)
        """
        # Track seen domains
        if current_domain not in self.seen_domains:
            self.seen_domains.append(current_domain)

        # If no previous domains or replay_ratio is 0, return original batch
        if len(self.seen_domains) <= 1 or self.replay_ratio == 0.0:
            return new_tasks

        # Get previous domains
        previous_domains = self._get_previous_domains(current_domain)

        if not previous_domains:
            return new_tasks

        # Calculate number of replay samples
        num_new = len(new_tasks)
        num_replay = int(num_new * self.replay_ratio)

        if num_replay == 0:
            return new_tasks

        # Sample replay tasks from buffer
        replay_tasks = self._sample_replay_tasks(
            previous_domains=previous_domains,
            num_samples=num_replay,
        )

        # Update statistics
        self.total_replay_samples += len(replay_tasks)

        # Combine new and replay tasks
        augmented_batch = new_tasks + replay_tasks

        return augmented_batch

    def _get_previous_domains(self, current_domain: str) -> list[str]:
        """Get list of previous domains to replay from.

        Args:
            current_domain: Current domain

        Returns:
            List of previous domain names
        """
        if self.replay_all_domains:
            # Replay from all previous domains
            previous = [d for d in self.seen_domains if d != current_domain]
        else:
            # Replay only from most recent domain
            if len(self.seen_domains) >= 2:
                # Get the domain before current
                current_idx = self.seen_domains.index(current_domain)
                if current_idx > 0:
                    previous = [self.seen_domains[current_idx - 1]]
                else:
                    previous = []
            else:
                previous = []

        return previous

    def _sample_replay_tasks(
        self, previous_domains: list[str], num_samples: int
    ) -> list[Task]:
        """Sample replay tasks from trajectory buffer.

        This method needs access to the trainer's trajectory buffer,
        which is passed via the trainer reference.

        Args:
            previous_domains: List of domains to sample from
            num_samples: Number of samples to draw

        Returns:
            List of replay tasks
        """
        # Note: This method will be called with trainer context
        # We'll access the buffer through the trainer instance
        # This is set up in the augment_batch call from the trainer

        if not hasattr(self, "_trainer"):
            # No trainer reference, return empty
            return []

        trainer = self._trainer
        buffer = trainer.trajectory_buffer

        # Check if buffer has enough samples
        total_buffer_size = sum(
            buffer.get_size(domain) for domain in previous_domains
        )

        if total_buffer_size < self.min_buffer_size:
            # Not enough samples yet
            return []

        replay_tasks = []

        if self.replay_strategy == "balanced":
            # Balance samples across domains
            samples_per_domain = max(1, num_samples // len(previous_domains))

            for domain in previous_domains:
                domain_samples = buffer.sample(
                    domain=domain,
                    num_samples=samples_per_domain,
                    strategy="random",
                )
                replay_tasks.extend([record.task for record in domain_samples])

                # Track statistics
                if domain not in self.replay_samples_per_domain:
                    self.replay_samples_per_domain[domain] = 0
                self.replay_samples_per_domain[domain] += len(domain_samples)

            # If we need more samples, sample randomly from all domains
            if len(replay_tasks) < num_samples:
                remaining = num_samples - len(replay_tasks)
                additional_samples = buffer.sample_multi_domain(
                    domains=previous_domains,
                    num_samples_per_domain=remaining // len(previous_domains) + 1,
                    strategy="random",
                )
                replay_tasks.extend([record.task for record in additional_samples])

        else:
            # Sample from all previous domains using specified strategy
            samples = buffer.sample_multi_domain(
                domains=previous_domains,
                num_samples_per_domain=num_samples // len(previous_domains) + 1,
                strategy=self.replay_strategy,
            )
            replay_tasks = [record.task for record in samples]

            # Track statistics
            for record in samples:
                if record.domain not in self.replay_samples_per_domain:
                    self.replay_samples_per_domain[record.domain] = 0
                self.replay_samples_per_domain[record.domain] += 1

        # Trim to exact number needed
        replay_tasks = replay_tasks[:num_samples]

        return replay_tasks

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after each training step.

        Store trainer reference for buffer access.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        # Store trainer reference for buffer access
        self._trainer = trainer

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Hook called after finishing a task.

        Log replay statistics.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score (unused in replay)
        """
        if trainer.is_main_process():
            print(f"\n{'='*80}")
            print(f"Experience Replay Statistics for {domain}")
            print(f"{'='*80}")
            print(f"Total replay samples used: {self.total_replay_samples}")
            print(f"Replay samples per domain:")
            for d, count in self.replay_samples_per_domain.items():
                print(f"  {d}: {count}")
            print(f"Buffer sizes:")
            for d in self.seen_domains:
                size = trainer.trajectory_buffer.get_size(d)
                print(f"  {d}: {size}")
            print(f"{'='*80}\n")

    def get_statistics(self) -> dict:
        """Get replay statistics.

        Returns:
            Dictionary with replay statistics
        """
        return {
            "replay_ratio": self.replay_ratio,
            "replay_strategy": self.replay_strategy,
            "total_replay_samples": self.total_replay_samples,
            "replay_samples_per_domain": self.replay_samples_per_domain.copy(),
            "seen_domains": self.seen_domains.copy(),
        }

    def reset_statistics(self):
        """Reset replay statistics."""
        self.total_replay_samples = 0
        self.replay_samples_per_domain = {}


class AdaptiveReplayCL(ReplayCL):
    """Adaptive Experience Replay with dynamic replay ratio.

    This variant adjusts the replay ratio based on forgetting signals.
    If performance on old tasks drops significantly, it increases replay.

    Args:
        initial_replay_ratio: Initial replay ratio (default: 0.2)
        max_replay_ratio: Maximum replay ratio (default: 0.5)
        min_replay_ratio: Minimum replay ratio (default: 0.1)
        adaptation_rate: How quickly to adapt (default: 0.1)
        forgetting_threshold: Performance drop threshold to trigger increase (default: 0.1)
        replay_strategy: Strategy for sampling from buffer
        min_buffer_size: Minimum buffer size before starting replay
        replay_all_domains: Whether to replay from all previous domains

    Example:
        >>> config = GRPOConfig(cl_algorithm="adaptive_replay")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = AdaptiveReplayCL(
        ...     initial_replay_ratio=0.2,
        ...     max_replay_ratio=0.5,
        ...     forgetting_threshold=0.1
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        initial_replay_ratio: float = 0.2,
        max_replay_ratio: float = 0.5,
        min_replay_ratio: float = 0.1,
        adaptation_rate: float = 0.1,
        forgetting_threshold: float = 0.1,
        replay_strategy: str = "random",
        min_buffer_size: int = 10,
        replay_all_domains: bool = True,
    ):
        """Initialize Adaptive Experience Replay algorithm."""
        super().__init__(
            replay_ratio=initial_replay_ratio,
            replay_strategy=replay_strategy,
            min_buffer_size=min_buffer_size,
            replay_all_domains=replay_all_domains,
        )

        self.initial_replay_ratio = initial_replay_ratio
        self.max_replay_ratio = max_replay_ratio
        self.min_replay_ratio = min_replay_ratio
        self.adaptation_rate = adaptation_rate
        self.forgetting_threshold = forgetting_threshold

        # Track performance on previous tasks
        self.previous_performance = {}

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Adapt replay ratio based on forgetting.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score (unused)
        """
        # Call parent hook for statistics
        super().post_task_hook(trainer, domain, performance=performance)

        if not trainer.is_main_process():
            return

        # Evaluate on all previous domains
        for prev_domain in self.seen_domains[:-1]:  # Exclude current domain
            # Evaluate
            metrics = trainer.evaluate_task(prev_domain, num_eval_tasks=10)
            current_perf = metrics["reward_mean"]

            # Check if we have previous performance
            if prev_domain in self.previous_performance:
                prev_perf = self.previous_performance[prev_domain]
                performance_drop = prev_perf - current_perf

                # If significant forgetting, increase replay ratio
                if performance_drop > self.forgetting_threshold:
                    old_ratio = self.replay_ratio
                    self.replay_ratio = min(
                        self.max_replay_ratio,
                        self.replay_ratio + self.adaptation_rate,
                    )
                    print(
                        f"Detected forgetting on {prev_domain} "
                        f"(drop: {performance_drop:.3f}). "
                        f"Increasing replay ratio: {old_ratio:.3f} -> {self.replay_ratio:.3f}"
                    )

                # If performance is stable or improving, can decrease replay
                elif performance_drop < -0.05:  # Performance improved
                    old_ratio = self.replay_ratio
                    self.replay_ratio = max(
                        self.min_replay_ratio,
                        self.replay_ratio - self.adaptation_rate / 2,
                    )
                    print(
                        f"Performance stable on {prev_domain}. "
                        f"Decreasing replay ratio: {old_ratio:.3f} -> {self.replay_ratio:.3f}"
                    )

            # Update previous performance
            self.previous_performance[prev_domain] = current_perf

    def get_statistics(self) -> dict:
        """Get replay statistics including adaptation info.

        Returns:
            Dictionary with replay statistics
        """
        stats = super().get_statistics()
        stats.update(
            {
                "current_replay_ratio": self.replay_ratio,
                "initial_replay_ratio": self.initial_replay_ratio,
                "max_replay_ratio": self.max_replay_ratio,
                "min_replay_ratio": self.min_replay_ratio,
                "previous_performance": self.previous_performance.copy(),
            }
        )
        return stats
