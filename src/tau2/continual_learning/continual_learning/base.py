"""Base classes for continual learning algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from AGentCL.data_model.tasks import Task

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class CLAlgorithm(ABC):
    """Base class for continual learning algorithms.

    This abstract class defines the interface for continual learning algorithms
    that can be plugged into the GRPO trainer. Algorithms can modify the training
    process through three hooks:

    1. augment_batch: Modify the batch before training (e.g., add replay samples)
    2. post_step_hook: Called after each training step (e.g., update regularization)
    3. post_task_hook: Called after finishing a task (e.g., consolidate memory)

    Example algorithms that can be implemented:
    - Sequential: Vanilla sequential training (no forgetting prevention)
    - Replay: Experience replay with stored trajectories
    - EWC: Elastic Weight Consolidation with Fisher information
    - PackNet: Iterative pruning and packing
    - Progressive Networks: Add new capacity for each task
    """

    @abstractmethod
    def augment_batch(self, new_tasks: list[Task], current_domain: str) -> list[Task]:
        """Augment batch with replay/regularization samples.

        This method is called before each training step to potentially modify
        the batch of tasks. For example, experience replay would add samples
        from previous tasks.

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Augmented list of tasks (may include replay samples)
        """
        pass

    @abstractmethod
    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after each training step.

        This method is called after the policy has been updated. It can be used
        to update regularization parameters, compute Fisher information, etc.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        pass

    @abstractmethod
    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Hook called after finishing a task.

        This method is called after all training steps for a task are complete.
        It can be used to consolidate memory, prune networks, etc.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score on this task (optional, to avoid re-evaluation)
        """
        pass


class SequentialCL(CLAlgorithm):
    """Vanilla sequential training (no forgetting prevention).

    This is the baseline continual learning approach where tasks are
    trained sequentially without any mechanism to prevent catastrophic
    forgetting. It serves as the lower bound for CL performance and
    demonstrates the forgetting problem.

    This algorithm:
    - Does not modify batches (no replay)
    - Does not add regularization
    - Does not consolidate memory

    It simply trains on each task in sequence, allowing the model to
    freely overwrite previous knowledge.
    """

    def augment_batch(self, new_tasks: list[Task], current_domain: str) -> list[Task]:
        """No augmentation - just return new tasks.

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Unmodified list of new tasks
        """
        return new_tasks

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """No action needed after training step.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """No action needed after task completion.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score (unused in sequential)
        """
        pass


# Future CL algorithms can be added here by extending CLAlgorithm:
#
# class ReplayCL(CLAlgorithm):
#     """Experience replay continual learning."""
#
#     def __init__(self, replay_ratio: float = 0.2):
#         self.replay_ratio = replay_ratio
#
#     def augment_batch(self, new_tasks, current_domain):
#         # Sample from trajectory buffer
#         # Mix with new tasks
#         pass
#
#
# class EWCCL(CLAlgorithm):
#     """Elastic Weight Consolidation."""
#
#     def __init__(self, ewc_lambda: float = 0.4):
#         self.ewc_lambda = ewc_lambda
#         self.fisher_information = {}
#         self.optimal_params = {}
#
#     def post_step_hook(self, trainer, domain):
#         # Add EWC regularization to loss
#         pass
#
#     def post_task_hook(self, trainer, domain):
#         # Compute Fisher information matrix
#         # Store optimal parameters
#         pass
#
#
# class PackNetCL(CLAlgorithm):
#     """PackNet: Iterative pruning and packing."""
#
#     def __init__(self, prune_ratio: float = 0.5):
#         self.prune_ratio = prune_ratio
#         self.masks = {}
#
#     def post_task_hook(self, trainer, domain):
#         # Prune least important weights
#         # Create mask for this task
#         # Pack weights for next task
#         pass
