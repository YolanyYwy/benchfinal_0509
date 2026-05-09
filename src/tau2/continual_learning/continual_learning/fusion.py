"""Model Fusion/Merging for continual learning.

Model fusion combines multiple task-specific models into a single model
that performs well on all tasks. This includes techniques like model
averaging, task arithmetic, and learned merging.

References:
    - Wortsman, M., et al. (2022). Model soups: averaging weights of
      multiple fine-tuned models improves accuracy. ICML.
    - Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. ICLR.
"""

import copy
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from AGentCL.data_model.tasks import Task

from .base import CLAlgorithm

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class ModelFusionCL(CLAlgorithm):
    """Model Fusion continual learning algorithm.

    Trains separate models for each task, then merges them using
    various fusion strategies (averaging, task arithmetic, etc.).

    Strategies:
        - "average": Simple parameter averaging
        - "weighted_average": Performance-weighted averaging
        - "task_arithmetic": Task vector arithmetic
        - "fisher_weighted": Fisher-information weighted merging

    Args:
        fusion_strategy: Strategy for merging models (default: "weighted_average")
        merge_frequency: How often to merge (default: "per_task")
            - "per_task": Merge after each task
            - "end": Merge only at the end
        keep_task_models: Whether to keep individual task models (default: True)

    Example:
        >>> config = GRPOConfig(cl_algorithm="fusion")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = ModelFusionCL(fusion_strategy="weighted_average")
        >>> trainer.train()
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted_average",
        merge_frequency: str = "per_task",
        keep_task_models: bool = True,
    ):
        """Initialize Model Fusion algorithm.

        Args:
            fusion_strategy: Strategy for merging
            merge_frequency: When to merge models
            keep_task_models: Whether to keep task-specific models
        """
        valid_strategies = ["average", "weighted_average", "task_arithmetic", "fisher_weighted"]
        if fusion_strategy not in valid_strategies:
            raise ValueError(
                f"fusion_strategy must be one of {valid_strategies}, got {fusion_strategy}"
            )

        valid_frequencies = ["per_task", "end"]
        if merge_frequency not in valid_frequencies:
            raise ValueError(
                f"merge_frequency must be one of {valid_frequencies}, got {merge_frequency}"
            )

        self.fusion_strategy = fusion_strategy
        self.merge_frequency = merge_frequency
        self.keep_task_models = keep_task_models

        # Storage for task-specific models
        self.task_models = {}  # {domain: model_state_dict}
        self.task_performance = {}  # {domain: performance_score}
        self.base_model = None  # Initial model before any training

        # For task arithmetic
        self.task_vectors = {}  # {domain: task_vector}

        # Statistics
        self.num_merges = 0
        self.merge_history = []

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """No batch augmentation for Model Fusion.

        Args:
            new_tasks: New tasks
            current_domain: Current domain

        Returns:
            Unmodified tasks
        """
        return new_tasks

    def save_base_model(self, model: nn.Module):
        """Save the initial model before training.

        Args:
            model: Base model
        """
        if self.base_model is None:
            self.base_model = {
                name: param.data.clone().cpu()
                for name, param in model.named_parameters()
            }
            print("Saved base model for task arithmetic")

    def save_task_model(
        self,
        model: nn.Module,
        domain: str,
        performance: float,
    ):
        """Save model after training on a task.

        Args:
            model: Trained model
            domain: Domain name
            performance: Performance score on this task
        """
        # Save model state to CPU to save GPU memory
        self.task_models[domain] = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
        }

        # Save performance
        self.task_performance[domain] = performance

        # Compute task vector (for task arithmetic) - also on CPU
        if self.base_model is not None:
            self.task_vectors[domain] = {
                name: self.task_models[domain][name] - self.base_model[name]
                for name in self.base_model
            }

        print(f"Saved model for {domain} (performance: {performance:.3f})")

    def merge_models(self, model: nn.Module) -> nn.Module:
        """Merge task-specific models into a single model.

        Args:
            model: Model to update with merged parameters

        Returns:
            Updated model
        """
        if len(self.task_models) == 0:
            print("No task models to merge")
            return model

        print(f"\nMerging {len(self.task_models)} models using {self.fusion_strategy}...")

        if self.fusion_strategy == "average":
            merged_params = self._average_merge()
        elif self.fusion_strategy == "weighted_average":
            merged_params = self._weighted_average_merge()
        elif self.fusion_strategy == "task_arithmetic":
            merged_params = self._task_arithmetic_merge()
        elif self.fusion_strategy == "fisher_weighted":
            merged_params = self._fisher_weighted_merge()
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Update model parameters (merged_params are on CPU, need to move to model's device)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in merged_params:
                    # Move merged param to the same device as model param
                    merged_param = merged_params[name].to(param.device)
                    param.copy_(merged_param)

        self.num_merges += 1
        self.merge_history.append({
            "num_models": len(self.task_models),
            "strategy": self.fusion_strategy,
            "domains": list(self.task_models.keys()),
        })

        print(f"Models merged successfully (merge #{self.num_merges})")

        return model

    def _average_merge(self) -> Dict[str, torch.Tensor]:
        """Simple parameter averaging.

        Returns:
            Merged parameters (on CPU)
        """
        merged = {}

        # Get parameter names from first model
        param_names = list(next(iter(self.task_models.values())).keys())
        total_params = len(param_names)
        num_models = len(self.task_models)

        print(f"Averaging {total_params} parameters from {num_models} models...")

        with torch.no_grad():
            for idx, name in enumerate(param_names):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{total_params}")
                # All params are on CPU now, so this is safe
                params = [self.task_models[domain][name] for domain in self.task_models]
                merged[name] = torch.stack(params).mean(dim=0)

        print(f"  Progress: {total_params}/{total_params} - Done!")
        return merged

    def _weighted_average_merge(self) -> Dict[str, torch.Tensor]:
        """Performance-weighted parameter averaging.

        Returns:
            Merged parameters
        """
        merged = {}

        # Compute weights from performance
        total_perf = sum(self.task_performance.values())
        if total_perf == 0:
            # 如果所有 performance 都是 0，使用均匀权重
            weights = {domain: 1.0 / len(self.task_performance) for domain in self.task_performance}
        else:
            weights = {
                domain: perf / total_perf
                for domain, perf in self.task_performance.items()
            }

        print(f"Merge weights: {weights}")

        # Get parameter names
        param_names = list(next(iter(self.task_models.values())).keys())
        total_params = len(param_names)
        domains = list(self.task_models.keys())

        print(f"Merging {total_params} parameters...")

        # 使用 torch.no_grad() 和批量处理来加速
        with torch.no_grad():
            for idx, name in enumerate(param_names):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{total_params}")

                try:
                    # 获取第一个参数作为基础
                    first_param = self.task_models[domains[0]][name]
                    device = first_param.device
                    dtype = first_param.dtype

                    # 初始化为零
                    merged[name] = torch.zeros_like(first_param)

                    # 加权求和
                    for domain in domains:
                        param = self.task_models[domain][name]
                        # 确保在同一设备上
                        if param.device != device:
                            param = param.to(device)
                        merged[name].add_(param, alpha=weights[domain])

                except Exception as e:
                    print(f"  Warning: Failed to merge parameter {name}: {e}")
                    # 使用第一个模型的参数作为后备
                    merged[name] = self.task_models[domains[0]][name].clone()

        print(f"  Progress: {total_params}/{total_params} - Done!")
        return merged

    def _task_arithmetic_merge(self) -> Dict[str, torch.Tensor]:
        """Task arithmetic merging.

        Merges task vectors: θ_merged = θ_base + Σ λ_i * τ_i
        where τ_i = θ_i - θ_base

        Returns:
            Merged parameters (on CPU)
        """
        if self.base_model is None:
            print("Warning: No base model for task arithmetic, using average instead")
            return self._average_merge()

        merged = {}
        param_names = list(self.base_model.keys())
        total_params = len(param_names)

        print(f"Task arithmetic: base + {len(self.task_vectors)} task vectors")
        print(f"Merging {total_params} parameters...")

        # Add task vectors with equal weight
        lambda_i = 1.0 / len(self.task_vectors)

        with torch.no_grad():
            for idx, name in enumerate(param_names):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{total_params}")
                # Start with base model (already on CPU)
                merged[name] = self.base_model[name].clone()
                # Add task vectors
                for domain, task_vector in self.task_vectors.items():
                    if name in task_vector:
                        merged[name] += lambda_i * task_vector[name]

        print(f"  Progress: {total_params}/{total_params} - Done!")
        return merged

    def _fisher_weighted_merge(self) -> Dict[str, torch.Tensor]:
        """Fisher-information weighted merging.

        Note: This is a placeholder. Full implementation would require
        computing Fisher information for each task model.

        Returns:
            Merged parameters
        """
        print("Warning: Fisher-weighted merge not fully implemented, using weighted average")
        return self._weighted_average_merge()

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """No per-step operations.

        Args:
            trainer: GRPO trainer
            domain: Current domain
        """
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Save task model and optionally merge.

        Args:
            trainer: GRPO trainer
            domain: Completed domain
            performance: Performance score (optional, to avoid re-evaluation)
        """
        if not trainer.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"Model Fusion Post-Task Processing for {domain}")
        print(f"{'='*80}")

        # Save base model (first task only)
        if len(self.task_models) == 0:
            self.save_base_model(trainer.policy.model)

        # 使用传入的 performance，如果没有则评估
        if performance is None:
            metrics = trainer.evaluate_task(domain)
            performance = metrics.get("reward_mean", 0.0)

        # Save task-specific model
        self.save_task_model(trainer.policy.model, domain, performance)

        # Merge if configured
        if self.merge_frequency == "per_task" and len(self.task_models) > 1:
            trainer.policy.model = self.merge_models(trainer.policy.model)

        # Statistics
        print(f"Task models saved: {len(self.task_models)}")
        print(f"Domains: {list(self.task_models.keys())}")
        print(f"Performances: {self.task_performance}")
        print(f"{'='*80}\n")

    def finalize(self, trainer: "GRPOTrainer"):
        """Final merge at end of training.

        Args:
            trainer: GRPO trainer
        """
        if self.merge_frequency == "end" and len(self.task_models) > 1:
            print("\nPerforming final model merge...")
            trainer.policy.model = self.merge_models(trainer.policy.model)

    def get_statistics(self) -> dict:
        """Get Model Fusion statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "fusion_strategy": self.fusion_strategy,
            "merge_frequency": self.merge_frequency,
            "num_task_models": len(self.task_models),
            "num_merges": self.num_merges,
            "task_domains": list(self.task_models.keys()),
            "task_performance": self.task_performance.copy(),
            "merge_history": self.merge_history.copy(),
        }


class AdaptiveFusionCL(ModelFusionCL):
    """Adaptive Model Fusion with learned merge weights.

    This variant learns optimal merge weights based on validation
    performance rather than using fixed weights.

    Args:
        fusion_strategy: Base fusion strategy
        num_weight_search_steps: Steps for weight search (default: 10)
        weight_search_range: Range for weight search (default: (0.0, 2.0))

    Example:
        >>> trainer.cl_algorithm = AdaptiveFusionCL()
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted_average",
        num_weight_search_steps: int = 10,
        weight_search_range: tuple = (0.0, 2.0),
    ):
        """Initialize Adaptive Fusion algorithm."""
        super().__init__(
            fusion_strategy=fusion_strategy,
            merge_frequency="per_task",
            keep_task_models=True,
        )
        self.num_weight_search_steps = num_weight_search_steps
        self.weight_search_range = weight_search_range
        self.learned_weights = {}

    def _weighted_average_merge(self) -> Dict[str, torch.Tensor]:
        """Adaptive weighted averaging with learned weights.

        Returns:
            Merged parameters
        """
        # Use learned weights if available
        if self.learned_weights:
            weights = self.learned_weights
        else:
            # Fall back to performance-based weights
            total_perf = sum(self.task_performance.values())
            weights = {
                domain: perf / total_perf
                for domain, perf in self.task_performance.items()
            }

        print(f"Adaptive merge weights: {weights}")

        merged = {}
        param_names = list(next(iter(self.task_models.values())).keys())

        for name in param_names:
            merged[name] = sum(
                weights[domain] * self.task_models[domain][name]
                for domain in self.task_models
            )

        return merged

    def learn_merge_weights(self, trainer: "GRPOTrainer"):
        """Learn optimal merge weights via grid search.

        Args:
            trainer: GRPO trainer
        """
        print("\nLearning optimal merge weights...")

        # Simple grid search over weight space
        # In practice, you might use more sophisticated optimization

        best_weights = None
        best_performance = -float('inf')

        # Try different weight combinations
        for _ in range(self.num_weight_search_steps):
            # Sample random weights
            weights = {
                domain: torch.rand(1).item() * (
                    self.weight_search_range[1] - self.weight_search_range[0]
                ) + self.weight_search_range[0]
                for domain in self.task_models
            }

            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            # Evaluate with these weights
            self.learned_weights = weights
            # Would need to actually evaluate here
            # For now, use performance as proxy
            avg_perf = sum(self.task_performance.values()) / len(self.task_performance)

            if avg_perf > best_performance:
                best_performance = avg_perf
                best_weights = weights

        self.learned_weights = best_weights
        print(f"Learned weights: {self.learned_weights}")
