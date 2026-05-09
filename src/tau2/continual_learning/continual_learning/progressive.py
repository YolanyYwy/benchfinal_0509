"""Progressive Neural Networks for continual learning.

Progressive Neural Networks prevent forgetting by allocating new network
capacity for each task while keeping previous task parameters frozen.
Lateral connections allow knowledge transfer from old to new columns.

References:
    - Rusu, A. A., et al. (2016). Progressive Neural Networks. arXiv.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from AGentCL.data_model.tasks import Task

from .base import CLAlgorithm

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class ProgressiveNetsCL(CLAlgorithm):
    """Progressive Neural Networks continual learning algorithm.

    Progressive Networks add a new "column" (sub-network) for each task,
    while freezing parameters from previous tasks. Lateral connections
    allow new columns to leverage knowledge from old columns.

    Architecture:
        Task 1: [Column 1]
        Task 2: [Column 1 (frozen)] -> [Column 2]
        Task 3: [Column 1 (frozen)] -> [Column 2 (frozen)] -> [Column 3]

    Args:
        adapter_size: Size of adapter layers (default: 256)
            - Smaller = less capacity per task
            - Larger = more capacity but more parameters
        use_lateral_connections: Whether to use lateral connections (default: True)
        freeze_previous_columns: Whether to freeze previous columns (default: True)
        max_columns: Maximum number of columns (default: None = unlimited)

    Example:
        >>> config = GRPOConfig(cl_algorithm="progressive")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = ProgressiveNetsCL(adapter_size=256)
        >>> trainer.train()

    Note:
        This is a simplified implementation that adds adapter layers
        rather than full network columns. For full Progressive Networks,
        you would need to modify the base model architecture.
    """

    def __init__(
        self,
        adapter_size: int = 256,
        use_lateral_connections: bool = True,
        freeze_previous_columns: bool = True,
        max_columns: int = None,
    ):
        """Initialize Progressive Networks algorithm.

        Args:
            adapter_size: Size of adapter layers
            use_lateral_connections: Whether to use lateral connections
            freeze_previous_columns: Whether to freeze previous columns
            max_columns: Maximum number of columns
        """
        self.adapter_size = adapter_size
        self.use_lateral_connections = use_lateral_connections
        self.freeze_previous_columns = freeze_previous_columns
        self.max_columns = max_columns

        # Track columns (adapters) for each task
        self.columns = []  # List of adapter modules
        self.column_domains = []  # Domain for each column
        self.current_column_idx = -1

        # Statistics
        self.total_parameters = 0
        self.parameters_per_column = []

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """No batch augmentation for Progressive Networks.

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Unmodified list of tasks
        """
        return new_tasks

    def create_new_column(
        self,
        model: torch.nn.Module,
        domain: str,
    ):
        """Create a new column (adapter) for the current task.

        Args:
            model: Base model
            domain: Domain for this column
        """
        print(f"\nCreating new column for {domain}...")

        # Check max columns limit
        if self.max_columns and len(self.columns) >= self.max_columns:
            print(f"Warning: Reached max columns ({self.max_columns}). Reusing last column.")
            return

        # Create adapter module
        # In a full implementation, this would be a complete network column
        # Here we use a simplified adapter approach
        adapter = TaskAdapter(
            hidden_size=self.adapter_size,
            num_previous_columns=len(self.columns),
            use_lateral=self.use_lateral_connections,
        )

        # Move to same device as model
        device = next(model.parameters()).device
        adapter = adapter.to(device)

        # Add to columns
        self.columns.append(adapter)
        self.column_domains.append(domain)
        self.current_column_idx = len(self.columns) - 1

        # Count parameters
        num_params = sum(p.numel() for p in adapter.parameters())
        self.parameters_per_column.append(num_params)
        self.total_parameters += num_params

        print(f"Column {self.current_column_idx} created with {num_params:,} parameters")

    def freeze_previous_columns(self):
        """Freeze all columns except the current one."""
        if not self.freeze_previous_columns:
            return

        for idx, column in enumerate(self.columns[:-1]):
            for param in column.parameters():
                param.requires_grad = False

        print(f"Frozen {len(self.columns) - 1} previous columns")

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after each training step.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        # Progressive Networks doesn't need per-step operations
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Hook called after finishing a task.

        Creates new column and freezes previous ones.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score (unused in progressive)
        """
        if not trainer.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"Progressive Networks Post-Task Processing for {domain}")
        print(f"{'='*80}")

        # Freeze current column
        if self.current_column_idx >= 0:
            self.freeze_previous_columns()

        # Statistics
        print(f"Total columns: {len(self.columns)}")
        print(f"Total parameters: {self.total_parameters:,}")
        print(f"Parameters per column: {self.parameters_per_column}")
        print(f"Column domains: {self.column_domains}")
        print(f"{'='*80}\n")

    def get_statistics(self) -> dict:
        """Get Progressive Networks statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "adapter_size": self.adapter_size,
            "use_lateral_connections": self.use_lateral_connections,
            "num_columns": len(self.columns),
            "total_parameters": self.total_parameters,
            "parameters_per_column": self.parameters_per_column.copy(),
            "column_domains": self.column_domains.copy(),
        }


class TaskAdapter(nn.Module):
    """Adapter module for Progressive Networks.

    This is a simplified adapter that can be inserted into the model.
    In a full Progressive Networks implementation, this would be a
    complete network column.

    Args:
        hidden_size: Size of hidden layers
        num_previous_columns: Number of previous columns for lateral connections
        use_lateral: Whether to use lateral connections
    """

    def __init__(
        self,
        hidden_size: int,
        num_previous_columns: int = 0,
        use_lateral: bool = True,
    ):
        """Initialize task adapter."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_previous_columns = num_previous_columns
        self.use_lateral = use_lateral

        # Main adapter layers
        self.adapter_down = nn.Linear(hidden_size, hidden_size // 4)
        self.adapter_up = nn.Linear(hidden_size // 4, hidden_size)
        self.activation = nn.ReLU()

        # Lateral connections from previous columns
        if use_lateral and num_previous_columns > 0:
            self.lateral_connections = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size // 4)
                for _ in range(num_previous_columns)
            ])
        else:
            self.lateral_connections = None

    def forward(self, x, previous_column_outputs=None):
        """Forward pass through adapter.

        Args:
            x: Input tensor
            previous_column_outputs: Outputs from previous columns (for lateral connections)

        Returns:
            Adapted output
        """
        # Main adapter path
        h = self.adapter_down(x)
        h = self.activation(h)

        # Add lateral connections
        if self.lateral_connections and previous_column_outputs:
            for i, (lateral, prev_output) in enumerate(
                zip(self.lateral_connections, previous_column_outputs)
            ):
                h = h + lateral(prev_output)

        # Project back
        output = self.adapter_up(h)

        # Residual connection
        output = output + x

        return output


class DynamicExpansionCL(CLAlgorithm):
    """Dynamic network expansion based on task difficulty.

    This variant dynamically decides how much capacity to add based on
    the difficulty of the new task (measured by initial performance).

    Args:
        base_adapter_size: Base size for adapters (default: 256)
        min_adapter_size: Minimum adapter size (default: 128)
        max_adapter_size: Maximum adapter size (default: 512)
        difficulty_threshold: Performance threshold for expansion (default: 0.3)

    Example:
        >>> trainer.cl_algorithm = DynamicExpansionCL(base_adapter_size=256)
    """

    def __init__(
        self,
        base_adapter_size: int = 256,
        min_adapter_size: int = 128,
        max_adapter_size: int = 512,
        difficulty_threshold: float = 0.3,
    ):
        """Initialize Dynamic Expansion algorithm."""
        self.base_adapter_size = base_adapter_size
        self.min_adapter_size = min_adapter_size
        self.max_adapter_size = max_adapter_size
        self.difficulty_threshold = difficulty_threshold

        # Track adapters
        self.adapters = []
        self.adapter_sizes = []
        self.adapter_domains = []

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """No batch augmentation.

        Args:
            new_tasks: New tasks
            current_domain: Current domain

        Returns:
            Unmodified tasks
        """
        return new_tasks

    def determine_adapter_size(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ) -> int:
        """Determine adapter size based on task difficulty.

        Args:
            trainer: GRPO trainer
            domain: Domain to evaluate

        Returns:
            Adapter size
        """
        # Evaluate initial performance on new task
        metrics = trainer.evaluate_task(domain, num_eval_tasks=10)
        initial_performance = metrics["reward_mean"]

        # Determine size based on difficulty
        if initial_performance < self.difficulty_threshold:
            # Difficult task - use larger adapter
            adapter_size = self.max_adapter_size
            print(f"Difficult task detected (perf={initial_performance:.3f}). Using large adapter.")
        elif initial_performance < 0.6:
            # Medium difficulty - use base adapter
            adapter_size = self.base_adapter_size
            print(f"Medium difficulty task (perf={initial_performance:.3f}). Using base adapter.")
        else:
            # Easy task - use smaller adapter
            adapter_size = self.min_adapter_size
            print(f"Easy task detected (perf={initial_performance:.3f}). Using small adapter.")

        return adapter_size

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """No per-step operations."""
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Create adapter based on task difficulty.

        Args:
            trainer: GRPO trainer
            domain: Completed domain
            performance: Performance score (unused)
        """
        if not trainer.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"Dynamic Expansion Post-Task Processing for {domain}")
        print(f"{'='*80}")

        # Determine adapter size
        adapter_size = self.determine_adapter_size(trainer, domain)

        # Create adapter (simplified - in practice would modify model)
        self.adapter_sizes.append(adapter_size)
        self.adapter_domains.append(domain)

        print(f"Total adapters: {len(self.adapter_sizes)}")
        print(f"Adapter sizes: {self.adapter_sizes}")
        print(f"{'='*80}\n")

    def get_statistics(self) -> dict:
        """Get statistics."""
        return {
            "num_adapters": len(self.adapters),
            "adapter_sizes": self.adapter_sizes.copy(),
            "adapter_domains": self.adapter_domains.copy(),
            "total_parameters": sum(
                size * size // 4 * 2 for size in self.adapter_sizes
            ),
        }
