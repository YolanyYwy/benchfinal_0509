"""Data loader for tau2-bench tasks across multiple domains."""

import json
import random
from pathlib import Path
from typing import Iterator, Optional

from AGentCL.data_model.tasks import Task

from .config import GRPOConfig


class TaskDataLoader:
    """Load and manage tasks from tau2-bench domains.

    Supports loading tasks from airline, retail, and telecom domains,
    with configurable task ordering and sampling strategies.
    """

    def __init__(self, config: GRPOConfig, data_root: Optional[Path] = None):
        """Initialize task data loader.

        Args:
            config: GRPO configuration
            data_root: Root directory for data files (default: auto-detect from project structure)
        """
        self.config = config

        # Auto-detect data root if not provided
        if data_root is None:
            # Assume we're in src/tau2/continual_learning, go up to project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            data_root = project_root / "data" / "tau2" / "domains"

        self.data_root = Path(data_root)

        # Load tasks from all domains
        self.tasks_by_domain = self._load_tasks()

        # Split tasks into train/eval
        self.train_tasks_by_domain = {}
        self.eval_tasks_by_domain = {}
        self._split_tasks()

    def _load_tasks(self) -> dict[str, list[Task]]:
        """Load tasks from all configured domains.

        Returns:
            Dictionary mapping domain name to list of Task objects
        """
        tasks = {}

        for domain in self.config.task_order:
            # Use train_tasks.json for all domains
            task_file = self.data_root / domain / "train_tasks.json"

            # Fallback to tasks.json if train_tasks.json doesn't exist
            if not task_file.exists():
                print(f"Warning: train_tasks.json not found for {domain}, falling back to tasks.json")
                task_file = self.data_root / domain / "tasks.json"

            # Load tasks
            if not task_file.exists():
                print(f"Warning: task file not found for {domain}, skipping")
                continue

            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)

            # Convert to Task objects
            domain_tasks = [Task(**t) for t in task_data]

            # Limit number of tasks if configured
            if self.config.max_tasks_per_domain is not None:
                domain_tasks = domain_tasks[:self.config.max_tasks_per_domain]

            tasks[domain] = domain_tasks

            print(f"Loaded {len(domain_tasks)} tasks from {domain} domain (from {task_file.name})")

        return tasks

    def _split_tasks(self):
        """Split tasks into train and eval sets.

        Now loads eval tasks from test_tasks.json instead of random splitting.
        """
        for domain, tasks in self.tasks_by_domain.items():
            # Train tasks are already loaded from train_tasks.json
            self.train_tasks_by_domain[domain] = tasks

            # Load eval tasks from test_tasks.json
            test_file = self.data_root / domain / "test_tasks.json"

            if test_file.exists():
                with open(test_file, "r", encoding="utf-8") as f:
                    test_data = json.load(f)

                # Convert to Task objects
                eval_tasks = [Task(**t) for t in test_data]
                self.eval_tasks_by_domain[domain] = eval_tasks

                print(f"{domain}: {len(self.train_tasks_by_domain[domain])} train, "
                      f"{len(self.eval_tasks_by_domain[domain])} eval tasks (from test_tasks.json)")
            else:
                # Fallback: use old random split method if test_tasks.json doesn't exist
                print(f"Warning: test_tasks.json not found for {domain}, using random split")
                tasks_copy = tasks.copy()
                random.shuffle(tasks_copy)

                split_idx = int(len(tasks_copy) * self.config.train_split)

                self.train_tasks_by_domain[domain] = tasks_copy[:split_idx]
                self.eval_tasks_by_domain[domain] = tasks_copy[split_idx:]

                print(f"{domain}: {len(self.train_tasks_by_domain[domain])} train, "
                      f"{len(self.eval_tasks_by_domain[domain])} eval tasks (random split)")


    def get_train_tasks(self, domain: str) -> list[Task]:
        """Get training tasks for a domain.

        Args:
            domain: Domain name (airline, retail, telecom)

        Returns:
            List of training tasks
        """
        if domain not in self.train_tasks_by_domain:
            raise ValueError(f"Unknown domain: {domain}")

        return self.train_tasks_by_domain[domain]

    def get_eval_tasks(self, domain: str) -> list[Task]:
        """Get evaluation tasks for a domain.

        Args:
            domain: Domain name (airline, retail, telecom)

        Returns:
            List of evaluation tasks
        """
        if domain not in self.eval_tasks_by_domain:
            raise ValueError(f"Unknown domain: {domain}")

        return self.eval_tasks_by_domain[domain]

    def get_task_iterator(self, domain: str, shuffle: bool = True, split: str = "train") -> Iterator[Task]:
        """Get iterator over tasks for a domain.

        Args:
            domain: Domain name
            shuffle: Whether to shuffle tasks
            split: Which split to use ("train" or "eval")

        Returns:
            Iterator over tasks
        """
        if split == "train":
            tasks = self.get_train_tasks(domain)
        elif split == "eval":
            tasks = self.get_eval_tasks(domain)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")

        tasks_copy = tasks.copy()
        if shuffle:
            random.shuffle(tasks_copy)

        return iter(tasks_copy)

    def sample_batch(self, domain: str, batch_size: int, split: str = "train") -> list[Task]:
        """Sample a batch of tasks from a domain.

        Args:
            domain: Domain name
            batch_size: Number of tasks to sample
            split: Which split to use ("train" or "eval")

        Returns:
            List of sampled tasks
        """
        if split == "train":
            tasks = self.get_train_tasks(domain)
        elif split == "eval":
            tasks = self.get_eval_tasks(domain)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")

        # Sample with replacement if batch_size > len(tasks)
        if batch_size > len(tasks):
            return random.choices(tasks, k=batch_size)
        else:
            return random.sample(tasks, batch_size)

    def get_num_tasks(self, domain: str, split: str = "train") -> int:
        """Get number of tasks in a domain.

        Args:
            domain: Domain name
            split: Which split to use ("train" or "eval")

        Returns:
            Number of tasks
        """
        if split == "train":
            return len(self.get_train_tasks(domain))
        elif split == "eval":
            return len(self.get_eval_tasks(domain))
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")

    def get_all_domains(self) -> list[str]:
        """Get list of all loaded domains.

        Returns:
            List of domain names
        """
        return list(self.tasks_by_domain.keys())

    def get_task_by_id(self, domain: str, task_id: str) -> Optional[Task]:
        """Get a specific task by ID.

        Args:
            domain: Domain name
            task_id: Task ID

        Returns:
            Task object if found, None otherwise
        """
        if domain not in self.tasks_by_domain:
            return None

        for task in self.tasks_by_domain[domain]:
            if task.id == task_id:
                return task

        return None
