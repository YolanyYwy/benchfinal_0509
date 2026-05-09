"""Trajectory buffer for storing and replaying agent trajectories."""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from AGentCL.data_model.tasks import Task

from .config import GRPOConfig
from .reward_oracle import Trajectory


class TrajectoryRecord:
    """Record of a trajectory with metadata."""

    def __init__(
        self,
        domain: str,
        task: Task,
        trajectory: Trajectory,
        reward: float,
        timestamp: Optional[datetime] = None
    ):
        """Initialize trajectory record.

        Args:
            domain: Domain name (airline, retail, telecom)
            task: Task definition
            trajectory: Agent trajectory
            reward: Reward achieved
            timestamp: When this trajectory was generated
        """
        self.domain = domain
        self.task = task
        self.trajectory = trajectory
        self.reward = reward
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        """Convert record to dictionary for serialization."""
        return {
            "domain": self.domain,
            "task": self.task.model_dump(),
            "trajectory": self.trajectory.to_dict(),
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryRecord":
        """Create record from dictionary."""
        return cls(
            domain=data["domain"],
            task=Task(**data["task"]),
            trajectory=Trajectory.from_dict(data["trajectory"]),
            reward=data["reward"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class TrajectoryBuffer:
    """Store and manage trajectories for continual learning.

    This buffer stores trajectories from different domains and supports
    sampling for experience replay. Trajectories are stored with their
    rewards and metadata for analysis.
    """

    def __init__(self, config: GRPOConfig):
        """Initialize trajectory buffer.

        Args:
            config: GRPO configuration
        """
        self.config = config
        self.buffer: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        self.max_size = config.replay_buffer_size

    def add(
        self,
        domain: str,
        task: Task,
        trajectory: Trajectory,
        reward: float
    ):
        """Add a trajectory to the buffer.

        Args:
            domain: Domain name
            task: Task definition
            trajectory: Agent trajectory
            reward: Reward achieved
        """
        record = TrajectoryRecord(
            domain=domain,
            task=task,
            trajectory=trajectory,
            reward=reward,
            timestamp=datetime.now()
        )

        self.buffer[domain].append(record)

        # Evict oldest if over capacity
        if len(self.buffer[domain]) > self.max_size:
            self.buffer[domain].pop(0)

    def sample(
        self,
        domain: str,
        num_samples: int,
        strategy: str = "random"
    ) -> list[TrajectoryRecord]:
        """Sample trajectories from buffer.

        Args:
            domain: Domain to sample from
            num_samples: Number of samples to draw
            strategy: Sampling strategy (random, high_reward, recent)

        Returns:
            List of trajectory records
        """
        if domain not in self.buffer or len(self.buffer[domain]) == 0:
            return []

        domain_buffer = self.buffer[domain]

        # If requesting more than available, return all
        if num_samples >= len(domain_buffer):
            return domain_buffer.copy()

        # Apply sampling strategy
        if strategy == "random":
            return random.sample(domain_buffer, num_samples)

        elif strategy == "high_reward":
            # Sample from top-k high-reward trajectories
            sorted_buffer = sorted(domain_buffer, key=lambda x: x.reward, reverse=True)
            top_k = sorted_buffer[:min(num_samples * 2, len(sorted_buffer))]
            return random.sample(top_k, num_samples)

        elif strategy == "recent":
            # Sample from most recent trajectories
            sorted_buffer = sorted(domain_buffer, key=lambda x: x.timestamp, reverse=True)
            recent_k = sorted_buffer[:min(num_samples * 2, len(sorted_buffer))]
            return random.sample(recent_k, num_samples)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def sample_multi_domain(
        self,
        domains: list[str],
        num_samples_per_domain: int,
        strategy: str = "random"
    ) -> list[TrajectoryRecord]:
        """Sample trajectories from multiple domains.

        Args:
            domains: List of domains to sample from
            num_samples_per_domain: Number of samples per domain
            strategy: Sampling strategy

        Returns:
            List of trajectory records from all domains
        """
        all_samples = []
        for domain in domains:
            samples = self.sample(domain, num_samples_per_domain, strategy)
            all_samples.extend(samples)

        return all_samples

    def get_size(self, domain: Optional[str] = None) -> int:
        """Get buffer size.

        Args:
            domain: Specific domain (None = total across all domains)

        Returns:
            Number of trajectories in buffer
        """
        if domain is None:
            return sum(len(records) for records in self.buffer.values())
        else:
            return len(self.buffer.get(domain, []))

    def get_domains(self) -> list[str]:
        """Get list of domains with stored trajectories.

        Returns:
            List of domain names
        """
        return list(self.buffer.keys())

    def get_statistics(self, domain: Optional[str] = None) -> dict:
        """Get buffer statistics.

        Args:
            domain: Specific domain (None = all domains)

        Returns:
            Dictionary with statistics
        """
        if domain is not None:
            records = self.buffer.get(domain, [])
            if not records:
                return {"size": 0}

            rewards = [r.reward for r in records]
            return {
                "size": len(records),
                "mean_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "high_reward_count": sum(1 for r in rewards if r > 0.5)
            }
        else:
            # Aggregate statistics across all domains
            stats = {}
            for dom in self.buffer.keys():
                stats[dom] = self.get_statistics(dom)

            return stats

    def clear(self, domain: Optional[str] = None):
        """Clear buffer.

        Args:
            domain: Specific domain to clear (None = clear all)
        """
        if domain is None:
            self.buffer.clear()
        elif domain in self.buffer:
            self.buffer[domain].clear()

    def save(self, path: str):
        """Save buffer to disk.

        Args:
            path: Path to save file (JSON format)
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            domain: [record.to_dict() for record in records]
            for domain, records in self.buffer.items()
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved trajectory buffer to {save_path} ({self.get_size()} trajectories)")

    def load(self, path: str):
        """Load buffer from disk.

        Args:
            path: Path to load file (JSON format)
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Buffer file not found: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.buffer.clear()
        for domain, records_data in data.items():
            self.buffer[domain] = [TrajectoryRecord.from_dict(r) for r in records_data]

        print(f"Loaded trajectory buffer from {load_path} ({self.get_size()} trajectories)")

    def export_trajectories(
        self,
        output_path: str,
        domain: Optional[str] = None,
        min_reward: Optional[float] = None
    ):
        """Export trajectories to file for analysis.

        Args:
            output_path: Path to output file
            domain: Specific domain (None = all domains)
            min_reward: Minimum reward threshold (None = no filtering)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect trajectories to export
        trajectories_to_export = []

        domains_to_export = [domain] if domain else self.get_domains()

        for dom in domains_to_export:
            for record in self.buffer.get(dom, []):
                if min_reward is None or record.reward >= min_reward:
                    trajectories_to_export.append({
                        "domain": record.domain,
                        "task_id": record.task.id,
                        "reward": record.reward,
                        "timestamp": record.timestamp.isoformat(),
                        "trajectory": record.trajectory.to_dict()
                    })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectories_to_export, f, indent=2)

        print(f"Exported {len(trajectories_to_export)} trajectories to {output_path}")
