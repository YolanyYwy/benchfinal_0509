"""Trajectory logger for saving detailed trajectory information to files."""

import json
from pathlib import Path
from typing import Optional

from AGentCL.data_model.message import AssistantMessage, UserMessage, ToolMessage
from AGentCL.data_model.tasks import Task

from .reward_oracle import Trajectory


class TrajectoryLogger:
    """Logger for saving trajectory details to files.

    This logger saves detailed trajectory information including:
    - Task information
    - All messages (user, assistant, tool)
    - Rewards and evaluation results
    - Termination reasons
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        """Initialize trajectory logger.

        Args:
            log_dir: Directory to save trajectory logs
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.trajectory_log_file = self.log_dir / "trajectories.jsonl"
            self.readable_log_file = self.log_dir / "trajectories_readable.txt"

            # Create/clear log files
            self.trajectory_log_file.touch()
            self.readable_log_file.touch()

    def log_trajectory(
        self,
        task: Task,
        trajectory: Trajectory,
        reward: float,
        domain: str,
        step: int,
        sample_idx: int = 0,
        reward_info: Optional[any] = None,
    ):
        """Log a single trajectory.

        Args:
            task: Task information
            trajectory: Generated trajectory
            reward: Computed reward
            domain: Domain name
            step: Training step number
            sample_idx: Sample index (for multiple samples per task)
            reward_info: Optional reward info with detailed breakdown
        """
        if not self.enabled:
            return

        # Determine success/failure status
        is_success = reward > 0.5
        status = "SUCCESS" if is_success else "FAILED"

        # Create structured log entry
        log_entry = {
            "step": step,
            "domain": domain,
            "task_id": task.id,
            "sample_idx": sample_idx,
            "reward": reward,
            "status": status,
            "termination_reason": trajectory.termination_reason,
            "num_messages": len(trajectory.messages),
            "messages": self._serialize_messages(trajectory.messages),
        }

        # Add reward breakdown if available
        if reward_info and hasattr(reward_info, 'info') and reward_info.info:
            log_entry["reward_details"] = reward_info.info

        # Append to JSONL file
        with open(self.trajectory_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Append to readable text file
        self._write_readable_log(task, trajectory, reward, domain, step, sample_idx, reward_info)

    def _serialize_messages(self, messages):
        """Serialize messages to JSON-compatible format.

        Args:
            messages: List of messages

        Returns:
            List of serialized messages
        """
        serialized = []
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }

            # Add tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in msg.tool_calls
                ]

            # Add tool call ID for tool messages
            if isinstance(msg, ToolMessage):
                msg_dict["tool_call_id"] = msg.id

            serialized.append(msg_dict)

        return serialized

    def _write_readable_log(
        self,
        task: Task,
        trajectory: Trajectory,
        reward: float,
        domain: str,
        step: int,
        sample_idx: int,
        reward_info: Optional[any] = None,
    ):
        """Write human-readable trajectory log.

        Args:
            task: Task information
            trajectory: Generated trajectory
            reward: Computed reward
            domain: Domain name
            step: Training step number
            sample_idx: Sample index
            reward_info: Optional reward info with detailed breakdown
        """
        # Determine success/failure status
        is_success = reward > 0.5
        status_symbol = "✓" if is_success else "✗"
        status_text = "SUCCESS" if is_success else "FAILED"

        with open(self.readable_log_file, "a", encoding="utf-8") as f:
            # Header with status
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{status_symbol}] {status_text} | Step {step} | Domain: {domain} | Task: {task.id} | Sample: {sample_idx}\n")
            f.write(f"Reward: {reward:.4f} | Termination: {trajectory.termination_reason}\n")

            # Print reward breakdown if available
            if reward_info and hasattr(reward_info, 'info') and reward_info.info:
                info = reward_info.info
                if "completion_details" in info:
                    details = info["completion_details"]
                    f.write(f"H1={details.get('h1_score', 0):.2f}, "
                           f"H2={details.get('h2_score', 0):.2f}, "
                           f"H3={details.get('h3_score', 0):.2f}\n")

            f.write("="*80 + "\n\n")

            # Task description
            f.write("Task Description:\n")
            f.write(f"{task.user_scenario}\n\n")

            # Messages
            f.write("Trajectory:\n")
            f.write("-"*80 + "\n")

            for i, msg in enumerate(trajectory.messages, 1):
                role_display = msg.role.upper()

                # Check if this is a tool call message
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls

                if has_tool_calls:
                    f.write(f"\n[Message {i}] {role_display} (Tool Call)\n")
                    for tc in msg.tool_calls:
                        f.write(f"  Tool: {tc.name}\n")
                        f.write(f"  Arguments: {json.dumps(tc.arguments, ensure_ascii=False, indent=4)}\n")
                elif isinstance(msg, ToolMessage):
                    f.write(f"\n[Message {i}] TOOL (Response)\n")
                    f.write(f"  Tool Call ID: {msg.id}\n")
                    if msg.content:
                        # Truncate long tool responses
                        content = msg.content
                        if len(content) > 500:
                            content = content[:500] + "... (truncated)"
                        f.write(f"  Result: {content}\n")
                else:
                    f.write(f"\n[Message {i}] {role_display}\n")
                    if msg.content:
                        f.write(f"{msg.content}\n")

                f.write("-"*80 + "\n")

            f.write("\n")

    def log_batch_summary(
        self,
        step: int,
        domain: str,
        num_tasks: int,
        avg_reward: float,
        max_reward: float,
        min_reward: float,
    ):
        """Log summary for a batch of trajectories.

        Args:
            step: Training step number
            domain: Domain name
            num_tasks: Number of tasks in batch
            avg_reward: Average reward
            max_reward: Maximum reward
            min_reward: Minimum reward
        """
        if not self.enabled:
            return

        with open(self.readable_log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "#"*80 + "\n")
            f.write(f"BATCH SUMMARY - Step {step} | Domain: {domain}\n")
            f.write(f"Tasks: {num_tasks} | Avg Reward: {avg_reward:.4f} | "
                   f"Max: {max_reward:.4f} | Min: {min_reward:.4f}\n")
            f.write("#"*80 + "\n\n")

    def close(self):
        """Close the logger (placeholder for future cleanup)."""
        pass
