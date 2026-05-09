"""Reward oracle for computing rewards using tau2-bench evaluators."""

import uuid
from datetime import datetime
from typing import Optional

from AGentCL.data_model.message import Message
from AGentCL.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from AGentCL.data_model.tasks import Task
from AGentCL.evaluator.evaluator import EvaluationType, evaluate_simulation
from AGentCL.registry import registry


# 使用滑动窗口评估的 domain（这些 domain 使用 expected_states 而不是 actions）
TRAJECTORY_EVAL_DOMAINS = {"delivery", "instore", "ota"}


class Trajectory:
    """Represents a single agent trajectory (sequence of messages)."""

    def __init__(
        self,
        task_id: str,
        messages: list[Message],
        termination_reason: TerminationReason,
        cost: float = 0.0
    ):
        """Initialize trajectory.

        Args:
            task_id: ID of the task this trajectory is for
            messages: List of messages in the trajectory
            termination_reason: How the trajectory terminated
            cost: API cost for this trajectory (0 for open-source models)
        """
        self.task_id = task_id
        self.messages = messages
        self.termination_reason = termination_reason
        self.cost = cost

    def to_dict(self) -> dict:
        """Convert trajectory to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "messages": [msg.model_dump() for msg in self.messages],
            "termination_reason": self.termination_reason.value,
            "cost": self.cost
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Create trajectory from dictionary."""
        from AGentCL.data_model.message import (
            AssistantMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )

        # Reconstruct messages
        messages = []
        for msg_data in data["messages"]:
            role = msg_data.get("role")
            if role == "system":
                messages.append(SystemMessage(**msg_data))
            elif role == "user":
                messages.append(UserMessage(**msg_data))
            elif role == "assistant":
                messages.append(AssistantMessage(**msg_data))
            elif role == "tool":
                messages.append(ToolMessage(**msg_data))

        return cls(
            task_id=data["task_id"],
            messages=messages,
            termination_reason=TerminationReason(data["termination_reason"]),
            cost=data.get("cost", 0.0)
        )


class RewardOracle:
    """Compute rewards for trajectories using tau2-bench evaluators.

    This oracle uses the existing tau2-bench evaluation infrastructure
    to compute rewards for agent trajectories. It supports multiple
    evaluation types (environment, action, communication, NL assertions).
    """

    def __init__(self, evaluation_type: str = "ALL", task_order: Optional[list[str]] = None):
        self.evaluation_type = EvaluationType(evaluation_type.lower())

        domains = task_order if task_order is not None else registry.get_domains()

        env_constructors = {}
        missing = []
        for d in domains:
            try:
                env_constructors[d] = registry.get_env_constructor(d)
            except KeyError:
                missing.append(d)

        if missing:
            available = registry.get_domains()
            raise KeyError(
                f"[RewardOracle] Domains not found in registry: {missing}. "
                f"Available domains: {available}"
            )

        self.env_constructors = env_constructors
        self._trajectory_evaluator = None

    def _get_trajectory_evaluator(self):
        """懒加载滑动窗口评估器"""
        if self._trajectory_evaluator is None:
            from AGentCL.evaluator.evaluator_traj import TrajectoryEvaluator
            self._trajectory_evaluator = TrajectoryEvaluator()
        return self._trajectory_evaluator

    def compute_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
        solo_mode: bool = False
    ) -> RewardInfo:
        """Compute reward for a single trajectory.

        Args:
            task: Task definition with evaluation criteria
            trajectory: Agent trajectory to evaluate
            domain: Domain name (airline, retail, telecom, instore, delivery, ota)
            solo_mode: Whether agent operated in solo mode

        Returns:
            RewardInfo object with reward and detailed breakdown
        """
        # 对于 instore/delivery/ota，使用滑动窗口评估
        if domain in TRAJECTORY_EVAL_DOMAINS:
            return self._compute_trajectory_reward(task, trajectory, domain)

        # 对于 airline/retail/telecom，使用原有的任务完成度评估
        return self._compute_task_completion_reward(task, trajectory, domain, solo_mode)

    def _compute_trajectory_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
    ) -> RewardInfo:
        """使用滑动窗口评估 instore/delivery/ota 任务"""
        try:
            evaluator = self._get_trajectory_evaluator()
            return evaluator.calculate_reward(
                task=task,
                messages=trajectory.messages,
                window_size=20,
                overlap=2,
            )
        except Exception as e:
            print(f"[RewardOracle] Trajectory evaluation failed: {e}")
            # 回退到默认评估
            term_reason = trajectory.termination_reason
            if hasattr(term_reason, 'value'):
                term_reason = term_reason.value

            if term_reason in ['user_stop', 'agent_stop', 'TerminationReason.USER_STOP', 'TerminationReason.AGENT_STOP']:
                default_reward = 0.5
            else:
                default_reward = 0.0

            return RewardInfo(
                reward=default_reward,
                reward_basis=None,
                info={
                    "note": f"Trajectory evaluation failed: {str(e)[:200]}",
                    "fallback_reward": True,
                },
            )

    def _compute_task_completion_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
        solo_mode: bool = False
    ) -> RewardInfo:
        """使用任务完成度评估 airline/retail/telecom 任务"""
        # Create simulation object for evaluation
        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            messages=trajectory.messages,
            termination_reason=trajectory.termination_reason,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration=0.0,
            agent_cost=trajectory.cost,
            user_cost=0.0,
            seed=0
        )

        try:
            # Evaluate using tau2-bench evaluator
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=self.evaluation_type,
                solo_mode=solo_mode,
                domain=domain
            )

            # Reward shaping: replace multiplicative combination with additive mean.
            # Original evaluator multiplies sub-scores (env * action * communicate),
            # so any single 0 kills the whole reward — almost always 0 or 1.
            # Additive mean gives partial credit and produces a continuous signal.
            if reward_info.reward_breakdown and len(reward_info.reward_breakdown) > 1:
                sub_scores = list(reward_info.reward_breakdown.values())
                shaped_reward = sum(sub_scores) / len(sub_scores)
                # Preserve original binary reward in info for analysis
                reward_info = RewardInfo(
                    reward=shaped_reward,
                    db_check=reward_info.db_check,
                    env_assertions=reward_info.env_assertions,
                    action_checks=reward_info.action_checks,
                    nl_assertions=reward_info.nl_assertions,
                    communicate_checks=reward_info.communicate_checks,
                    reward_basis=reward_info.reward_basis,
                    reward_breakdown=reward_info.reward_breakdown,
                    info={
                        **(reward_info.info or {}),
                        "original_reward": reward_info.reward,
                        "shaped_reward": shaped_reward,
                        "sub_scores": sub_scores,
                    },
                )

        except ValueError as e:
            # 环境状态验证失败（常见于 tool call 结果不一致）
            # 根据 termination_reason 给一个合理的默认 reward
            term_reason = trajectory.termination_reason
            if hasattr(term_reason, 'value'):
                term_reason = term_reason.value

            # 如果是正常结束（USER_STOP 或 AGENT_STOP），给 0.5 作为中性 reward
            # 否则给 0.0
            if term_reason in ['user_stop', 'agent_stop', 'TerminationReason.USER_STOP', 'TerminationReason.AGENT_STOP']:
                default_reward = 0.5
            else:
                default_reward = 0.0

            reward_info = RewardInfo(
                reward=default_reward,
                reward_basis=None,
                info={
                    "note": f"Environment validation failed: {str(e)[:200]}",
                    "fallback_reward": True,
                    "termination_reason": term_reason,
                },
            )

        return reward_info

    def compute_batch_rewards(
        self,
        task: Task,
        trajectories: list[Trajectory],
        domain: str,
        solo_mode: bool = False
    ) -> list[float]:
        """Compute rewards for multiple trajectories.

        Args:
            task: Task definition
            trajectories: List of trajectories to evaluate
            domain: Domain name
            solo_mode: Whether agent operated in solo mode

        Returns:
            List of reward values (one per trajectory)
        """
        rewards = []
        for traj in trajectories:
            reward_info = self.compute_reward(task, traj, domain, solo_mode)
            rewards.append(reward_info.reward)

        return rewards

    def compute_batch_rewards_with_info(
        self,
        task: Task,
        trajectories: list[Trajectory],
        domain: str,
        solo_mode: bool = False
    ) -> list[RewardInfo]:
        """Compute rewards with full info for multiple trajectories.

        Args:
            task: Task definition
            trajectories: List of trajectories to evaluate
            domain: Domain name
            solo_mode: Whether agent operated in solo mode

        Returns:
            List of RewardInfo objects with detailed breakdown
        """
        reward_infos = []
        for traj in trajectories:
            reward_info = self.compute_reward(task, traj, domain, solo_mode)
            reward_infos.append(reward_info)

        return reward_infos

    def get_reward_breakdown(self, reward_info: RewardInfo) -> dict:
        """Extract reward breakdown from RewardInfo.

        Args:
            reward_info: RewardInfo object

        Returns:
            Dictionary with reward components
        """
        breakdown = {
            "total_reward": reward_info.reward,
            "reward_basis": [rb.value for rb in reward_info.reward_basis] if reward_info.reward_basis else []
        }

        # Add component-specific rewards if available
        if hasattr(reward_info, "reward_breakdown") and reward_info.reward_breakdown:
            breakdown.update(reward_info.reward_breakdown)

        # Add check results
        if reward_info.db_check:
            breakdown["db_match"] = reward_info.db_check.db_match
            breakdown["db_reward"] = reward_info.db_check.db_reward

        if reward_info.action_checks:
            breakdown["action_matches"] = sum(1 for ac in reward_info.action_checks if ac.action_match)
            breakdown["action_total"] = len(reward_info.action_checks)

        if reward_info.env_assertions:
            breakdown["env_assertion_passes"] = sum(1 for ea in reward_info.env_assertions if ea.result)
            breakdown["env_assertion_total"] = len(reward_info.env_assertions)

        return breakdown
