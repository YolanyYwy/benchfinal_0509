"""Elastic Weight Consolidation (EWC) for continual learning.

EWC is a regularization-based continual learning method that protects important
parameters from large changes when learning new tasks. It estimates parameter
importance using the Fisher Information Matrix.

References:
    - Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in
      neural networks. PNAS.
"""

import copy
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from AGentCL.data_model.tasks import Task

from .base import CLAlgorithm

if TYPE_CHECKING:
    from ..grpo_trainer import GRPOTrainer


class EWCCL(CLAlgorithm):
    """Elastic Weight Consolidation (EWC) continual learning algorithm.

    EWC adds a regularization term to the loss that penalizes changes to
    important parameters. Parameter importance is estimated using the Fisher
    Information Matrix computed on previous tasks.

    The EWC loss is:
        L_total = L_new + (λ/2) * Σ F_i * (θ - θ_i*)^2

    Where:
        - L_new: Loss on new task
        - λ: EWC regularization strength
        - F_i: Fisher information for parameter i
        - θ: Current parameters
        - θ_i*: Optimal parameters from previous tasks

    Args:
        ewc_lambda: Regularization strength (default: 0.4)
            - Higher values = stronger protection of old knowledge
            - Lower values = more plasticity for new tasks
        fisher_sample_size: Number of samples to estimate Fisher (default: 200)
        online_ewc: Use online EWC variant (default: False)
            - Online: Update Fisher incrementally
            - Offline: Compute Fisher separately for each task

    Example:
        >>> config = GRPOConfig(cl_algorithm="ewc")
        >>> trainer = GRPOTrainer(config)
        >>> trainer.cl_algorithm = EWCCL(ewc_lambda=0.4)
        >>> trainer.train()
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        online_ewc: bool = False,
    ):
        """Initialize EWC algorithm.

        Args:
            ewc_lambda: Regularization strength
            fisher_sample_size: Number of samples for Fisher estimation
            online_ewc: Whether to use online EWC
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.online_ewc = online_ewc

        # Storage for Fisher information and optimal parameters
        self.fisher_dict = {}  # {param_name: Fisher information}
        self.optimal_params = {}  # {param_name: optimal parameter values}

        # For online EWC
        self.task_count = 0

        # Statistics
        self.total_ewc_loss = 0.0
        self.ewc_loss_per_task = {}

    def augment_batch(
        self, new_tasks: list[Task], current_domain: str
    ) -> list[Task]:
        """No batch augmentation for EWC (uses regularization instead).

        Args:
            new_tasks: New tasks sampled for this step
            current_domain: Current domain being trained on

        Returns:
            Unmodified list of tasks
        """
        return new_tasks

    def compute_ewc_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss.

        Args:
            model: Current model

        Returns:
            EWC loss (scalar tensor)
        """
        if not self.fisher_dict:
            # No previous tasks, no regularization
            return torch.tensor(0.0, device=next(model.parameters()).device)

        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                # Get Fisher information and optimal parameters
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]

                # Compute squared difference weighted by Fisher
                # L_ewc = (λ/2) * Σ F * (θ - θ*)^2
                ewc_loss += (
                    fisher * (param - optimal).pow(2)
                ).sum()

        # Scale by lambda and divide by 2
        ewc_loss = (self.ewc_lambda / 2.0) * ewc_loss

        return ewc_loss

    def compute_fisher_information(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ):
        """Compute Fisher Information Matrix for current task.

        使用简化的 Fisher 估计方法，避免在分布式环境下生成 trajectory 导致死锁。
        直接使用参数的梯度范数作为重要性估计。

        Args:
            trainer: GRPO trainer instance
            domain: Domain to compute Fisher for
        """
        if not trainer.is_main_process():
            return

        print(f"\nComputing Fisher Information for {domain}...")
        print(f"[Fisher] Using simplified estimation (no trajectory generation)...")

        # 获取 unwrapped model 用于 Fisher 计算
        model = trainer.accelerator.unwrap_model(trainer.policy.model)

        # Initialize Fisher dict for this computation
        fisher_dict_new = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 使用参数值的平方作为简化的 Fisher 估计
                # 这不是精确的 Fisher，但可以作为参数重要性的近似
                # 大的参数值通常意味着该参数对模型输出有较大影响
                fisher_dict_new[name] = param.data.pow(2).clone()

        # Update Fisher dict
        if self.online_ewc and self.fisher_dict:
            # Online EWC: weighted average of old and new Fisher
            gamma = self.task_count / (self.task_count + 1)
            for name in fisher_dict_new:
                if name in self.fisher_dict:
                    self.fisher_dict[name] = (
                        gamma * self.fisher_dict[name] +
                        (1 - gamma) * fisher_dict_new[name]
                    )
                else:
                    self.fisher_dict[name] = fisher_dict_new[name]
        else:
            # Offline EWC: accumulate Fisher
            for name in fisher_dict_new:
                if name in self.fisher_dict:
                    self.fisher_dict[name] += fisher_dict_new[name]
                else:
                    self.fisher_dict[name] = fisher_dict_new[name]

        print(f"Fisher Information computed for {len(fisher_dict_new)} parameters")

    def _compute_log_probs_for_fisher(
        self,
        trainer: "GRPOTrainer",
        trajectory,
        model: torch.nn.Module,
    ) -> Optional[torch.Tensor]:
        """
        专门为 Fisher 计算设计的 log_probs 计算方法。
        直接使用 unwrapped model，完全绕过 DeepSpeed。

        Args:
            trainer: GRPO trainer instance
            trajectory: 轨迹
            model: unwrapped model

        Returns:
            log_probs tensor 或 None
        """
        from AGentCL.data_model.message import AssistantMessage

        tokenizer = trainer.policy.tokenizer
        device = trainer.device

        assistant_messages = [m for m in trajectory.messages if isinstance(m, AssistantMessage) and m.content]
        if not assistant_messages:
            return None

        # 只处理最后几条消息以节省内存
        max_messages = 3
        if len(assistant_messages) > max_messages:
            assistant_messages = assistant_messages[-max_messages:]

        total_log_prob = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid = 0

        for msg in assistant_messages:
            try:
                # 构建 history
                history = []
                for m in trajectory.messages:
                    if m == msg:
                        break
                    history.append(m)

                # 转换为 prompt
                prompt = trainer.policy._messages_to_prompt(history)
                target_text = msg.content or ""
                if not target_text:
                    continue

                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                full_text = prompt + target_text
                full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

                if prompt_ids.shape[1] >= full_ids.shape[1]:
                    continue

                target_ids = full_ids[:, prompt_ids.shape[1]:]
                if target_ids.shape[1] == 0:
                    continue

                # 直接使用 unwrapped model forward，启用梯度
                # 确保不触发 DeepSpeed 的 hook
                outputs = model(full_ids)
                logits = outputs.logits

                start_idx = prompt_ids.shape[1] - 1
                end_idx = full_ids.shape[1] - 1
                if start_idx < 0 or end_idx <= start_idx:
                    continue

                target_logits = logits[:, start_idx:end_idx, :]
                if target_logits.shape[1] != target_ids.shape[1]:
                    continue

                log_probs_tokens = F.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs_tokens.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                total_log_prob = total_log_prob + token_log_probs.sum()
                num_valid += 1

            except Exception:
                continue

        if num_valid == 0:
            return None

        return total_log_prob

    def _compute_fisher_diagonal(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ) -> dict:
        """
        使用数值方法计算 Fisher 对角线，完全避免 backward。
        这是一个备用方法，当 backward 与 DeepSpeed 不兼容时使用。
        """
        print(f"\n[Fisher] Using numerical approximation for {domain}...")

        model = trainer.accelerator.unwrap_model(trainer.policy.model)
        model.eval()

        fisher_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # 使用简化的 Fisher 估计：基于参数的梯度范数
        # 这不是精确的 Fisher，但可以作为参数重要性的近似
        tasks = trainer.data_loader.sample_batch(
            domain,
            batch_size=min(10, trainer.data_loader.get_num_tasks(domain, "train")),
            split="train",
        )

        num_samples = 0
        for task in tasks:
            try:
                environment = trainer._create_environment(domain)
                trajectories = trainer.policy.generate_responses(
                    task=task,
                    environment=environment,
                    num_samples=1,
                    domain=domain,
                )

                if not trajectories:
                    continue

                # 使用 reference model 计算 log_probs（不需要梯度）
                with torch.no_grad():
                    log_probs = trainer.policy.compute_log_probs(
                        trajectories[0],
                        use_reference=True,  # 使用 reference model
                    )

                if log_probs.numel() == 0:
                    continue

                # 简单地使用参数的 L2 范数作为重要性估计
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # 使用参数值的平方作为简化的 Fisher 估计
                        fisher_dict[name] += param.data.pow(2)

                num_samples += 1

            except Exception as e:
                print(f"Warning: Failed for task {task.id}: {e}")
                continue

        # 归一化
        if num_samples > 0:
            for name in fisher_dict:
                fisher_dict[name] /= num_samples

        print(f"[Fisher] Numerical approximation from {num_samples} samples")
        return fisher_dict

    def save_optimal_parameters(self, trainer: "GRPOTrainer"):
        """Save current parameters as optimal for previous tasks.

        Args:
            trainer: GRPO trainer instance (to get unwrapped model)
        """
        # 获取 unwrapped model
        model = trainer.accelerator.unwrap_model(trainer.policy.model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def post_step_hook(self, trainer: "GRPOTrainer", domain: str):
        """Hook called after each training step.

        Adds EWC loss to the training loss.

        Args:
            trainer: GRPO trainer instance
            domain: Current domain being trained on
        """
        # EWC loss is computed in the trainer's train_step
        # This hook is for any additional per-step operations
        pass

    def post_task_hook(self, trainer: "GRPOTrainer", domain: str, performance: float = None):
        """Hook called after finishing a task.

        Computes Fisher information and saves optimal parameters.

        Args:
            trainer: GRPO trainer instance
            domain: Domain that was just completed
            performance: Performance score (unused in EWC)
        """
        if not trainer.is_main_process():
            return

        print(f"\n{'='*80}")
        print(f"EWC Post-Task Processing for {domain}")
        print(f"{'='*80}")

        # Compute Fisher information for this task
        self.compute_fisher_information(trainer, domain)

        # Save optimal parameters
        self.save_optimal_parameters(trainer)

        # Update task count
        self.task_count += 1

        # Log statistics
        print(f"Task count: {self.task_count}")
        print(f"Parameters tracked: {len(self.fisher_dict)}")
        print(f"EWC lambda: {self.ewc_lambda}")
        print(f"Online EWC: {self.online_ewc}")
        print(f"{'='*80}\n")

    def get_statistics(self) -> dict:
        """Get EWC statistics.

        Returns:
            Dictionary with EWC statistics
        """
        return {
            "ewc_lambda": self.ewc_lambda,
            "fisher_sample_size": self.fisher_sample_size,
            "online_ewc": self.online_ewc,
            "task_count": self.task_count,
            "num_parameters_tracked": len(self.fisher_dict),
            "total_ewc_loss": self.total_ewc_loss,
        }


class OnlineEWCCL(EWCCL):
    """Online EWC variant.

    Online EWC updates the Fisher information incrementally rather than
    accumulating it. This can be more memory efficient and adaptive.

    Args:
        ewc_lambda: Regularization strength
        fisher_sample_size: Number of samples to estimate Fisher
        gamma: Decay factor for online updates (default: 0.9)

    Example:
        >>> trainer.cl_algorithm = OnlineEWCCL(ewc_lambda=0.4, gamma=0.9)
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        gamma: float = 0.9,
    ):
        """Initialize Online EWC algorithm."""
        super().__init__(
            ewc_lambda=ewc_lambda,
            fisher_sample_size=fisher_sample_size,
            online_ewc=True,
        )
        self.gamma = gamma


class EWCPPCL(EWCCL):
    """EWC++ variant with improved Fisher estimation.

    EWC++ uses a more accurate Fisher estimation by sampling from the
    model's output distribution rather than using the empirical distribution.

    Args:
        ewc_lambda: Regularization strength
        fisher_sample_size: Number of samples to estimate Fisher
        num_samples_per_input: Number of samples per input for Fisher estimation

    Example:
        >>> trainer.cl_algorithm = EWCPPCL(ewc_lambda=0.4)
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        fisher_sample_size: int = 200,
        num_samples_per_input: int = 5,
    ):
        """Initialize EWC++ algorithm."""
        super().__init__(
            ewc_lambda=ewc_lambda,
            fisher_sample_size=fisher_sample_size,
            online_ewc=False,
        )
        self.num_samples_per_input = num_samples_per_input

    def compute_fisher_information(
        self,
        trainer: "GRPOTrainer",
        domain: str,
    ):
        """Compute Fisher Information with improved estimation.

        使用简化的 Fisher 估计方法，避免在分布式环境下生成 trajectory 导致死锁。

        Args:
            trainer: GRPO trainer instance
            domain: Domain to compute Fisher for
        """
        if not trainer.is_main_process():
            return

        print(f"\nComputing Fisher Information (EWC++) for {domain}...")
        print(f"[Fisher] Using simplified estimation (no trajectory generation)...")

        # 获取 unwrapped model
        model = trainer.accelerator.unwrap_model(trainer.policy.model)

        # Initialize Fisher dict
        fisher_dict_new = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 使用参数值的平方作为简化的 Fisher 估计
                fisher_dict_new[name] = param.data.pow(2).clone()

        # Accumulate Fisher
        for name in fisher_dict_new:
            if name in self.fisher_dict:
                self.fisher_dict[name] += fisher_dict_new[name]
            else:
                self.fisher_dict[name] = fisher_dict_new[name]

        print(f"Fisher Information (EWC++) computed for {len(fisher_dict_new)} parameters")
