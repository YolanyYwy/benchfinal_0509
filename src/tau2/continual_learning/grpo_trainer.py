"""GRPO trainer for continual learning on agent tool-use tasks (Accelerate + wandb)."""

import os
import pickle
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator

from AGentCL.data_model.tasks import Task
from AGentCL.registry import registry

from .config import GRPOConfig, set_all_seeds
from .continual_learning.base import CLAlgorithm, SequentialCL
from .data_loader import TaskDataLoader
from .metrics_tracker import MetricsTracker
from .policy_model import PolicyModel
from .reward_oracle import RewardOracle, Trajectory
from .trajectory_buffer import TrajectoryBuffer
from .trajectory_logger import TrajectoryLogger


class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config

        # Set random seeds for reproducibility
        set_all_seeds(config.seed)

        # Explicitly set CUDA device before creating Accelerator
        # DO NOT modify CUDA_VISIBLE_DEVICES here - let accelerate handle it
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if local_rank == 0:
            print(f"[Seed] All random seeds set to {config.seed}")

        # Initialize Accelerator
        # When using DeepSpeed config file, don't specify mixed_precision here
        # as it's already defined in the DeepSpeed config
        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
        if use_deepspeed:
            self.accelerator = Accelerator()
        else:
            mp = "bf16" if config.model_dtype == "bfloat16" else ("fp16" if config.model_dtype == "float16" else "no")
            self.accelerator = Accelerator(mixed_precision=mp)
        self.device = self.accelerator.device

        # sync config with accelerator runtime
        self.config.local_rank = self.accelerator.local_process_index
        self.config.world_size = self.accelerator.num_processes

        self._data_loader = None  # lazy-init, only needed for tau2 train() not train_rlla/train_apibank
        self.policy = PolicyModel(config, self.device)

        # prepare trainables
        self.policy.model, self.policy.optimizer = self.accelerator.prepare(
            self.policy.model, self.policy.optimizer
        )
        if self.policy.scheduler is not None:
            self.policy.scheduler = self.accelerator.prepare(self.policy.scheduler)

        self.oracle = RewardOracle(evaluation_type="ALL")
        self.trajectory_buffer = TrajectoryBuffer(config)

        # Metrics tracker only on main process
        if self.is_main_process():
            self.metrics = MetricsTracker(config)
            # ✅ init wandb here
            self.metrics.init_wandb()
        else:
            self.metrics = None

        # Trajectory logger only on main process
        if self.is_main_process():
            trajectory_log_dir = Path(config.log_dir) / "trajectory_logs"
            self.trajectory_logger = TrajectoryLogger(
                log_dir=str(trajectory_log_dir),
                enabled=config.save_trajectory_logs,
            )
        else:
            self.trajectory_logger = None

        self.cl_algorithm = self._create_cl_algorithm()

        # 从 checkpoint 恢复模型（如果指定）
        resume_checkpoint = getattr(config, 'resume_from_checkpoint', None)
        if resume_checkpoint:
            self._load_step_checkpoint(resume_checkpoint)

        if self.is_main_process():
            print(f"[Accelerate] world_size={self.config.world_size} local_rank={self.config.local_rank}")

    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def data_loader(self):
        if self._data_loader is None:
            self._data_loader = TaskDataLoader(self.config)
        return self._data_loader

    def _create_cl_algorithm(self) -> CLAlgorithm:
        if self.config.cl_algorithm == "sequential":
            return SequentialCL()
        elif self.config.cl_algorithm == "replay":
            from .continual_learning.replay import ReplayCL
            return ReplayCL(
                replay_ratio=self.config.replay_ratio,
                replay_strategy="random",
                min_buffer_size=10,
                replay_all_domains=True,
            )
        elif self.config.cl_algorithm == "adaptive_replay":
            from .continual_learning.replay import AdaptiveReplayCL
            return AdaptiveReplayCL(
                initial_replay_ratio=self.config.replay_ratio,
                max_replay_ratio=0.5,
                min_replay_ratio=0.1,
                adaptation_rate=0.1,
                forgetting_threshold=0.1,
                replay_strategy="random",
            )
        elif self.config.cl_algorithm == "ewc":
            from .continual_learning.ewc import EWCCL
            return EWCCL(ewc_lambda=0.4, fisher_sample_size=200, online_ewc=False)
        elif self.config.cl_algorithm == "online_ewc":
            from .continual_learning.ewc import OnlineEWCCL
            return OnlineEWCCL(ewc_lambda=0.4, fisher_sample_size=200, gamma=0.9)
        elif self.config.cl_algorithm == "ewc_pp":
            from .continual_learning.ewc import EWCPPCL
            return EWCPPCL(ewc_lambda=0.4, fisher_sample_size=200, num_samples_per_input=5)
        elif self.config.cl_algorithm == "progressive":
            from .continual_learning.progressive import ProgressiveNetsCL
            return ProgressiveNetsCL(adapter_size=256, use_lateral_connections=True, freeze_previous_columns=True)
        elif self.config.cl_algorithm == "dynamic_expansion":
            from .continual_learning.progressive import DynamicExpansionCL
            return DynamicExpansionCL(base_adapter_size=256, min_adapter_size=128, max_adapter_size=512)
        elif self.config.cl_algorithm == "fusion":
            from .continual_learning.fusion import ModelFusionCL
            return ModelFusionCL(fusion_strategy="weighted_average", merge_frequency="per_task", keep_task_models=True)
        elif self.config.cl_algorithm == "adaptive_fusion":
            from .continual_learning.fusion import AdaptiveFusionCL
            return AdaptiveFusionCL(fusion_strategy="weighted_average", num_weight_search_steps=10)
        else:
            raise ValueError(f"Unknown CL algorithm: {self.config.cl_algorithm}")

    def train(self):
        """
        完整的持续学习训练流程，包含：
        - Zero-shot 基准测试（随机初始化模型在所有 domain 上的性能）
        - 单任务 baseline 评估（加载预训练的单任务模型，用于 FWT 计算）
        - 持续学习训练
        - FWT (Forward Transfer): 顺序训练完任务 1~j 后在任务 j 上的性能 - 只在任务 j 上单独训练后的性能
        - BWT (Backward Transfer): 训练完任务 i 后在任务 j 上的性能 - 刚训练完任务 j 时的性能 (i > j)
          形成一个下三角矩阵，每训完一个 domain 都测试之前所有 domain
        - AP (Average Performance): 训练完任务 t 后，在所有已见任务上的平均性能
        - AIP (Average Incremental Performance): 所有 AP_t 的平均值
        - Generalization: 最后一个 domain 不参与训练，比较训练后 vs 未训练模型
        """
        # 最后一个 domain 用于泛化测试，不参与训练
        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order
        generalization_domain = self.config.task_order[-1] if len(self.config.task_order) > 1 else None
        num_train_tasks = len(train_domains)

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Setup] Training domains: {train_domains}")
            print(f"[Setup] Generalization test domain: {generalization_domain}")
            print(f"{'='*60}\n")

        # ========== Phase 1: Zero-shot 基准测试 ==========
        zero_shot_performance = {}  # zero_shot_performance[domain] = score

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 1] Zero-shot evaluation (before any training)")
            print(f"{'='*60}")

        # 所有 rank 都参与评估
        for domain in train_domains:
            if self.is_main_process():
                print(f"\n[Zero-shot] Testing {domain}...")
            results = self.evaluate_task_pass_at_k_distributed(domain)
            if self.is_main_process():
                zero_shot_performance[domain] = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"zero_shot_{domain}", results)
                print(f"[Zero-shot] {domain}: pass@{self.config.pass_at_k}_mean = {zero_shot_performance[domain]:.3f}")

        if generalization_domain:
            if self.is_main_process():
                print(f"\n[Zero-shot] Testing {generalization_domain} (generalization domain)...")
            results = self.evaluate_task_pass_at_k_distributed(generalization_domain)
            if self.is_main_process():
                zero_shot_performance[generalization_domain] = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"zero_shot_{generalization_domain}", results)
                print(f"[Zero-shot] {generalization_domain}: pass@{self.config.pass_at_k}_mean = {zero_shot_performance[generalization_domain]:.3f}")

        if self.is_main_process():
            print(f"\n[Zero-shot Summary]")
            for domain, score in zero_shot_performance.items():
                print(f"  {domain}: {score:.3f}")

        self.accelerator.wait_for_everyone()

        # ========== Phase 1.5: 单任务 Baseline 评估（用于 FWT 计算）==========
        # FWT_j = 顺序训练后性能 - 单任务训练后性能
        single_task_performance = {}  # single_task_performance[domain] = score

        if self.config.single_domain_checkpoint_dir:
            if self.is_main_process():
                print(f"\n{'='*60}")
                print(f"[Phase 1.5] Single-task baseline evaluation (for FWT)")
                print(f"[Phase 1.5] Loading from: {self.config.single_domain_checkpoint_dir}")
                print(f"{'='*60}")

            single_ckpt_dir = Path(self.config.single_domain_checkpoint_dir)
            for domain in train_domains:
                domain_ckpt = single_ckpt_dir / domain / "model"
                if domain_ckpt.exists():
                    if self.is_main_process():
                        print(f"\n[Single-task] Loading {domain} checkpoint...")
                    # 保存当前模型状态
                    current_state = {k: v.clone() for k, v in self.policy.model.state_dict().items()}
                    # 加载单任务模型（所有 rank 都需要加载）
                    self.policy.load_checkpoint(str(domain_ckpt))
                    self.accelerator.wait_for_everyone()
                    # 评估（所有 rank 参与）
                    results = self.evaluate_task_pass_at_k_distributed(domain)
                    if self.is_main_process():
                        single_task_performance[domain] = results.get('pass_at_k_mean', 0)
                        self.metrics.log_generalization(f"single_task_{domain}", results)
                        print(f"[Single-task] {domain}: pass@{self.config.pass_at_k}_mean = {single_task_performance[domain]:.3f}")
                    # 恢复当前模型状态（所有 rank 都需要恢复）
                    self.policy.model.load_state_dict(current_state)
                    self.accelerator.wait_for_everyone()
                else:
                    if self.is_main_process():
                        print(f"[Single-task] Warning: No checkpoint for {domain}, using zero-shot as baseline")
                        single_task_performance[domain] = zero_shot_performance.get(domain, 0)

            if self.is_main_process():
                print(f"\n[Single-task Summary]")
                for domain, score in single_task_performance.items():
                    print(f"  {domain}: {score:.3f}")
        else:
            if self.is_main_process():
                print(f"\n[Phase 1.5] No single_domain_checkpoint_dir specified, using zero-shot as FWT baseline")
                single_task_performance = zero_shot_performance.copy()

        self.accelerator.wait_for_everyone()

        # ========== Phase 2: 持续学习训练 ==========
        # performance_matrix[i][j] = 训练完任务 i 后在任务 j 上的性能
        # 用于计算 FWT, BWT, AP
        performance_matrix = {}  # performance_matrix[task_idx] = {domain: score}
        ap_list = []  # 记录每个阶段的 AP
        bwt_matrix = {}  # bwt_matrix[i][j] = BWT_{i,j} (i > j)
        eval_results_matrix = {}  # 保存完整的 pass@k 评估结果

        # 支持从指定 task 开始训练（断点续训）
        start_task_idx = getattr(self.config, 'resume_from_task', 0)
        skip_training_for_resume_task = getattr(self.config, 'skip_training_for_resume_task', False)
        resume_from_eval_domain = getattr(self.config, 'resume_from_eval_domain', None)

        if start_task_idx > 0 and self.is_main_process():
            print(f"\n[Resume] Starting from task {start_task_idx} ({train_domains[start_task_idx]})")
            if skip_training_for_resume_task:
                print(f"[Resume] Skipping training for task {start_task_idx}, starting from evaluation")
            if resume_from_eval_domain:
                print(f"[Resume] Will resume evaluation from domain: {resume_from_eval_domain}")

        for task_idx, domain in enumerate(train_domains):
            # 跳过已完成的 task
            if task_idx < start_task_idx:
                if self.is_main_process():
                    print(f"\n[Skip] Task {task_idx} ({domain}) - already completed")
                continue

            if self.is_main_process():
                print(f"\n{'='*60}")
                print(f"[Phase 2] Task {task_idx}: Training on {domain}")
                print(f"{'='*60}")
                self.metrics.start_task_progress(task_idx, domain)

            # 如果是 resume 的 task 且设置了跳过训练，则跳过训练阶段
            if task_idx == start_task_idx and skip_training_for_resume_task:
                if self.is_main_process():
                    print(f"[Skip] Skipping training for task {task_idx} ({domain}) - resuming from evaluation")
            else:
                # 训练当前任务
                self.train_task(domain, task_idx)

            if self.is_main_process():
                self.metrics.close_task_progress()

            # ========== 评估阶段（所有 rank 必须参与）==========
            # 在 DeepSpeed ZeRO 模式下，generate_responses 需要所有 rank 参与
            # 因此评估不能只在 main process 执行
            self.accelerator.wait_for_everyone()

            if self.is_main_process():
                print(f"\n[Evaluation after Task {task_idx}] Testing all trained domains (0~{task_idx})...")
                performance_matrix[task_idx] = {}

            # 确定从哪个 eval_idx 开始评估
            start_eval_idx = 0
            if task_idx == start_task_idx and resume_from_eval_domain:
                # 找到 resume_from_eval_domain 在 train_domains 中的索引
                for idx, d in enumerate(train_domains[:task_idx + 1]):
                    if d == resume_from_eval_domain:
                        start_eval_idx = idx
                        if self.is_main_process():
                            print(f"[Resume] Resuming evaluation from domain {resume_from_eval_domain} (index {idx})")
                        break
                # 清除 resume_from_eval_domain，只在第一个 task 使用
                resume_from_eval_domain = None

            # 评估所有已训练的 domain
            eval_results_full = {}  # 保存当前 task 的完整评估结果
            for eval_idx in range(task_idx + 1):
                eval_domain = train_domains[eval_idx]

                # 跳过已完成的评估
                if eval_idx < start_eval_idx:
                    if self.is_main_process():
                        print(f"\n[Skip Eval] {eval_domain} - already evaluated")
                    continue

                if self.is_main_process():
                    print(f"\n[Eval] Testing {eval_domain}...")

                # 所有 rank 都参与评估（内部会调用 generate_responses）
                results = self.evaluate_task_pass_at_k_distributed(eval_domain)

                if self.is_main_process():
                    eval_results_full[eval_domain] = results  # 保存完整结果
                    score = results.get('pass_at_k_mean', 0)
                    performance_matrix[task_idx][eval_domain] = score
                    print(f"[Eval] {eval_domain}: pass@{self.config.pass_at_k}_mean = {score:.3f}")

            # 同步确保所有评估完成
            self.accelerator.wait_for_everyone()

            if self.is_main_process():
                # 保存完整评估结果到 matrix
                eval_results_matrix[task_idx] = eval_results_full

                # 计算当前阶段的 AP (Average Performance)
                ap_t = sum(performance_matrix[task_idx].values()) / (task_idx + 1)
                ap_list.append(ap_t)

                # 计算当前任务的 FWT（使用单任务 baseline）
                # FWT_j = 顺序训练后性能 - 单任务训练后性能
                baseline = single_task_performance.get(domain, zero_shot_performance.get(domain, 0))
                fwt_j = performance_matrix[task_idx][domain] - baseline

                # 计算 BWT 矩阵：对于之前的每个任务 j < task_idx
                # BWT_{task_idx, j} = R_{task_idx, j} - R_{j, j}
                bwt_matrix[task_idx] = {}
                for prev_idx in range(task_idx):
                    prev_domain = train_domains[prev_idx]
                    current_perf = performance_matrix[task_idx][prev_domain]
                    after_train_perf = performance_matrix[prev_idx][prev_domain]
                    bwt_ij = current_perf - after_train_perf
                    bwt_matrix[task_idx][prev_domain] = bwt_ij

                print(f"\n[Metrics after Task {task_idx}]")
                print(f"  AP_{task_idx} (Average Performance): {ap_t:.3f}")
                print(f"  FWT_{task_idx} ({domain}): {fwt_j:+.3f}")
                print(f"    - After CL training: {performance_matrix[task_idx][domain]:.3f}")
                print(f"    - Single-task baseline: {baseline:.3f}")

                # 打印 BWT 矩阵当前行
                if task_idx > 0:
                    print(f"  BWT after Task {task_idx}:")
                    for prev_idx in range(task_idx):
                        prev_domain = train_domains[prev_idx]
                        bwt_val = bwt_matrix[task_idx][prev_domain]
                        print(f"    BWT_{task_idx},{prev_idx} ({prev_domain}): {bwt_val:+.3f}")

                # 记录到 metrics
                transfer_metrics = {
                    "AP": ap_t,
                    f"FWT_{domain}": fwt_j,
                    "current_task": domain,
                    "num_tasks_seen": task_idx + 1,
                }
                # 添加 BWT 值
                for prev_domain, bwt_val in bwt_matrix.get(task_idx, {}).items():
                    transfer_metrics[f"BWT_{task_idx}_{prev_domain}"] = bwt_val
                self.metrics.log_transfer(task_idx, transfer_metrics)

            # 所有 rank 必须在 checkpoint 保存前同步
            if self.is_main_process():
                print(f"\n[Sync] Waiting for all ranks before checkpoint...")
            self.accelerator.wait_for_everyone()
            if self.is_main_process():
                print(f"[Sync] All ranks synchronized. Saving checkpoint...")

            # 所有 rank 都需要调用 save_checkpoint 来保存各自的 optimizer state
            self.save_checkpoint(task_idx, domain=domain)
            if self.is_main_process():
                print(f"[Checkpoint] Task {task_idx} ({domain}) checkpoint saved.")

            # 所有 rank 必须在 post_task_hook 前同步
            if self.is_main_process():
                print(f"[Sync] Waiting for all ranks before post_task_hook...")
            self.accelerator.wait_for_everyone()
            if self.is_main_process():
                print(f"[Sync] All ranks synchronized. Running post_task_hook...")

            # 传递已经计算好的 performance，避免重复评估
            current_performance = None
            if self.is_main_process() and task_idx in performance_matrix:
                current_performance = performance_matrix[task_idx].get(domain, None)

            self.cl_algorithm.post_task_hook(self, domain, performance=current_performance)

            # 每个 domain 完成后立即保存结果（增量保存，防止中断丢失）
            if self.is_main_process():
                output_path = Path(self.config.log_dir) / "phase2_training.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                training_results = {
                    "performance_matrix": {str(k): v for k, v in performance_matrix.items()},
                    "eval_results_matrix": {str(k): v for k, v in eval_results_matrix.items()},
                    "ap_list": ap_list,
                    "bwt_matrix": {str(k): v for k, v in bwt_matrix.items()},
                    "zero_shot_performance": zero_shot_performance,
                    "single_task_performance": single_task_performance,
                    "last_completed_task": task_idx,
                    "last_completed_domain": domain,
                }
                with open(output_path, "w") as f:
                    json.dump(training_results, f, indent=2)

                print(f"[Task {task_idx}] Completed. Results saved to {output_path}\n")

        # ========== Phase 3: 最终评估与指标计算 ==========
        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 3] Final evaluation and metrics computation")
            print(f"{'='*60}")

        # 3.1 最终性能评估（所有训练 domain）- 所有 rank 参与
        final_performance = {}
        if self.is_main_process():
            print(f"\n[Final Evaluation] Testing all trained domains...")

        for domain in train_domains:
            if self.is_main_process():
                print(f"\n[Final] Testing {domain}...")
            results = self.evaluate_task_pass_at_k_distributed(domain)
            if self.is_main_process():
                final_performance[domain] = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"final_{domain}", results)
                print(f"[Final] {domain}: pass@{self.config.pass_at_k}_mean = {final_performance[domain]:.3f}")

        # 3.2 泛化性测试 - 所有 rank 参与
        final_generalization = 0
        if generalization_domain:
            if self.is_main_process():
                print(f"\n[Generalization] Testing {generalization_domain} (unseen domain)...")
            results = self.evaluate_task_pass_at_k_distributed(generalization_domain)
            if self.is_main_process():
                final_generalization = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"final_{generalization_domain}", results)
                print(f"[Generalization] {generalization_domain}: pass@{self.config.pass_at_k}_mean = {final_generalization:.3f}")

        self.accelerator.wait_for_everyone()

        if self.is_main_process():
            # ========== 计算所有 CL 指标 ==========
            print(f"\n{'='*60}")
            print(f"[Final CL Metrics Summary]")
            print(f"{'='*60}")

            # FWT (Forward Transfer) for each task
            # FWT_j = 顺序训练后性能 - 单任务训练后性能
            print(f"\n[FWT] Forward Transfer (顺序训练后性能 - 单任务训练后性能):")
            fwt_values = []
            for task_idx, domain in enumerate(train_domains):
                baseline = single_task_performance.get(domain, zero_shot_performance.get(domain, 0))
                fwt_j = performance_matrix[task_idx][domain] - baseline
                fwt_values.append(fwt_j)
                print(f"  FWT_{task_idx} ({domain}): {fwt_j:+.3f} = {performance_matrix[task_idx][domain]:.3f} - {baseline:.3f}")
            avg_fwt = sum(fwt_values) / len(fwt_values) if fwt_values else 0
            print(f"  Average FWT: {avg_fwt:+.3f}")

            # BWT Matrix (Backward Transfer)
            # 打印完整的 BWT 矩阵
            print(f"\n[BWT Matrix] Backward Transfer Matrix:")
            print(f"  BWT_{{i,j}} = R_{{i,j}} - R_{{j,j}} (训练完任务i后在任务j上的性能 - 刚训练完任务j时的性能)")

            # 收集所有 BWT 值用于计算平均
            all_bwt_values = []

            # 打印矩阵头
            header = "After\\On  " + "  ".join([f"{d[:6]:>8}" for d in train_domains])
            print(f"  {header}")
            print(f"  {'-' * len(header)}")

            for i in range(len(train_domains)):
                row = f"  Task {i:<3} "
                for j in range(len(train_domains)):
                    if j < i:
                        # BWT_{i,j} = R_{i,j} - R_{j,j}
                        bwt_val = bwt_matrix.get(i, {}).get(train_domains[j], 0)
                        all_bwt_values.append(bwt_val)
                        row += f"{bwt_val:+8.3f}  "
                    elif j == i:
                        row += f"{'---':>8}  "
                    else:
                        row += f"{'':>8}  "
                print(row)

            avg_bwt = sum(all_bwt_values) / len(all_bwt_values) if all_bwt_values else 0
            print(f"\n  Average BWT (from matrix): {avg_bwt:+.3f}")
            if avg_bwt < 0:
                print(f"  (Negative BWT indicates forgetting)")

            # 最终 BWT（训练完所有任务后）
            print(f"\n[Final BWT] After all training:")
            final_bwt_values = []
            for task_idx, domain in enumerate(train_domains[:-1]):  # 除了最后一个任务
                final_score = final_performance[domain]
                after_training_score = performance_matrix[task_idx][domain]
                bwt_j = final_score - after_training_score
                final_bwt_values.append(bwt_j)
                print(f"  BWT_final_{task_idx} ({domain}): {bwt_j:+.3f} = {final_score:.3f} - {after_training_score:.3f}")
            avg_final_bwt = sum(final_bwt_values) / len(final_bwt_values) if final_bwt_values else 0
            print(f"  Average Final BWT: {avg_final_bwt:+.3f}")

            # AP (Average Performance) at each stage
            print(f"\n[AP] Average Performance at each stage:")
            for task_idx, ap in enumerate(ap_list):
                print(f"  AP_{task_idx}: {ap:.3f}")

            # AIP (Average Incremental Performance)
            aip = sum(ap_list) / len(ap_list) if ap_list else 0
            print(f"\n[AIP] Average Incremental Performance: {aip:.3f}")

            # Final Average Performance
            final_ap = sum(final_performance.values()) / len(final_performance) if final_performance else 0
            print(f"\n[Final AP] Final Average Performance: {final_ap:.3f}")

            # Generalization
            if generalization_domain:
                gen_improvement = final_generalization - zero_shot_performance.get(generalization_domain, 0)
                print(f"\n[Generalization] Domain: {generalization_domain}")
                print(f"  Zero-shot: {zero_shot_performance.get(generalization_domain, 0):.3f}")
                print(f"  After training: {final_generalization:.3f}")
                print(f"  Improvement: {gen_improvement:+.3f}")

            # 汇总表格
            print(f"\n{'='*80}")
            print(f"[Summary Table]")
            print(f"{'='*80}")
            print(f"{'Domain':<12} {'Zero-shot':<10} {'Single':<10} {'After CL':<10} {'Final':<10} {'FWT':<10} {'Final BWT':<10}")
            print(f"{'-'*80}")
            for task_idx, domain in enumerate(train_domains):
                zs = zero_shot_performance.get(domain, 0)
                st = single_task_performance.get(domain, zs)
                at = performance_matrix[task_idx][domain]
                fn = final_performance[domain]
                fwt = at - st  # FWT = 顺序训练后 - 单任务训练后
                bwt = fn - at  # Final BWT
                print(f"{domain:<12} {zs:<10.3f} {st:<10.3f} {at:<10.3f} {fn:<10.3f} {fwt:<+10.3f} {bwt:<+10.3f}")
            if generalization_domain:
                zs = zero_shot_performance.get(generalization_domain, 0)
                fn = final_generalization
                print(f"{generalization_domain:<12} {zs:<10.3f} {'N/A':<10} {'N/A':<10} {fn:<10.3f} {'N/A':<10} {'N/A':<10}")
            print(f"{'-'*80}")
            print(f"{'Average':<12} {'':<10} {'':<10} {'':<10} {final_ap:<10.3f} {avg_fwt:<+10.3f} {avg_final_bwt:<+10.3f}")
            print(f"{'AIP':<12} {aip:<10.3f}")
            print(f"{'Avg BWT(M)':<12} {avg_bwt:<+10.3f}  (from BWT matrix)")
            print(f"{'='*80}")

            # 保存最终指标
            final_metrics = {
                "avg_fwt": avg_fwt,
                "avg_bwt_matrix": avg_bwt,  # BWT 矩阵的平均值
                "avg_bwt_final": avg_final_bwt,  # 最终 BWT
                "aip": aip,
                "final_ap": final_ap,
                "bwt_matrix": bwt_matrix,  # 保存完整的 BWT 矩阵
                "performance_matrix": performance_matrix,  # 保存完整的性能矩阵
                "single_task_performance": single_task_performance,  # 单任务 baseline
            }
            if generalization_domain:
                final_metrics["generalization"] = gen_improvement
            self.metrics.log_transfer(-1, final_metrics)  # -1 表示最终指标

        if self.is_main_process():
            self.metrics.save()
            self.metrics.close()

    def train_task(self, domain: str, task_idx: int):
        train_tasks = self.data_loader.get_train_tasks(domain)

        # 检查是否需要从 checkpoint 恢复
        start_step = 0
        resume_checkpoint = getattr(self.config, 'resume_from_checkpoint', None)
        if resume_checkpoint:
            # 从 checkpoint 路径解析 step 信息
            import re
            match = re.search(r'step_(\d+)', resume_checkpoint)
            if match:
                start_step = int(match.group(1)) + 1  # 从下一个 step 开始
                if self.is_main_process():
                    print(f"[Resume] Resuming from step {start_step}")
                # 只在第一次使用后清除，避免重复加载
                self.config.resume_from_checkpoint = None

        # 早停配置（基于训练 reward 饱和）
        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)
        early_stopping_threshold = getattr(self.config, 'early_stopping_threshold', 0.8)
        recent_rewards = []
        window_size = 5
        patience_counter = 0
        best_avg_reward = -float('inf')

        # Epoch iterator: shuffle all tasks, iterate in order, repeat when exhausted.
        # This ensures every task is seen once per epoch before any task is repeated.
        epoch_tasks = []
        epoch_num = 0

        def next_task_from_epoch():
            nonlocal epoch_tasks, epoch_num
            if not epoch_tasks:
                epoch_tasks = train_tasks.copy()
                random.shuffle(epoch_tasks)
                epoch_num += 1
                if self.is_main_process():
                    print(f"  [Epoch {epoch_num}] Starting new epoch over {len(epoch_tasks)} tasks")
            return epoch_tasks.pop(0)

        # Fast-forward the epoch iterator to match start_step so resume is consistent
        for _ in range(start_step % max(len(train_tasks), 1)):
            next_task_from_epoch()

        early_stop_flag = False

        for step in range(start_step, self.config.num_steps_per_task):
            # ToolRL style: sample batch_size_per_gpu tasks, generate G responses each
            batch_size = getattr(self.config, 'batch_size_per_gpu', 4)
            batch_tasks = [next_task_from_epoch() for _ in range(batch_size)]
            batch_tasks = self.cl_algorithm.augment_batch(batch_tasks, domain)
            metrics = self.train_step_toolrl(batch_tasks, domain, step)

            if self.is_main_process() and metrics:
                self.metrics.log_step(task_idx, step, metrics)

                # 早停检查：基于训练 reward
                if early_stopping_patience > 0 and metrics.get('reward_mean') is not None:
                    recent_rewards.append(metrics['reward_mean'])
                    if len(recent_rewards) > window_size:
                        recent_rewards.pop(0)

                    if len(recent_rewards) >= window_size:
                        avg_reward = sum(recent_rewards) / len(recent_rewards)

                        if avg_reward >= early_stopping_threshold:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                print(f"\n[Early Stopping] Reward saturated at {avg_reward:.3f} >= {early_stopping_threshold}")
                                print(f"[Early Stopping] Stopping at step {step}")
                                early_stop_flag = True
                        else:
                            patience_counter = 0

                        if avg_reward > best_avg_reward + 0.01:
                            best_avg_reward = avg_reward

            # 广播早停决策给所有 rank，确保一起退出避免死锁
            if self.config.world_size > 1:
                import torch.distributed as _dist
                flag_tensor = torch.tensor([1 if early_stop_flag else 0], dtype=torch.long, device=self.device)
                _dist.broadcast(flag_tensor, src=0)
                early_stop_flag = flag_tensor.item() == 1

            if early_stop_flag:
                self._save_step_checkpoint(task_idx, domain, step)
                return

            # 定期保存 checkpoint，用于断点续训
            checkpoint_interval = getattr(self.config, 'checkpoint_interval', 5)
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                self._save_step_checkpoint(task_idx, domain, step)

            # 评估阶段：只在多 domain 训练时进行中间评估，单 domain 跳过
            # 通过 skip_intermediate_eval 配置控制
            skip_eval = getattr(self.config, 'skip_intermediate_eval', False)
            if not skip_eval and step % self.config.eval_interval == 0 and step > 0:
                if self.is_main_process():
                    print(f"\n{'='*50}")
                    print(f"[Step {step}] Entering evaluation phase...")
                    print(f"{'='*50}")
                    eval_metrics = self.evaluate_task(domain)
                    self.metrics.log_eval(task_idx, step, eval_metrics)
                    print(f"{'='*50}")
                    print(f"[Step {step}] Evaluation completed. Metrics: reward_mean={eval_metrics.get('reward_mean', 0):.3f}, pass_rate={eval_metrics.get('pass_rate', 0):.1%}")
                    print(f"{'='*50}\n")
                # 关键：所有 rank 在评估后同步，避免死锁
                if self.is_main_process():
                    print(f"[Sync] Waiting for all ranks to synchronize...")
                self.accelerator.wait_for_everyone()
                if self.is_main_process():
                    print(f"[Sync] All ranks synchronized. Continuing to step {step + 1}...\n")

        # optimizer steps are now handled inside train_step_toolrl's mini-batch loop

    # ==================================================================
    # ToolRL-style single-domain training step
    # ==================================================================

    def _compute_grpo_advantage(
        self,
        rewards: List[float],
        uids: List[str],
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Exact port of ToolRL verl core_algos.compute_grpo_outcome_advantage.

        Group by uid, normalize within group:
          advantage_i = (score_i - group_mean) / (group_std + epsilon)

        Edge cases (matching ToolRL exactly):
          - group size == 1: mean=score, std=0 → advantage = 0
          - group size  > 1, std == 0: all same reward → advantage = 0
        """
        from collections import defaultdict
        uid2scores: dict = defaultdict(list)
        uid2idx:    dict = defaultdict(list)

        for i, (r, uid) in enumerate(zip(rewards, uids)):
            uid2scores[uid].append(r)
            uid2idx[uid].append(i)

        id2mean: dict = {}
        id2std:  dict = {}
        for uid, scores in uid2scores.items():
            t = torch.tensor(scores, dtype=torch.float32)
            id2mean[uid] = t.mean()
            id2std[uid]  = t.std() if len(scores) > 1 else torch.tensor(0.0)

        advantages = torch.zeros(len(rewards), dtype=torch.float32, device=self.device)
        for i, (r, uid) in enumerate(zip(rewards, uids)):
            advantages[i] = (r - id2mean[uid]) / (id2std[uid] + epsilon)

        return advantages

    def _apply_kl_penalty_to_rewards(
        self,
        rewards: List[float],
        trajectories: list,
        old_log_probs: list,
    ) -> List[float]:
        """
        Subtract KL penalty from scalar rewards before advantage computation.
        token_level_reward = reward - kl_coef * mean(low_var_kl)

        Mirrors ray_trainer.apply_kl_penalty() but on sequence-level scalars.
        """
        kl_coef = getattr(self.config, "kl_coef", 0.001)
        if kl_coef == 0.0:
            return rewards

        penalised = []
        for reward, traj, old_lp in zip(rewards, trajectories, old_log_probs):
            try:
                with torch.no_grad():
                    new_lp, ref_lp, n = self.policy.compute_log_probs_single_turn(traj)
                if n == 0:
                    penalised.append(reward)
                    continue
                min_len  = min(new_lp.shape[0], old_lp.shape[0])
                new_lp_a = new_lp[:min_len]
                old_lp_a = old_lp[:min_len].to(self.device)
                # low_var_kl between old and new (measures policy drift)
                kl_diff  = old_lp_a - new_lp_a
                kl_val   = torch.clamp(torch.exp(kl_diff) - kl_diff - 1.0,
                                       -10.0, 10.0).mean().item()
                penalised.append(reward - kl_coef * kl_val)
            except Exception:
                penalised.append(reward)

        return penalised

    def train_step_toolrl(
        self,
        tasks: list,
        domain: str,
        step: int = 0,
    ) -> Optional[dict]:
        """
        ToolRL-style GRPO step:

          Phase A  — generate B×G trajectories (multi-GPU parallel)
          Phase B  — compute rewards (ThreadPool, rank-0 only, then broadcast)
          Phase C  — apply KL penalty to rewards
          Phase D  — compute GRPO advantage (group by prompt uid)
          Phase E  — PPO clip loss with mini-batch updates
        """
        import pickle as _pickle
        import torch.distributed as _dist
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        rank       = self.config.local_rank
        world_size = self.config.world_size
        G          = self.config.num_samples_per_prompt
        B          = len(tasks)

        if self.is_main_process():
            print(f"\n[Step {step}] {B} tasks × G={G} → {B*G} trajectories", flush=True)

        # ---- Phase A: parallel rollout ----
        environments = [self._create_environment(domain) for _ in tasks]
        trajs, old_lps, uids = self.policy.generate_group(
            tasks=tasks,
            environments=environments,
            domain=domain,
            G=G,
            temperature=self.config.temperature,
        )
        self.accelerator.wait_for_everyone()

        if not trajs:
            if self.is_main_process():
                print(f"  ⏭ No trajectories generated, skipping step {step}")
            return None

        # ---- Phase B: rewards (rank 0 only, then broadcast) ----
        all_rewards:      List[float] = []
        all_reward_infos: list        = []

        if self.is_main_process():
            # tasks list is B long; trajs is B*G long (task i → trajs[i*G:(i+1)*G])
            task_per_traj = []
            for i, task in enumerate(tasks):
                task_per_traj.extend([task] * G)
            # pad if CL replay added extra tasks
            while len(task_per_traj) < len(trajs):
                task_per_traj.append(tasks[-1])

            TIMEOUT = 300

            def _reward(args):
                idx, task, traj = args
                return idx, self.oracle.compute_reward(
                    task=task, trajectory=traj, domain=domain, solo_mode=False
                )

            results = [None] * len(trajs)
            with ThreadPoolExecutor(max_workers=min(len(trajs), 16)) as ex:
                fmap = {ex.submit(_reward, (i, t, tr)): i
                        for i, (t, tr) in enumerate(zip(task_per_traj, trajs))}
                for fut in fmap:
                    i = fmap[fut]
                    try:
                        _, info = fut.result(timeout=TIMEOUT)
                        results[i] = info
                    except FuturesTimeout:
                        print(f"  Reward TIMEOUT traj {i}", flush=True)
                    except Exception as e:
                        print(f"  Reward ERROR traj {i}: {e}", flush=True)

            for info in results:
                if info is not None:
                    all_reward_infos.append(info)
                    all_rewards.append(info.reward)
                else:
                    all_reward_infos.append(None)
                    all_rewards.append(0.0)

            print(f"  rewards: mean={np.mean(all_rewards):.3f} "
                  f"min={np.min(all_rewards):.3f} max={np.max(all_rewards):.3f}",
                  flush=True)

        # broadcast rewards to all ranks
        if world_size > 1:
            payload = _pickle.dumps((all_rewards, all_reward_infos)) \
                      if self.is_main_process() else b""
            length_t = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
            _dist.broadcast(length_t, src=0)
            max_len = length_t.item()
            if self.is_main_process():
                padded_bytes = bytes(payload) + bytes(max_len - len(payload))
                payload_t = torch.tensor(list(padded_bytes), dtype=torch.uint8, device=self.device)
            else:
                payload_t = torch.zeros(max_len, dtype=torch.uint8, device=self.device)
            _dist.broadcast(payload_t, src=0)
            if not self.is_main_process():
                all_rewards, all_reward_infos = _pickle.loads(
                    payload_t.cpu().numpy().tobytes()
                )

        # ---- Phase C: GRPO advantage (no KL penalty — no frozen ref model) ----
        advantages = self._compute_grpo_advantage(all_rewards, uids)

        if self.is_main_process():
            print(f"  adv: mean={advantages.mean():.3f} std={advantages.std():.3f}",
                  flush=True)

        # log trajectories
        for traj, reward, info in zip(trajs, all_rewards, all_reward_infos):
            self.trajectory_buffer.add(
                domain=domain,
                task=tasks[0],
                trajectory=traj,
                reward=reward,
            )

        # ---- Phase D: PPO clip loss ----
        self.policy.model.train()
        ppo_metrics = self.policy.compute_grpo_loss_toolrl(
            trajectories=trajs,
            advantages=advantages,
            old_log_probs=old_lps,
            accelerator=self.accelerator,
        )

        self.cl_algorithm.post_step_hook(self, domain)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # aggregate metrics
        step_metrics = {
            "loss":          ppo_metrics.get("pg_loss", 0.0),
            "pg_loss":       ppo_metrics.get("pg_loss", 0.0),
            "pg_clipfrac":   ppo_metrics.get("pg_clipfrac", 0.0),
            "kl":            ppo_metrics.get("kl", 0.0),
            "entropy":       ppo_metrics.get("entropy", 0.0),
            "grad_norm":     ppo_metrics.get("grad_norm", 0.0),
            "reward_mean":   float(np.mean(all_rewards)),
            "reward_max":    float(np.max(all_rewards)),
            "reward_min":    float(np.min(all_rewards)),
            "reward_std":    float(np.std(all_rewards)),
            "num_trajs":     len(trajs),
        }

        if self.is_main_process():
            print(f"[Step {step}] pg_loss={step_metrics['pg_loss']:.4f} "
                  f"clip={step_metrics['pg_clipfrac']:.3f} "
                  f"kl={step_metrics['kl']:.4f} "
                  f"reward={step_metrics['reward_mean']:.3f}", flush=True)

        return step_metrics

    def _save_step_checkpoint(self, task_idx: int, domain: str, step: int):
        """保存训练中间 checkpoint，用于断点续训。所有 rank 都需要调用。"""
        ckpt_dir = Path(self.config.log_dir) / f"checkpoint_task_{task_idx}_step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.is_main_process():
            print(f"[Checkpoint] Saving step checkpoint to {ckpt_dir}...")

        # 所有 rank 都保存模型和 optimizer state
        unwrapped = self.accelerator.unwrap_model(self.policy.model)
        self.policy.save_checkpoint(str(ckpt_dir / "model"), model_override=unwrapped)

        # 关键：等待所有 rank 完成保存后再继续
        self.accelerator.wait_for_everyone()

        # 只有 main process 保存其他文件
        if self.is_main_process():
            import json
            state = {
                "task_idx": task_idx,
                "domain": domain,
                "step": step,
                "config": self.config.to_dict(),
            }
            with open(ckpt_dir / "training_state.json", "w") as f:
                json.dump(state, f, indent=2)

            # 保存 trajectory buffer
            self.trajectory_buffer.save(str(ckpt_dir / "trajectories.json"))

            # 保存 metrics
            self.metrics.save()

            print(f"[Checkpoint] Step {step} checkpoint saved.")

            # 清理旧的 step checkpoint，只保留最近 2 个（只在 main process 执行）
            self._cleanup_old_step_checkpoints(task_idx, keep_last=2)

        # 再次同步，确保 cleanup 完成后所有 rank 一起继续
        self.accelerator.wait_for_everyone()

    def _cleanup_old_step_checkpoints(self, task_idx: int, keep_last: int = 2):
        """清理旧的 step checkpoint，只保留最近几个"""
        import shutil
        log_dir = Path(self.config.log_dir)

        # 找到所有 step checkpoint
        pattern = f"checkpoint_task_{task_idx}_step_*"
        checkpoints = sorted(log_dir.glob(pattern), key=lambda p: int(p.name.split("_")[-1]))

        # 删除旧的，只保留最近的
        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[:-keep_last]:
                try:
                    shutil.rmtree(old_ckpt)
                    print(f"[Checkpoint] Removed old checkpoint: {old_ckpt.name}")
                except Exception as e:
                    print(f"[Checkpoint] Failed to remove {old_ckpt.name}: {e}")

    def _load_step_checkpoint(self, checkpoint_dir: str):
        """从 step checkpoint 恢复训练状态"""
        import json
        ckpt_path = Path(checkpoint_dir)

        if not ckpt_path.exists():
            if self.is_main_process():
                print(f"[Resume] Checkpoint not found: {checkpoint_dir}")
            return

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Resume] Loading checkpoint from {checkpoint_dir}")
            print(f"{'='*60}")

        # 加载模型
        model_path = ckpt_path / "model"
        if model_path.exists():
            self.policy.load_checkpoint(str(model_path))
            if self.is_main_process():
                print(f"[Resume] Model loaded from {model_path}")

        # 加载训练状态
        state_path = ckpt_path / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            if self.is_main_process():
                print(f"[Resume] Training state: task_idx={state.get('task_idx')}, "
                      f"domain={state.get('domain')}, step={state.get('step')}")

        # 加载 trajectory buffer
        traj_path = ckpt_path / "trajectories.json"
        if traj_path.exists():
            self.trajectory_buffer.load(str(traj_path))
            if self.is_main_process():
                print(f"[Resume] Trajectory buffer loaded")

        # 同步所有 rank
        self.accelerator.wait_for_everyone()

        if self.is_main_process():
            print(f"[Resume] Checkpoint loaded successfully")
            print(f"{'='*60}\n")

    def _sync_forward_count(self, local_count: int) -> int:
        """
        Synchronize the number of forward passes across all ranks.
        Returns the maximum count so all ranks can execute the same number of forwards.
        """
        if self.config.world_size == 1:
            return local_count

        count_tensor = torch.tensor([local_count], dtype=torch.long, device=self.device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
        return count_tensor.item()

    def _gather_rewards_across_ranks(self, local_rewards: List[float]) -> Tuple[List[float], bool, bool]:
        """
        Gather rewards from all ranks to determine if we have both positive and negative samples.

        Returns:
            all_rewards: List of all rewards across all ranks
            has_positive: Whether there's at least one positive sample (reward > 0)
            has_negative: Whether there's at least one negative sample (reward <= 0)
        """
        world_size = self.config.world_size
        rank = self.accelerator.process_index

        if world_size == 1:
            has_positive = any(r > 0 for r in local_rewards)
            has_negative = any(r <= 0 for r in local_rewards)
            return local_rewards, has_positive, has_negative

        # Multi-GPU: gather rewards across all ranks
        print(f"    [Rank {rank}] all_gather step 1: gathering counts (local_count={len(local_rewards)})...", flush=True)
        local_count = torch.tensor([len(local_rewards)], dtype=torch.long, device=self.device)
        all_counts = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        all_counts = [c.item() for c in all_counts]
        print(f"    [Rank {rank}] all_gather step 1 done: all_counts={all_counts}", flush=True)

        max_count = max(all_counts) if all_counts else 0
        if max_count == 0:
            return [], False, False

        padded_rewards = local_rewards + [-999.0] * (max_count - len(local_rewards))
        local_tensor = torch.tensor(padded_rewards, dtype=torch.float32, device=self.device)

        print(f"    [Rank {rank}] all_gather step 2: gathering rewards (max_count={max_count})...", flush=True)
        all_tensors = [torch.zeros(max_count, dtype=torch.float32, device=self.device) for _ in range(world_size)]
        dist.all_gather(all_tensors, local_tensor)
        print(f"    [Rank {rank}] all_gather step 2 done", flush=True)

        all_rewards = []
        for rank_idx, (tensor, count) in enumerate(zip(all_tensors, all_counts)):
            all_rewards.extend(tensor[:count].tolist())

        has_positive = any(r > 0 for r in all_rewards)
        has_negative = any(r <= 0 for r in all_rewards)

        return all_rewards, has_positive, has_negative

    def _sync_skip_decision(self, should_skip: bool) -> bool:
        """
        Synchronize skip decision across all ranks.
        All ranks must agree to skip or not skip to avoid deadlock.
        """
        if self.config.world_size == 1:
            return should_skip

        # Convert to tensor and all_reduce with MAX
        # If any rank says don't skip (0), result will reflect that
        skip_tensor = torch.tensor([1 if should_skip else 0], dtype=torch.long, device=self.device)
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MIN)  # All must agree to skip

        return skip_tensor.item() == 1

    def train_step(self, batch: list[Task], domain: str, step: int = 0) -> Optional[dict]:
        """
        Two-phase train step:
          Phase A: collect one trajectory per task in batch, compute rewards
          Phase B: normalize advantages across ALL tasks, single backward

        This fixes the core stagnation issue: when batch had 1 task × N trajectories,
        advantage normalization was within a single task (std≈0 → advantage≈0 → no gradient).
        Now batch has N tasks × 1 trajectory each, advantage is normalized across tasks,
        giving a meaningful learning signal even with binary rewards.
        """
        import pickle as _pickle
        import torch.distributed as _dist
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        self.policy.model.train()
        step_metrics = defaultdict(list)
        accum_steps = self.config.gradient_accumulation_steps
        rank = self.config.local_rank
        world_size = self.config.world_size

        # ========== Phase A: Generate trajectories for single task ==========
        # GRPO 原始设计：同一个任务生成 num_samples_per_prompt 条 trajectory
        # advantage 在组内归一化，控制任务难度变量，只反映策略好坏
        assert len(batch) == 1, "train_step expects exactly 1 task per batch (GRPO group sampling)"
        task = batch[0]
        all_trajectories = []

        def is_valid_trajectory(traj) -> tuple:
            if not traj.messages or len(traj.messages) < 2:
                return False, f"too_few_messages({len(traj.messages) if traj.messages else 0})"
            for msg_idx, msg in enumerate(traj.messages):
                if hasattr(msg, 'content') and msg.content and len(msg.content) > 500:
                    most_common = max(set(msg.content), key=msg.content.count)
                    if msg.content.count(most_common) / len(msg.content) > 0.9:
                        return False, f"degenerate_content(msg={msg_idx})"
            return True, "ok"

        env = self._create_environment(domain)
        if self.is_main_process():
            print(f"\n[Step {step}] Task: {task.id}, generating {self.config.num_samples_per_prompt} trajectories...")

        try:
            trajs = self.policy.generate_responses(
                task=task, environment=env,
                num_samples=self.config.num_samples_per_prompt,
                domain=domain,
            )
        except Exception as e:
            if self.is_main_process():
                print(f"  ✗ Generation failed for {task.id}: {e}")
            trajs = []

        for traj in trajs:
            ok, reason = is_valid_trajectory(traj)
            if ok:
                all_trajectories.append(traj)
            elif self.is_main_process():
                print(f"  ⚠ Filtered trajectory: {reason}")

        all_tasks = [task] * len(all_trajectories)

        # sync after all generation
        self.accelerator.wait_for_everyone()

        # ========== Phase A2: Compute rewards on rank 0, broadcast ==========
        all_rewards = []
        all_reward_infos = []

        if self.is_main_process():
            REWARD_TIMEOUT = 300

            def _compute_reward(args):
                idx, task, traj = args
                return idx, self.oracle.compute_reward(
                    task=task, trajectory=traj, domain=domain, solo_mode=False,
                )

            results = [None] * len(all_tasks)
            with ThreadPoolExecutor(max_workers=len(all_tasks)) as executor:
                future_map = {
                    executor.submit(_compute_reward, (i, t, tr)): i
                    for i, (t, tr) in enumerate(zip(all_tasks, all_trajectories))
                }
                for future in future_map:
                    i = future_map[future]
                    try:
                        idx, reward_info = future.result(timeout=REWARD_TIMEOUT)
                        results[idx] = reward_info
                    except FuturesTimeoutError:
                        print(f"  Reward TIMEOUT for task {i+1}", flush=True)
                    except Exception as e:
                        print(f"  Reward ERROR for task {i+1}: {e}", flush=True)

            for reward_info in results:
                if reward_info is not None:
                    all_reward_infos.append(reward_info)
                    all_rewards.append(reward_info.reward)
                else:
                    all_reward_infos.append(None)
                    all_rewards.append(0.0)

            print(f"  [Rank 0] Rewards: {all_rewards}", flush=True)

        # broadcast rewards to all ranks
        if world_size > 1:
            payload = _pickle.dumps((all_rewards, all_reward_infos)) if self.is_main_process() else b""
            length_tensor = torch.tensor([len(payload)], dtype=torch.long, device=self.device)
            _dist.broadcast(length_tensor, src=0)
            payload_len = length_tensor.item()
            if self.is_main_process():
                padded_bytes = bytes(payload) + bytes(payload_len - len(payload))
                payload_tensor = torch.tensor(list(padded_bytes), dtype=torch.uint8, device=self.device)
            else:
                payload_tensor = torch.zeros(payload_len, dtype=torch.uint8, device=self.device)
            _dist.broadcast(payload_tensor, src=0)
            if not self.is_main_process():
                all_rewards, all_reward_infos = _pickle.loads(payload_tensor.cpu().numpy().tobytes())

        # ========== Phase B: Normalize advantages within group (same task) ==========
        # Only skip if we have zero valid trajectories
        local_should_skip = (len(all_trajectories) == 0 or len(all_rewards) == 0)
        should_skip = self._sync_skip_decision(local_should_skip)

        # log all trajectories regardless
        for traj, reward, reward_info in zip(all_trajectories, all_rewards, all_reward_infos):
            self.trajectory_buffer.add(domain=domain, task=task, trajectory=traj, reward=reward)
            if self.is_main_process() and self.trajectory_logger and step % self.config.trajectory_log_interval == 0:
                self.trajectory_logger.log_trajectory(
                    task=task, trajectory=traj, reward=reward,
                    domain=domain, step=step, sample_idx=0, reward_info=reward_info,
                )

        if should_skip:
            if self.is_main_process():
                print(f"  ⏭ Skipping update: no valid trajectories for task {task.id}")
            micro_count = 0
        else:
            # GRPO group-level advantage normalization (same task, G samples)
            global_mean = float(np.mean(all_rewards))
            global_std  = max(float(np.std(all_rewards)), 1e-6)
            raw_advs = [(r - global_mean) / global_std for r in all_rewards]
            advantages = torch.tensor(
                [max(-5.0, min(5.0, a)) for a in raw_advs],
                dtype=torch.float32, device=self.device,
            )

            if self.is_main_process():
                print(f"  Task: {task.id} | mean={global_mean:.3f}, std={global_std:.3f}, "
                      f"rewards={all_rewards}")
                for i, (traj, reward, adv) in enumerate(
                    zip(all_trajectories, all_rewards, advantages)
                ):
                    status = "✓" if reward > 0.5 else "✗"
                    print(f"  [{status}] Sample {i+1}: reward={reward:.3f}, "
                          f"adv={adv.item():+.3f}, messages={len(traj.messages)}")

            # all trajectories are valid (no None in list)
            valid_trajs = all_trajectories
            valid_advs  = advantages

            # ========== Phase B2: Compute loss and backward ==========
            self.policy.model.train()
            print(f"  [Rank {rank}] Computing GRPO loss for {len(valid_trajs)} trajectories, "
                  f"advantages={valid_advs.tolist()}")

            micro_count = 0
            try:
                with self.accelerator.autocast():
                    loss, num_valid = self.policy.compute_grpo_loss(valid_trajs, valid_advs)

                    ewc_loss = torch.tensor(0.0, device=self.device)
                    if hasattr(self.cl_algorithm, 'compute_ewc_loss'):
                        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
                        ewc_loss = self.cl_algorithm.compute_ewc_loss(unwrapped_model)
                        if ewc_loss.item() > 0:
                            loss = loss + ewc_loss
                            if self.is_main_process():
                                print(f"  [EWC] Added EWC loss: {ewc_loss.item():.4f}")

                print(f"  [Rank {rank}] Loss computed: {loss.item():.4f}, num_valid={num_valid}")
                print(f"  [Rank {rank}] Entering backward...")
                self.accelerator.backward(loss / accum_steps)
                print(f"  [Rank {rank}] Backward done.")
                micro_count = 1

                step_metrics["loss"].append(float(loss.detach().float().item()))
                step_metrics["reward_mean"].append(global_mean)
                step_metrics["reward_max"].append(float(np.max(all_rewards)))
                step_metrics["reward_min"].append(float(np.min(all_rewards)))
                step_metrics["reward_std"].append(float(np.std(all_rewards)))
                step_metrics["num_valid_traj"].append(num_valid)

            except Exception as e:
                print(f"  [Rank {rank}] ✗ Loss computation failed: {e}")
                import traceback; traceback.print_exc()
                # Do NOT do a dummy backward here — DeepSpeed ZeRO-2 forbids a second
                # reduce in the same step and will crash with "gradient computed twice".
                # compute_grpo_loss already guarantees a gradient path via mask×0 terms,
                # so if we reach here the loss itself failed to compute; just skip.
                micro_count = 0

        # ========== Optimizer step (respect gradient_accumulation_steps) ==========
        # micro_count=1 means we did a real backward this step.
        # micro_count=0 means we skipped (no data or loss failed).
        # On skip we still need to decide whether to flush accumulated gradients:
        # if (step+1) % accum_steps == 0 and we have any accumulated grad, flush it.
        should_optimizer_step = ((step + 1) % accum_steps == 0)
        if should_optimizer_step:
            self.accelerator.clip_grad_norm_(self.policy.model.parameters(), self.config.max_grad_norm)
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
            if self.policy.scheduler is not None:
                self.policy.scheduler.step()

        self.cl_algorithm.post_step_hook(self, domain)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.is_main_process() and step_metrics:
            avg_loss = float(np.mean(step_metrics["loss"])) if step_metrics.get("loss") else 0.0
            avg_reward = float(np.mean(step_metrics["reward_mean"])) if step_metrics.get("reward_mean") else 0.0
            print(f"\n[Step {step} Summary] loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}")

        if step_metrics:
            return self._aggregate_metrics(step_metrics)
        return None

    def _aggregate_metrics(self, metrics: dict) -> dict:
        aggregated = {}
        for key, values in metrics.items():
            if not values:
                continue
            mean_value = float(np.mean(values))
            t = torch.tensor(mean_value, device=self.device)
            t = self.accelerator.reduce(t, reduction="mean")
            aggregated[key] = t.item()
        return aggregated

    def _sample_batch(self, train_tasks: list[Task], domain: str) -> list[Task]:
        batch = self.data_loader.sample_batch(domain, self.config.batch_size_per_gpu, split="train")
        return self.cl_algorithm.augment_batch(batch, domain)

    def _create_environment(self, domain: str):
        env_constructor = registry.get_env_constructor(domain)
        return env_constructor()

    def evaluate_all_tasks(self, current_task_idx: int):
        if not self.is_main_process():
            return
        results = {}
        # 只评估已训练的 domain（不包括最后一个用于泛化测试的 domain）
        for task_idx in range(current_task_idx + 1):
            domain = self.config.task_order[task_idx]
            results[domain] = self.evaluate_task_pass_at_k(domain)

        transfer_metrics = self._compute_transfer_metrics(results, current_task_idx)
        self.metrics.log_transfer(current_task_idx, transfer_metrics)

        buffer_stats = self.trajectory_buffer.get_statistics()
        self.metrics.log_buffer_stats(current_task_idx, buffer_stats)

    def evaluate_task_pass_at_k(self, domain: str, num_eval_tasks: int = None, num_samples: int = None, k: int = None) -> dict:
        """
        使用 pass@k 指标评估任务性能。
        一次性计算 pass@1 到 pass@n，充分利用采样结果。

        pass@k 表示在 n 次尝试中至少有 k 次成功的概率。
        计算公式: pass@k = C(c, k) / C(n, k)，其中 c 是成功次数，n 是总尝试次数。

        Args:
            domain: 要评估的领域
            num_eval_tasks: 评估的任务数量
            num_samples: 每个任务生成的样本数
            k: pass@k 中的 k 值（用于主要指标，但会同时计算 pass@1 到 pass@n）

        Returns:
            包含 pass@1 到 pass@n 等指标的字典
        """
        if not self.is_main_process():
            return {}

        import time
        import math

        # 使用配置中的默认值
        if num_eval_tasks is None:
            num_eval_tasks = self.config.num_eval_tasks
        if num_samples is None:
            num_samples = self.config.num_eval_samples
        if k is None:
            k = self.config.pass_at_k

        print(f"\n[Eval] Starting pass@1~{num_samples} evaluation on {domain} with {num_eval_tasks} tasks, {num_samples} samples each...")
        self.policy.model.eval()
        eval_tasks = self.data_loader.get_eval_tasks(domain)[:num_eval_tasks]

        # 为每个 k 值维护一个 scores 列表
        pass_at_k_scores = {i: [] for i in range(1, num_samples + 1)}
        all_rewards = []
        tool_accs = []

        with torch.no_grad():
            for idx, task in enumerate(eval_tasks):
                print(f"[Eval] Task {idx+1}/{len(eval_tasks)}: {task.id}", end=" ", flush=True)

                # 为每个任务生成多个样本
                task_rewards = []
                start_time = time.time()

                for sample_idx in range(num_samples):
                    env = self._create_environment(domain)
                    original_temp = self.policy.config.temperature
                    self.policy.config.temperature = 0.7  # 使用较高温度增加多样性

                    try:
                        trajectories = self.policy.generate_responses(task, env, 1, domain)
                        if trajectories:
                            reward_info = self.oracle.compute_reward(task, trajectories[0], domain, solo_mode=False)
                            task_rewards.append(reward_info.reward)

                            if reward_info.action_checks:
                                matches = sum(1 for ac in reward_info.action_checks if ac.action_match)
                                total = len(reward_info.action_checks)
                                tool_accs.append(matches / total if total > 0 else 0.0)
                    except Exception as e:
                        pass
                    finally:
                        self.policy.config.temperature = original_temp

                gen_time = time.time() - start_time

                n = len(task_rewards)
                c = sum(1 for r in task_rewards if r > 0.5)  # 成功次数
                all_rewards.extend(task_rewards)

                # 计算该任务的 pass@1 到 pass@n
                pass_at_k_str = []
                for k_val in range(1, num_samples + 1):
                    if n >= k_val:
                        # pass@k = C(c, k) / C(n, k)
                        if c >= k_val:
                            pass_at_k_val = math.comb(c, k_val) / math.comb(n, k_val)
                        else:
                            pass_at_k_val = 0.0
                        pass_at_k_scores[k_val].append(pass_at_k_val)
                        pass_at_k_str.append(f"@{k_val}={pass_at_k_val:.2f}")

                print(f"({gen_time:.1f}s) -> success={c}/{n}, {', '.join(pass_at_k_str)}")

        # 构建结果
        result = {
            "reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "reward_std": float(np.std(all_rewards)) if all_rewards else 0.0,
            "tool_accuracy": float(np.mean(tool_accs)) if tool_accs else 0.0,
            "num_evaluated": len(eval_tasks),
            "num_samples": num_samples,
        }

        # 添加 pass@1 到 pass@n 的结果
        print(f"\n[Eval] Results for {domain}:")
        for k_val in range(1, num_samples + 1):
            if pass_at_k_scores[k_val]:
                mean_val = float(np.mean(pass_at_k_scores[k_val]))
                std_val = float(np.std(pass_at_k_scores[k_val]))
                result[f"pass_at_{k_val}_mean"] = mean_val
                result[f"pass_at_{k_val}_std"] = std_val
                print(f"  pass@{k_val}: {mean_val:.3f} ± {std_val:.3f}")
            else:
                result[f"pass_at_{k_val}_mean"] = 0.0
                result[f"pass_at_{k_val}_std"] = 0.0

        # 保持向后兼容：pass_at_k_mean 使用配置中指定的 k 值
        result["pass_at_k_mean"] = result.get(f"pass_at_{k}_mean", 0.0)
        result["pass_at_k_std"] = result.get(f"pass_at_{k}_std", 0.0)
        result["k"] = k

        return result

    def evaluate_task_pass_at_k_distributed(self, domain: str, num_eval_tasks: int = None, num_samples: int = None, k: int = None) -> dict:
        """
        分布式版本的 pass@k 评估，所有 rank 都参与生成。

        优化：每个 rank 生成 1 个 trajectory，收集前 num_samples 个 rank 的结果。
        这样每个任务只需要 1 次同步，而不是 num_samples 次。

        Args:
            domain: 要评估的领域
            num_eval_tasks: 评估的任务数量
            num_samples: 每个任务需要的样本数（从各 rank 收集）
            k: pass@k 中的 k 值

        Returns:
            包含 pass@1 到 pass@n 等指标的字典（只在 main process 返回有效值）
        """
        import time
        import math
        import pickle

        # 使用配置中的默认值
        if num_eval_tasks is None:
            num_eval_tasks = self.config.num_eval_tasks
        if num_samples is None:
            num_samples = self.config.num_eval_samples
        if k is None:
            k = self.config.pass_at_k

        rank = self.accelerator.process_index
        world_size = self.config.world_size

        if self.is_main_process():
            print(f"\n[Eval] Starting distributed pass@k evaluation on {domain}")
            print(f"  Tasks: {num_eval_tasks}, Samples needed: {num_samples}, Ranks available: {world_size}")

        self.policy.model.eval()
        eval_tasks = self.data_loader.get_eval_tasks(domain)[:num_eval_tasks]

        # 为每个 k 值维护一个 scores 列表
        pass_at_k_scores = {i: [] for i in range(1, num_samples + 1)}
        all_rewards = []
        tool_accs = []

        with torch.no_grad():
            for idx, task in enumerate(eval_tasks):
                if self.is_main_process():
                    print(f"[Eval] Task {idx+1}/{len(eval_tasks)}: {task.id}", end=" ", flush=True)

                start_time = time.time()
                env = self._create_environment(domain)

                original_temp = self.policy.config.temperature
                self.policy.config.temperature = 0.7  # 使用较高温度增加多样性

                # 每个 rank 生成 1 个 trajectory
                local_traj = None
                local_reward = -999.0  # sentinel value
                try:
                    trajectories = self.policy.generate_responses(task, env, 1, domain)
                    if trajectories:
                        local_traj = trajectories[0]
                        # 每个 rank 都计算自己的 reward
                        reward_info = self.oracle.compute_reward(task, local_traj, domain, solo_mode=False)
                        local_reward = reward_info.reward

                        if reward_info.action_checks:
                            matches = sum(1 for ac in reward_info.action_checks if ac.action_match)
                            total = len(reward_info.action_checks)
                            local_tool_acc = matches / total if total > 0 else 0.0
                        else:
                            local_tool_acc = -1.0  # sentinel
                    else:
                        local_tool_acc = -1.0
                except Exception as e:
                    print(f"[Rank {rank}] Error: {e}", flush=True)
                    local_tool_acc = -1.0
                finally:
                    self.policy.config.temperature = original_temp

                # 同步所有 rank
                self.accelerator.wait_for_everyone()

                # 收集所有 rank 的 reward 到 rank 0
                local_reward_tensor = torch.tensor([local_reward], dtype=torch.float32, device=self.device)
                local_tool_acc_tensor = torch.tensor([local_tool_acc], dtype=torch.float32, device=self.device)

                all_reward_tensors = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(world_size)]
                all_tool_acc_tensors = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(world_size)]

                dist.all_gather(all_reward_tensors, local_reward_tensor)
                dist.all_gather(all_tool_acc_tensors, local_tool_acc_tensor)

                # 只在 main process 计算统计
                if self.is_main_process():
                    # 收集有效的 rewards（排除 sentinel value）
                    task_rewards = []
                    task_tool_accs = []

                    for r_idx in range(min(num_samples, world_size)):
                        r = all_reward_tensors[r_idx].item()
                        t = all_tool_acc_tensors[r_idx].item()
                        if r > -900:  # 有效值
                            task_rewards.append(r)
                        if t >= 0:  # 有效值
                            task_tool_accs.append(t)

                    gen_time = time.time() - start_time
                    n = len(task_rewards)
                    c = sum(1 for r in task_rewards if r > 0.5)
                    all_rewards.extend(task_rewards)
                    tool_accs.extend(task_tool_accs)

                    # 计算 pass@k
                    pass_at_k_str = []
                    for k_val in range(1, min(n, num_samples) + 1):
                        if n >= k_val:
                            if c >= k_val:
                                pass_at_k_val = math.comb(c, k_val) / math.comb(n, k_val)
                            else:
                                pass_at_k_val = 0.0
                            pass_at_k_scores[k_val].append(pass_at_k_val)
                            pass_at_k_str.append(f"@{k_val}={pass_at_k_val:.2f}")

                    print(f"({gen_time:.1f}s) -> success={c}/{n}, {', '.join(pass_at_k_str)}")

        # 最终同步前打印调试信息
        rank = self.accelerator.process_index
        print(f"[Rank {rank}] Eval loop done, entering final sync...", flush=True)

        # 最终同步
        self.accelerator.wait_for_everyone()

        print(f"[Rank {rank}] Final sync done.", flush=True)

        # 只在 main process 返回有效结果
        if self.is_main_process():
            result = {
                "reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "reward_std": float(np.std(all_rewards)) if all_rewards else 0.0,
                "tool_accuracy": float(np.mean(tool_accs)) if tool_accs else 0.0,
                "num_evaluated": len(eval_tasks),
                "num_samples": num_samples,
            }

            print(f"\n[Eval] Results for {domain}:")
            for k_val in range(1, num_samples + 1):
                if pass_at_k_scores[k_val]:
                    mean_val = float(np.mean(pass_at_k_scores[k_val]))
                    std_val = float(np.std(pass_at_k_scores[k_val]))
                    result[f"pass_at_{k_val}_mean"] = mean_val
                    result[f"pass_at_{k_val}_std"] = std_val
                    print(f"  pass@{k_val}: {mean_val:.3f} ± {std_val:.3f}")
                else:
                    result[f"pass_at_{k_val}_mean"] = 0.0
                    result[f"pass_at_{k_val}_std"] = 0.0

            result["pass_at_k_mean"] = result.get(f"pass_at_{k}_mean", 0.0)
            result["pass_at_k_std"] = result.get(f"pass_at_{k}_std", 0.0)
            result["k"] = k

            return result

        return {}


    def evaluate_generalization(self) -> dict:
        """
        评估模型在最后一个 domain（未训练）上的泛化性能。

        Returns:
            包含泛化性能指标的字典
        """
        if not self.is_main_process():
            return

        if len(self.config.task_order) < 2:
            print("[Generalization] Need at least 2 domains for generalization test")
            return {}

        # 最后一个 domain 用于泛化测试
        generalization_domain = self.config.task_order[-1]

        print(f"\n{'='*60}")
        print(f"[Generalization] Testing on unseen domain: {generalization_domain}")
        print(f"{'='*60}")

        results = self.evaluate_task_pass_at_k(generalization_domain)

        print(f"{'='*60}")
        print(f"[Generalization] Results on {generalization_domain}:")
        print(f"  pass@{results.get('k', self.config.pass_at_k)}_mean: {results.get('pass_at_k_mean', 0):.3f}")
        print(f"  reward_mean: {results.get('reward_mean', 0):.3f}")
        print(f"{'='*60}\n")

        return results

    def evaluate_task(self, domain: str, num_eval_tasks: int = None) -> dict:
        """保留原有的简单评估方法，用于训练过程中的快速评估"""
        if not self.is_main_process():
            return {}

        if num_eval_tasks is None:
            num_eval_tasks = self.config.num_eval_tasks

        import time
        print(f"\n[Eval] Starting evaluation on {domain} with {num_eval_tasks} tasks...")
        self.policy.model.eval()
        eval_tasks = self.data_loader.get_eval_tasks(domain)[:num_eval_tasks]

        rewards, tool_accs = [], []
        pass_count = 0

        with torch.no_grad():
            for idx, task in enumerate(eval_tasks):
                print(f"[Eval] Task {idx+1}/{len(eval_tasks)}: {task.id}", end=" ", flush=True)
                env = self._create_environment(domain)

                original_temp = self.policy.config.temperature
                self.policy.config.temperature = 0.1

                print("(generating...)", end=" ", flush=True)
                start_time = time.time()
                try:
                    trajectories = self.policy.generate_responses(task, env, 1, domain)
                    gen_time = time.time() - start_time
                    print(f"({gen_time:.1f}s)", end=" ", flush=True)
                except Exception as e:
                    print(f"-> ERROR: {e}")
                    self.policy.config.temperature = original_temp
                    continue

                self.policy.config.temperature = original_temp

                if not trajectories:
                    print("-> No trajectory generated")
                    continue

                reward_info = self.oracle.compute_reward(task, trajectories[0], domain, solo_mode=False)
                rewards.append(reward_info.reward)
                status = "✓" if reward_info.reward > 0.5 else "✗"
                print(f"-> reward={reward_info.reward:.3f} [{status}]")

                if reward_info.reward > 0.5:
                    pass_count += 1

                if reward_info.action_checks:
                    matches = sum(1 for ac in reward_info.action_checks if ac.action_match)
                    total = len(reward_info.action_checks)
                    tool_accs.append(matches / total if total > 0 else 0.0)

        print(f"[Eval] Completed: {len(rewards)} tasks evaluated, pass_rate={pass_count}/{len(rewards)}")

        if rewards:
            return {
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "pass_rate": pass_count / len(rewards),
                "tool_accuracy": float(np.mean(tool_accs)) if tool_accs else 0.0,
                "num_evaluated": len(rewards),
            }
        return {
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "pass_rate": 0.0,
            "tool_accuracy": 0.0,
            "num_evaluated": 0,
        }

    def _compute_transfer_metrics(self, results: dict, current_task_idx: int) -> dict:
        """
        计算迁移指标，使用 pass@k_mean 作为主要性能指标。
        """
        # 获取训练的 domain 列表（不包括最后一个泛化测试 domain）
        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order

        # 使用 pass_at_k_mean 作为主要指标，如果没有则回退到 reward_mean
        def get_performance(result):
            return result.get("pass_at_k_mean", result.get("reward_mean", 0.0))

        if current_task_idx > 0:
            # 后向迁移：之前任务的平均性能
            prev = [get_performance(results[train_domains[i]]) for i in range(current_task_idx)]
            backward_transfer = float(np.mean(prev))
        else:
            backward_transfer = 0.0

        current_domain = train_domains[current_task_idx]
        current_performance = get_performance(results[current_domain])
        avg_performance = float(np.mean([get_performance(r) for r in results.values()]))

        return {
            "backward_transfer": backward_transfer,
            "current_performance": current_performance,
            "average_performance": avg_performance,
            "num_tasks_seen": current_task_idx + 1,
            "metric_type": "pass_at_k" if "pass_at_k_mean" in results.get(current_domain, {}) else "reward",
        }

    def save_checkpoint(self, task_idx: int, domain: str = None):
        """
        保存 checkpoint。所有 rank 都需要调用此方法来保存各自的 optimizer state。

        Args:
            task_idx: 当前 task 索引
            domain: 当前 domain 名称，用于保存完整模型到 checkpoints/after_{domain}/
        """
        if self.is_main_process():
            print(f"[Checkpoint] Saving task {task_idx} checkpoint...")

        ckpt_dir = Path(self.config.log_dir) / f"checkpoint_task_{task_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 所有 rank 都调用 save_checkpoint 来保存各自的 optimizer state
        unwrapped = self.accelerator.unwrap_model(self.policy.model)
        self.policy.save_checkpoint(str(ckpt_dir / "model"), model_override=unwrapped)

        # 关键：等待所有 rank 完成保存后再继续
        self.accelerator.wait_for_everyone()

        # 同时保存完整模型到 checkpoints/after_{domain}/ 目录（类似 single domain）
        if domain:
            model_dir = Path(self.config.log_dir) / "checkpoints" / f"after_{domain}"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.policy.save_checkpoint(str(model_dir), model_override=unwrapped)

            # 等待所有 rank 完成第二次保存
            self.accelerator.wait_for_everyone()

        # 只有 main process 保存其他文件
        if self.is_main_process():
            self.trajectory_buffer.save(str(ckpt_dir / "trajectories.json"))
            self.metrics.save()

            import json
            with open(ckpt_dir / "config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            if domain:
                print(f"[Saved] Model checkpoint saved to {model_dir}")

        # 最终同步，确保所有操作完成
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, checkpoint_dir: str, load_optimizer: bool = False):
        """
        加载 checkpoint。自动检测目录结构：
        - 如果 checkpoint_dir/model 存在，从 checkpoint_dir/model 加载
        - 否则直接从 checkpoint_dir 加载（用于 checkpoints/after_X 目录）

        Args:
            checkpoint_dir: checkpoint 目录路径
            load_optimizer: 是否加载 optimizer state，默认 False（从 task 继续训练时不需要）
        """
        ckpt_path = Path(checkpoint_dir)
        model_subdir = ckpt_path / "model"

        if model_subdir.exists():
            # checkpoint_task_X 格式：包含 model 子目录
            self.policy.load_checkpoint(str(model_subdir), load_optimizer=load_optimizer)
        else:
            # checkpoints/after_X 格式：直接包含模型文件，不加载 optimizer
            self.policy.load_checkpoint(str(ckpt_path), load_optimizer=False)

    def train_single_domain_only(self):
        """
        简化的单任务训练流程，用于训练 FWT baseline。
        只训练一个 domain，不计算 FWT/BWT 等指标。
        """
        assert len(self.config.task_order) == 1, "train_single_domain_only requires exactly one domain"
        domain = self.config.task_order[0]

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Single Domain Training] {domain}")
            print(f"{'='*60}\n")
            # 启动进度条
            if self.metrics:
                self.metrics.start_task_progress(task_idx=0, domain=domain)

        # 训练
        self.train_task(domain, task_idx=0)

        if self.is_main_process():
            # 关闭进度条
            if self.metrics:
                self.metrics.close_task_progress()

        # 同步所有 rank
        self.accelerator.wait_for_everyone()

        # 保存模型 - 所有 rank 都需要保存 optimizer state
        save_dir = Path(self.config.log_dir).parent / "model"
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.policy.model)
        self.policy.save_checkpoint(str(save_dir), model_override=unwrapped)

        # 只有 main process 保存配置
        if self.is_main_process():
            import json
            with open(Path(self.config.log_dir).parent / "config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            print(f"\n[Saved] Model saved to {save_dir}")

        self.accelerator.wait_for_everyone()

        if self.is_main_process():
            if self.metrics:
                self.metrics.save()
                self.metrics.close()

    # ========== 分阶段运行方法 ==========

    def run_phase1_zero_shot(self):
        """
        Phase 1: Zero-shot 评估
        在未训练的模型上评估所有 domain 的性能，作为 baseline。
        结果保存到 {log_dir}/phase1_zero_shot.json
        """
        import json

        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order
        generalization_domain = self.config.task_order[-1] if len(self.config.task_order) > 1 else None

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 1] Zero-shot evaluation (before any training)")
            print(f"{'='*60}")
            print(f"Training domains: {train_domains}")
            print(f"Generalization domain: {generalization_domain}")

        zero_shot_performance = {}

        # 评估所有训练 domain
        for domain in train_domains:
            if self.is_main_process():
                print(f"\n[Zero-shot] Testing {domain}...")
            results = self.evaluate_task_pass_at_k_distributed(domain)
            if self.is_main_process():
                # 保存完整的评估结果（包含 pass@1~n）
                zero_shot_performance[domain] = results
                self.metrics.log_generalization(f"zero_shot_{domain}", results)

        # 评估泛化 domain
        if generalization_domain:
            if self.is_main_process():
                print(f"\n[Zero-shot] Testing {generalization_domain} (generalization domain)...")
            results = self.evaluate_task_pass_at_k_distributed(generalization_domain)
            if self.is_main_process():
                zero_shot_performance[generalization_domain] = results
                self.metrics.log_generalization(f"zero_shot_{generalization_domain}", results)

        self.accelerator.wait_for_everyone()

        # 保存结果
        if self.is_main_process():
            output_path = Path(self.config.log_dir) / "phase1_zero_shot.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(zero_shot_performance, f, indent=2)

            print(f"\n[Phase 1 Summary]")
            for domain, res in zero_shot_performance.items():
                if isinstance(res, dict):
                    pass_at_1 = res.get('pass_at_1_mean', res.get('pass_at_k_mean', 0))
                    print(f"  {domain}: pass@1={pass_at_1:.3f}")
                else:
                    print(f"  {domain}: {res:.3f}")
            print(f"\nResults saved to: {output_path}")

            self.metrics.save()

    def run_phase1_5_single_domain(self, load_phase1_results: str = None):
        """
        Phase 1.5: 单任务 baseline 评估
        加载单任务训练后的 checkpoint，评估各 domain 的性能，用于 FWT 计算。
        结果保存到 {log_dir}/phase1_5_single_domain.json
        """
        import json

        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order

        # 加载 phase 1 结果（用于 fallback）
        zero_shot_performance = {}
        if load_phase1_results and Path(load_phase1_results).exists():
            with open(load_phase1_results) as f:
                zero_shot_performance = json.load(f)
            if self.is_main_process():
                print(f"[Phase 1.5] Loaded zero-shot results from {load_phase1_results}")

        if not self.config.single_domain_checkpoint_dir:
            if self.is_main_process():
                print(f"\n[Phase 1.5] No single_domain_checkpoint_dir specified!")
                print(f"[Phase 1.5] Please provide --single_domain_checkpoint_dir")
            return

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 1.5] Single-task baseline evaluation (for FWT)")
            print(f"[Phase 1.5] Loading from: {self.config.single_domain_checkpoint_dir}")
            print(f"{'='*60}")

        single_task_performance = {}
        single_ckpt_dir = Path(self.config.single_domain_checkpoint_dir)

        for domain in train_domains:
            domain_ckpt = single_ckpt_dir / domain / "model"
            if domain_ckpt.exists():
                if self.is_main_process():
                    print(f"\n[Single-task] Loading {domain} checkpoint...")

                # 保存当前模型状态
                current_state = {k: v.clone() for k, v in self.policy.model.state_dict().items()}

                # 加载单任务模型（所有 rank 都需要加载）
                self.policy.load_checkpoint(str(domain_ckpt), load_optimizer=False)
                self.accelerator.wait_for_everyone()

                # 评估（所有 rank 参与）
                results = self.evaluate_task_pass_at_k_distributed(domain)
                if self.is_main_process():
                    # 保存完整的评估结果（包含 pass@1~n）
                    single_task_performance[domain] = results
                    self.metrics.log_generalization(f"single_task_{domain}", results)

                # 恢复当前模型状态（所有 rank 都需要恢复）
                self.policy.model.load_state_dict(current_state)
                self.accelerator.wait_for_everyone()
            else:
                if self.is_main_process():
                    print(f"[Single-task] Warning: No checkpoint for {domain}, using zero-shot as baseline")
                    single_task_performance[domain] = zero_shot_performance.get(domain, {})

        self.accelerator.wait_for_everyone()

        # 保存结果
        if self.is_main_process():
            output_path = Path(self.config.log_dir) / "phase1_5_single_domain.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(single_task_performance, f, indent=2)

            print(f"\n[Phase 1.5 Summary]")
            for domain, res in single_task_performance.items():
                if isinstance(res, dict):
                    pass_at_1 = res.get('pass_at_1_mean', res.get('pass_at_k_mean', 0))
                    print(f"  {domain}: pass@1={pass_at_1:.3f}")
                else:
                    print(f"  {domain}: {res:.3f}")
            print(f"\nResults saved to: {output_path}")

            self.metrics.save()

    def run_phase2_training(self, load_phase1_results: str = None, load_phase1_5_results: str = None):
        """
        Phase 2: 持续学习训练
        顺序训练所有 domain，每训完一个 domain 评估所有已训练的 domain。
        结果保存到 {log_dir}/phase2_training.json
        """
        import json

        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order

        # 加载之前阶段的结果
        zero_shot_performance = {}
        single_task_performance = {}

        if load_phase1_results and Path(load_phase1_results).exists():
            with open(load_phase1_results) as f:
                zero_shot_performance = json.load(f)
            if self.is_main_process():
                print(f"[Phase 2] Loaded zero-shot results from {load_phase1_results}")

        if load_phase1_5_results and Path(load_phase1_5_results).exists():
            with open(load_phase1_5_results) as f:
                single_task_performance = json.load(f)
            if self.is_main_process():
                print(f"[Phase 2] Loaded single-domain results from {load_phase1_5_results}")
        else:
            single_task_performance = zero_shot_performance.copy()

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 2] Continual Learning Training")
            print(f"{'='*60}")
            print(f"Training domains: {train_domains}")

        # 性能矩阵（只记录原始评测结果，BWT/FWT 在 phase3 计算）
        performance_matrix = {}
        eval_results_matrix = {}  # 保存完整的 pass@k 评估结果

        # 支持断点续训
        start_task_idx = getattr(self.config, 'resume_from_task', 0)
        skip_training_for_resume_task = getattr(self.config, 'skip_training_for_resume_task', False)
        resume_from_eval_domain = getattr(self.config, 'resume_from_eval_domain', None)

        # 如果是断点续训，尝试加载之前保存的结果
        if start_task_idx > 0:
            phase2_path = Path(self.config.log_dir) / "phase2_training.json"
            if phase2_path.exists():
                try:
                    with open(phase2_path) as f:
                        prev_results = json.load(f)
                    # 加载之前的 performance_matrix（key 是字符串，需要转换为 int）
                    for k, v in prev_results.get("performance_matrix", {}).items():
                        performance_matrix[int(k)] = v
                    # 加载之前的 eval_results_matrix
                    for k, v in prev_results.get("eval_results_matrix", {}).items():
                        eval_results_matrix[int(k)] = v
                    if self.is_main_process():
                        print(f"[Resume] Loaded previous results from {phase2_path}")
                        print(f"[Resume] performance_matrix keys: {list(performance_matrix.keys())}")
                except Exception as e:
                    if self.is_main_process():
                        print(f"[Warning] Failed to load previous results: {e}")

        if start_task_idx > 0 and self.is_main_process():
            print(f"\n[Resume] Starting from task {start_task_idx} ({train_domains[start_task_idx]})")
            if skip_training_for_resume_task:
                print(f"[Resume] Skipping training for task {start_task_idx}, starting from evaluation")
            if resume_from_eval_domain:
                print(f"[Resume] Will resume evaluation from domain: {resume_from_eval_domain}")

        for task_idx, domain in enumerate(train_domains):
            if task_idx < start_task_idx:
                if self.is_main_process():
                    print(f"\n[Skip] Task {task_idx} ({domain}) - already completed")
                continue

            if self.is_main_process():
                print(f"\n{'='*60}")
                print(f"[Phase 2] Task {task_idx}: Training on {domain}")
                print(f"{'='*60}")
                self.metrics.start_task_progress(task_idx, domain)

            # 切换 domain 时重置 optimizer 状态（避免旧 domain 的动量影响新 domain）
            # 包括断点重训的情况
            if task_idx > 0:
                if self.is_main_process():
                    print(f"[Domain Switch] Resetting optimizer for new domain: {domain}")
                self.policy.reset_optimizer()
                self.accelerator.wait_for_everyone()

            # 如果是 resume 的 task 且设置了跳过训练，则跳过训练阶段
            if task_idx == start_task_idx and skip_training_for_resume_task:
                if self.is_main_process():
                    print(f"[Skip] Skipping training for task {task_idx} ({domain}) - resuming from evaluation")
            else:
                # 训练当前任务
                self.train_task(domain, task_idx)

            if self.is_main_process():
                self.metrics.close_task_progress()

            # 评估阶段
            self.accelerator.wait_for_everyone()

            if self.is_main_process():
                print(f"\n[Evaluation after Task {task_idx}] Testing all trained domains (0~{task_idx})...")
                performance_matrix[task_idx] = {}

            # 确定从哪个 eval_idx 开始评估
            start_eval_idx = 0
            if task_idx == start_task_idx and resume_from_eval_domain:
                # 找到 resume_from_eval_domain 在 train_domains 中的索引
                for idx, d in enumerate(train_domains[:task_idx + 1]):
                    if d == resume_from_eval_domain:
                        start_eval_idx = idx
                        if self.is_main_process():
                            print(f"[Resume] Resuming evaluation from domain {resume_from_eval_domain} (index {idx})")
                        break
                # 清除 resume_from_eval_domain，只在第一个 task 使用
                resume_from_eval_domain = None

            # 评估所有已训练的 domain
            # 保存完整的评估结果（包含 pass@1~n）
            eval_results_full = {}
            for eval_idx in range(start_eval_idx, task_idx + 1):
                eval_domain = train_domains[eval_idx]
                if self.is_main_process():
                    print(f"\n[Eval] Testing {eval_domain}...")

                results = self.evaluate_task_pass_at_k_distributed(eval_domain)

                if self.is_main_process():
                    # 保存完整结果用于后续分析
                    eval_results_full[eval_domain] = results
                    # 用于 FWT/BWT 计算的主要指标
                    score = results.get('pass_at_k_mean', 0)
                    performance_matrix[task_idx][eval_domain] = score

            # 同步所有 rank
            rank = self.accelerator.process_index
            print(f"[Rank {rank}] Finished all evaluations, waiting for sync...", flush=True)
            self.accelerator.wait_for_everyone()
            if self.is_main_process():
                print(f"[Sync] All ranks synchronized after evaluation.")

            if self.is_main_process():
                # 保存完整评估结果到 matrix
                eval_results_matrix[task_idx] = eval_results_full

                # 简单打印当前评测结果（不计算 BWT/FWT，留到 phase3）
                print(f"\n[Evaluation Results after Task {task_idx}]")
                for eval_domain, score in performance_matrix[task_idx].items():
                    print(f"  {eval_domain}: {score:.3f}")

            # 保存 checkpoint
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(task_idx, domain=domain)

            # post_task_hook
            self.accelerator.wait_for_everyone()
            current_performance = None
            if self.is_main_process() and task_idx in performance_matrix:
                current_performance = performance_matrix[task_idx].get(domain, None)
            self.cl_algorithm.post_task_hook(self, domain, performance=current_performance)

            # 每个 domain 完成后立即保存结果（增量保存，防止中断丢失）
            if self.is_main_process():
                output_path = Path(self.config.log_dir) / "phase2_training.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                training_results = {
                    "performance_matrix": {str(k): v for k, v in performance_matrix.items()},
                    "eval_results_matrix": {str(k): v for k, v in eval_results_matrix.items()},
                    "zero_shot_performance": zero_shot_performance,
                    "single_task_performance": single_task_performance,
                    "train_domains": train_domains,
                    "last_completed_task": task_idx,
                    "last_completed_domain": domain,
                }
                with open(output_path, "w") as f:
                    json.dump(training_results, f, indent=2)

                print(f"[Task {task_idx}] Completed. Results saved to {output_path}\n")

        # 最终保存
        if self.is_main_process():
            output_path = Path(self.config.log_dir) / "phase2_training.json"
            training_results = {
                "performance_matrix": {str(k): v for k, v in performance_matrix.items()},
                "eval_results_matrix": {str(k): v for k, v in eval_results_matrix.items()},
                "zero_shot_performance": zero_shot_performance,
                "single_task_performance": single_task_performance,
                "train_domains": train_domains,
                "completed": True,
            }
            with open(output_path, "w") as f:
                json.dump(training_results, f, indent=2)

            print(f"\n[Phase 2] Training completed!")
            print(f"Results saved to: {output_path}")
            print(f"Run Phase 3 to compute BWT/FWT metrics.")

            self.metrics.save()

    def run_phase3_final_eval(self, load_phase1_results: str = None, load_phase1_5_results: str = None):
        """
        Phase 3: 最终评估
        加载 phase2 的结果，计算所有 CL 指标（BWT/FWT/AP），并进行最终评估。
        结果保存到 {log_dir}/phase3_final_eval.json
        """
        import json

        train_domains = self.config.task_order[:-1] if len(self.config.task_order) > 1 else self.config.task_order
        generalization_domain = self.config.task_order[-1] if len(self.config.task_order) > 1 else None

        # 加载之前阶段的结果
        zero_shot_performance = {}
        single_task_performance = {}

        if load_phase1_results and Path(load_phase1_results).exists():
            with open(load_phase1_results) as f:
                zero_shot_performance = json.load(f)
            if self.is_main_process():
                print(f"[Phase 3] Loaded zero-shot results from {load_phase1_results}")

        if load_phase1_5_results and Path(load_phase1_5_results).exists():
            with open(load_phase1_5_results) as f:
                single_task_performance = json.load(f)
            if self.is_main_process():
                print(f"[Phase 3] Loaded single-domain results from {load_phase1_5_results}")
        else:
            single_task_performance = zero_shot_performance.copy()

        # 加载 phase2 结果
        phase2_path = Path(self.config.log_dir) / "phase2_training.json"
        performance_matrix = {}
        eval_results_matrix = {}
        if phase2_path.exists():
            with open(phase2_path) as f:
                phase2_data = json.load(f)
                performance_matrix = {int(k): v for k, v in phase2_data.get("performance_matrix", {}).items()}
                eval_results_matrix = {int(k): v for k, v in phase2_data.get("eval_results_matrix", {}).items()}
                # 如果 phase2 保存了 zero_shot/single_task，优先使用
                if not zero_shot_performance and "zero_shot_performance" in phase2_data:
                    zero_shot_performance = phase2_data["zero_shot_performance"]
                if not single_task_performance and "single_task_performance" in phase2_data:
                    single_task_performance = phase2_data["single_task_performance"]
            if self.is_main_process():
                print(f"[Phase 3] Loaded training results from {phase2_path}")
                print(f"[Phase 3] performance_matrix keys: {list(performance_matrix.keys())}")

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Phase 3] Final Evaluation and Metrics Computation")
            print(f"{'='*60}")
            print(f"Train domains: {train_domains}")
            print(f"Generalization domain: {generalization_domain}")

        # ========== 1. 从 performance_matrix 计算中间指标 ==========
        if self.is_main_process():
            # 辅助函数
            def get_baseline_value(data, domain):
                val = data.get(domain, 0)
                if isinstance(val, dict):
                    return val.get('pass_at_k_mean', val.get('pass_at_1_mean', 0))
                return val

            print(f"\n{'='*60}")
            print(f"[Intermediate Metrics from Phase 2]")
            print(f"{'='*60}")

            # 计算 AP 列表（每个 task 训练后的平均性能）
            ap_list = []
            for task_idx in sorted(performance_matrix.keys()):
                if performance_matrix[task_idx]:
                    ap_t = sum(performance_matrix[task_idx].values()) / (task_idx + 1)
                    ap_list.append(ap_t)
                    print(f"  AP_{task_idx}: {ap_t:.3f}")

            # 计算 FWT（Forward Transfer）
            print(f"\n[FWT] Forward Transfer (after each task training):")
            fwt_values = []
            for task_idx, domain in enumerate(train_domains):
                baseline = get_baseline_value(single_task_performance, domain)
                if baseline == 0:
                    baseline = get_baseline_value(zero_shot_performance, domain)
                if task_idx in performance_matrix and domain in performance_matrix[task_idx]:
                    after_cl = performance_matrix[task_idx][domain]
                    fwt_j = after_cl - baseline
                    fwt_values.append(fwt_j)
                    print(f"  FWT_{task_idx} ({domain}): {fwt_j:+.3f} = {after_cl:.3f} - {baseline:.3f}")

            avg_fwt = sum(fwt_values) / len(fwt_values) if fwt_values else 0
            print(f"  Average FWT: {avg_fwt:+.3f}")

            # 计算 BWT 矩阵（Backward Transfer）
            print(f"\n[BWT Matrix] Backward Transfer:")
            bwt_matrix = {}
            for task_idx in sorted(performance_matrix.keys()):
                if task_idx == 0:
                    continue
                bwt_matrix[task_idx] = {}
                for prev_idx in range(task_idx):
                    prev_domain = train_domains[prev_idx]
                    if prev_idx in performance_matrix and prev_domain in performance_matrix[prev_idx]:
                        if prev_domain in performance_matrix[task_idx]:
                            current_perf = performance_matrix[task_idx][prev_domain]
                            after_train_perf = performance_matrix[prev_idx][prev_domain]
                            bwt_ij = current_perf - after_train_perf
                            bwt_matrix[task_idx][prev_domain] = bwt_ij
                            print(f"  BWT_{task_idx},{prev_idx} ({prev_domain}): {bwt_ij:+.3f} = {current_perf:.3f} - {after_train_perf:.3f}")

            # 计算平均 BWT
            all_bwt_values = []
            for task_bwt in bwt_matrix.values():
                all_bwt_values.extend(task_bwt.values())
            avg_bwt = sum(all_bwt_values) / len(all_bwt_values) if all_bwt_values else 0
            print(f"  Average BWT (from matrix): {avg_bwt:+.3f}")

        # ========== 2. 最终性能评估 ==========
        final_performance = {}
        final_performance_full = {}
        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[Final Evaluation] Testing all trained domains...")
            print(f"{'='*60}")

        for domain in train_domains:
            if self.is_main_process():
                print(f"\n[Final] Testing {domain}...")
            results = self.evaluate_task_pass_at_k_distributed(domain)
            if self.is_main_process():
                final_performance_full[domain] = results
                final_performance[domain] = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"final_{domain}", results)

        # 泛化性测试
        final_generalization = 0
        final_generalization_full = {}
        if generalization_domain:
            if self.is_main_process():
                print(f"\n[Generalization] Testing {generalization_domain} (unseen domain)...")
            results = self.evaluate_task_pass_at_k_distributed(generalization_domain)
            if self.is_main_process():
                final_generalization_full = results
                final_generalization = results.get('pass_at_k_mean', 0)
                self.metrics.log_generalization(f"final_{generalization_domain}", results)

        self.accelerator.wait_for_everyone()

        # ========== 3. 计算最终指标并输出 ==========
        if self.is_main_process():
            def get_baseline_value(data, domain):
                val = data.get(domain, 0)
                if isinstance(val, dict):
                    return val.get('pass_at_k_mean', val.get('pass_at_1_mean', 0))
                return val

            print(f"\n{'='*60}")
            print(f"[Final CL Metrics Summary]")
            print(f"{'='*60}")

            # Final BWT（最终模型 vs 刚训练完时）
            print(f"\n[Final BWT] After all training:")
            final_bwt_values = []
            for task_idx, domain in enumerate(train_domains[:-1]):
                final_score = final_performance.get(domain, 0)
                if task_idx in performance_matrix and domain in performance_matrix[task_idx]:
                    after_training_score = performance_matrix[task_idx][domain]
                else:
                    after_training_score = final_score
                bwt_j = final_score - after_training_score
                final_bwt_values.append(bwt_j)
                print(f"  BWT_final_{task_idx} ({domain}): {bwt_j:+.3f} = {final_score:.3f} - {after_training_score:.3f}")
            avg_final_bwt = sum(final_bwt_values) / len(final_bwt_values) if final_bwt_values else 0
            print(f"  Average Final BWT: {avg_final_bwt:+.3f}")

            # Final AP
            final_ap = sum(final_performance.values()) / len(final_performance) if final_performance else 0
            print(f"\n[Final AP] Final Average Performance: {final_ap:.3f}")

            # Generalization
            gen_improvement = 0
            if generalization_domain:
                zs_gen = get_baseline_value(zero_shot_performance, generalization_domain)
                gen_improvement = final_generalization - zs_gen
                print(f"\n[Generalization] Domain: {generalization_domain}")
                print(f"  Zero-shot: {zs_gen:.3f}")
                print(f"  After training: {final_generalization:.3f}")
                print(f"  Improvement: {gen_improvement:+.3f}")

            # 汇总表格
            print(f"\n{'='*80}")
            print(f"[Summary Table]")
            print(f"{'='*80}")
            print(f"{'Domain':<12} {'Zero-shot':<10} {'Single':<10} {'Final':<10} {'FWT':<10} {'Final BWT':<10}")
            print(f"{'-'*80}")
            for task_idx, domain in enumerate(train_domains):
                zs = get_baseline_value(zero_shot_performance, domain)
                st = get_baseline_value(single_task_performance, domain)
                if st == 0:
                    st = zs
                fn = final_performance.get(domain, 0)
                fwt = fwt_values[task_idx] if task_idx < len(fwt_values) else 0
                bwt = final_bwt_values[task_idx] if task_idx < len(final_bwt_values) else 0
                print(f"{domain:<12} {zs:<10.3f} {st:<10.3f} {fn:<10.3f} {fwt:<+10.3f} {bwt:<+10.3f}")
            if generalization_domain:
                zs = get_baseline_value(zero_shot_performance, generalization_domain)
                fn = final_generalization
                print(f"{generalization_domain:<12} {zs:<10.3f} {'N/A':<10} {fn:<10.3f} {'N/A':<10} {'N/A':<10}")
            print(f"{'-'*80}")
            print(f"{'Average':<12} {'':<10} {'':<10} {final_ap:<10.3f} {avg_fwt:<+10.3f} {avg_final_bwt:<+10.3f}")
            print(f"{'Avg BWT(M)':<12} {avg_bwt:<+10.3f}  (from intermediate BWT matrix)")
            print(f"{'='*80}")

            # 保存结果
            output_path = Path(self.config.log_dir) / "phase3_final_eval.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_results = {
                # 最终评估结果
                "final_performance": final_performance,
                "final_performance_full": final_performance_full,
                "final_generalization": final_generalization,
                "final_generalization_full": final_generalization_full,
                # 中间指标（从 phase2 计算）
                "ap_list": ap_list,
                "fwt_values": fwt_values,
                "bwt_matrix": {str(k): v for k, v in bwt_matrix.items()},
                # 汇总指标
                "avg_fwt": avg_fwt,
                "avg_bwt_matrix": avg_bwt,
                "avg_final_bwt": avg_final_bwt,
                "final_ap": final_ap,
                # baseline
                "zero_shot_performance": zero_shot_performance,
                "single_task_performance": single_task_performance,
                # phase2 原始数据
                "performance_matrix": {str(k): v for k, v in performance_matrix.items()},
            }
            if generalization_domain:
                final_results["generalization_improvement"] = gen_improvement

            with open(output_path, "w") as f:
                json.dump(final_results, f, indent=2)

            print(f"\nResults saved to: {output_path}")

            self.metrics.save()

    def cleanup(self):
        try:
            self.accelerator.wait_for_everyone()
        finally:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

    def _offload_model_to_cpu(self):
        """Move DeepSpeed model parameters to CPU to free VRAM for vLLM."""
        try:
            model = self.policy.model
            # DeepSpeed ZeRO-2: use engine's own offload if available
            if hasattr(model, "module"):
                inner = model.module
            else:
                inner = model
            for p in inner.parameters():
                p.data = p.data.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[_offload_model_to_cpu] warning: {e}", flush=True)

    def _reload_model_to_gpu(self):
        """Move DeepSpeed model parameters back to GPU after vLLM is done."""
        try:
            model = self.policy.model
            if hasattr(model, "module"):
                inner = model.module
            else:
                inner = model
            for p in inner.parameters():
                p.data = p.data.to(self.device)
        except Exception as e:
            print(f"[_reload_model_to_gpu] warning: {e}", flush=True)

    # ==================================================================
    # API-Bank single-turn training pipeline
    # ==================================================================

    def train_apibank(self, levels: list = None, max_samples: int = None):
        """
        API-Bank training loop — ToolRL style with vLLM generation.

        Each step:
          1. Load vLLM engine (tensor_parallel across all GPUs)
          2. Generate B*G responses in one batched call
          3. Unload vLLM, free VRAM
          4. Compute rewards, GRPO advantage, PPO clip loss (DeepSpeed)
        """
        from .apibank_data_loader import APIBankDataLoader

        loader = APIBankDataLoader(
            levels=levels or [1, 2, 3],
            max_samples=max_samples,
        )

        B = getattr(self.config, "batch_size_per_gpu", 64)
        G = self.config.num_samples_per_prompt

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[API-Bank ToolRL+vLLM] Training started")
            print(f"  {loader.stats()}")
            print(f"  steps={self.config.num_steps_per_task}  B={B}  G={G}  B*G={B*G}")
            print(f"  lr={self.config.learning_rate}  clip={self.config.clip_range}  kl={self.config.kl_coef}")
            print(f"{'='*60}\n")

        train_samples = loader.get_train_samples()
        eval_samples  = loader.get_eval_samples()

        epoch_pool: list = []
        epoch_num = 0

        def next_batch() -> list:
            nonlocal epoch_pool, epoch_num
            batch = []
            for _ in range(B):
                if not epoch_pool:
                    epoch_pool = train_samples.copy()
                    random.shuffle(epoch_pool)
                    epoch_num += 1
                    if self.is_main_process():
                        print(f"  [Epoch {epoch_num}] {len(epoch_pool)} samples")
                batch.append(epoch_pool.pop(0))
            return batch

        early_stop_flag = False
        recent_rewards: list = []
        patience_counter = 0
        patience  = getattr(self.config, "early_stopping_patience", 10)
        threshold = getattr(self.config, "early_stopping_threshold", 3.5)

        # vLLM config
        vllm_tp   = getattr(self.config, "vllm_tensor_parallel_size", self.config.world_size)
        vllm_mem  = getattr(self.config, "vllm_gpu_memory_utilization", 0.85)
        vllm_eager = getattr(self.config, "vllm_enforce_eager", False)
        max_model_len = (getattr(self.config, "max_prompt_length", 2048)
                         + self.config.max_new_tokens)

        for step in range(self.config.num_steps_per_task):
            batch   = next_batch()
            metrics = self.train_step_apibank_toolrl(
                batch, step,
                vllm_tp=vllm_tp,
                vllm_mem=vllm_mem,
                vllm_eager=vllm_eager,
                max_model_len=max_model_len,
            )

            if self.is_main_process() and metrics:
                self.metrics.log_step(0, step, metrics)

                if patience > 0 and metrics.get("reward_mean") is not None:
                    recent_rewards.append(metrics["reward_mean"])
                    if len(recent_rewards) > 5:
                        recent_rewards.pop(0)
                    if len(recent_rewards) >= 5:
                        avg = sum(recent_rewards) / len(recent_rewards)
                        if avg >= threshold:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"\n[Early Stop] reward={avg:.3f} >= {threshold}")
                                early_stop_flag = True
                        else:
                            patience_counter = 0

            if self.config.world_size > 1:
                import torch.distributed as _dist
                flag_t = torch.tensor([1 if early_stop_flag else 0],
                                      dtype=torch.long, device=self.device)
                _dist.broadcast(flag_t, src=0)
                early_stop_flag = flag_t.item() == 1

            if early_stop_flag:
                break

            ckpt_interval = getattr(self.config, "checkpoint_interval", 100)
            if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                self._save_step_checkpoint(0, "apibank", step)

            skip_eval = getattr(self.config, "skip_intermediate_eval", False)
            if not skip_eval and step > 0 and step % self.config.eval_interval == 0:
                # All ranks must call evaluate_apibank together (vllm_generate uses barrier)
                eval_metrics = self.evaluate_apibank(
                    eval_samples,
                    vllm_tp=vllm_tp, vllm_mem=vllm_mem,
                    vllm_eager=vllm_eager, max_model_len=max_model_len,
                )
                if self.is_main_process():
                    self.metrics.log_eval(0, step, eval_metrics)
                    print(f"[Eval step {step}] "
                          f"exact={eval_metrics.get('exact_match_rate', 0):.3f} "
                          f"reward={eval_metrics.get('reward_mean', 0):.3f}")

        if self.is_main_process():
            final_step = step if self.config.num_steps_per_task > 0 else 0
            print(f"\n[API-Bank] Training complete at step {final_step}")
            self.metrics.save()
            self.metrics.close()

    def train_step_apibank_toolrl(
        self,
        samples: list,
        step: int = 0,
        vllm_tp: int = 8,
        vllm_mem: float = 0.85,
        vllm_eager: bool = False,
        max_model_len: int = 3072,
    ) -> Optional[dict]:
        """
        ToolRL-style GRPO step for API-Bank:

        Phase A  — vLLM generates B*G responses (all GPUs, tensor parallel)
        Phase B  — compute rewards vs gold_tool_calls
        Phase C  — GRPO advantage (group by prompt uid, fallback to global)
        Phase D  — PPO clip loss (DeepSpeed, all GPUs parallel backward)
        """
        import torch.distributed as _dist
        from .vllm_generator import vllm_generate

        rank       = self.config.local_rank
        world_size = self.config.world_size
        G          = self.config.num_samples_per_prompt
        B          = len(samples)

        if self.is_main_process():
            print(f"\n[Step {step}] B={B} G={G} -> {B*G} trajs", flush=True)

        # ----------------------------------------------------------------
        # Phase A: vLLM generation (persistent per-rank engine, tp=1)
        # The engine offloads DS model to CPU, syncs weights into vLLM,
        # generates, then reloads DS model back to GPU.
        # ----------------------------------------------------------------
        all_trajs, all_olps, all_uids = vllm_generate(
            samples=samples,
            model_path=self.config.model_name_or_path,
            G=G,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=vllm_tp,
            gpu_memory_utilization=vllm_mem,
            dtype=self.config.model_dtype,
            enforce_eager=vllm_eager,
            device=self.device,
            world_size=world_size,
            ds_model=self.policy.model,
            ref_model=self.policy.reference_model,
        )

        if not all_trajs:
            if self.is_main_process():
                print(f"  No trajectories, skipping step {step}")
            return None

        if self.is_main_process():
            print(f"  Generated {len(all_trajs)} trajectories", flush=True)

        # ----------------------------------------------------------------
        # Phase B: rewards
        # ----------------------------------------------------------------
        from .apibank_reward import compute_reward as apibank_reward

        all_rewards:      List[float] = []
        all_reward_infos: list        = []

        # All ranks compute rewards (same data, deterministic)
        # Use reward_text (think-stripped) for reward; output_text stays raw for PPO
        for traj in all_trajs:
            text_for_reward = traj.reward_text if traj.reward_text else traj.output_text
            info = apibank_reward(text_for_reward, traj.gold_tool_calls)
            all_rewards.append(info.reward)
            all_reward_infos.append(info)

        if self.is_main_process():
            exact = sum(1 for i in all_reward_infos if i.exact_match)
            print(f"  rewards: mean={np.mean(all_rewards):.3f} "
                  f"std={np.std(all_rewards):.3f} "
                  f"min={np.min(all_rewards):.3f} max={np.max(all_rewards):.3f} "
                  f"exact={exact}/{len(all_trajs)}", flush=True)
            # Debug: print first trajectory output
            if all_trajs:
                t = all_trajs[0]
                print(f"  [debug] reward_text=\n{t.reward_text or t.output_text}", flush=True)
                print(f"  [debug] gold=\n{t.gold_tool_calls}", flush=True)
                print(f"  [debug] reward_info: parse_ok={all_reward_infos[0].parse_ok} "
                      f"pred={all_reward_infos[0].pred_tool_calls}", flush=True)

        # ----------------------------------------------------------------
        # Phase C: GRPO advantage
        # ----------------------------------------------------------------
        advantages = self._compute_grpo_advantage(all_rewards, all_uids)

        if self.is_main_process():
            print(f"  adv: mean={advantages.mean():.3f} std={advantages.std():.3f}",
                  flush=True)

        # ----------------------------------------------------------------
        # Phase D: PPO clip loss (DeepSpeed, all ranks parallel)
        # ----------------------------------------------------------------
        ppo_metrics = self.policy.compute_grpo_loss_toolrl(
            trajectories=all_trajs,
            advantages=advantages,
            old_log_probs=all_olps,
            accelerator=self.accelerator,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        format_scores      = [i.format_score  for i in all_reward_infos]
        correctness_scores = [i.correctness   for i in all_reward_infos]

        step_metrics = {
            "loss":               ppo_metrics.get("pg_loss", 0.0),
            "pg_loss":            ppo_metrics.get("pg_loss", 0.0),
            "pg_clipfrac":        ppo_metrics.get("pg_clipfrac", 0.0),
            "kl":                 ppo_metrics.get("kl", 0.0),
            "entropy":            ppo_metrics.get("entropy", 0.0),
            "grad_norm":          ppo_metrics.get("grad_norm", 0.0),
            "reward_mean":        float(np.mean(all_rewards)),
            "reward_max":         float(np.max(all_rewards)),
            "reward_min":         float(np.min(all_rewards)),
            "reward_std":         float(np.std(all_rewards)),
            "format_reward_mean": float(np.mean(format_scores)),
            "correctness_reward_mean": float(np.mean(correctness_scores)),
            "exact_match":        float(sum(1 for i in all_reward_infos if i.exact_match)
                                         / max(len(all_reward_infos), 1)),
            "parse_ok":           float(sum(1 for i in all_reward_infos if i.parse_ok)
                                         / max(len(all_reward_infos), 1)),
            "num_trajs":          len(all_trajs),
        }

        if self.is_main_process():
            print(f"[Step {step}] "
                  f"pg_loss={step_metrics['pg_loss']:.4f} "
                  f"clip={step_metrics['pg_clipfrac']:.3f} "
                  f"kl={step_metrics['kl']:.4f} "
                  f"reward={step_metrics['reward_mean']:.3f} "
                  f"fmt={step_metrics['format_reward_mean']:.3f} "
                  f"correct={step_metrics['correctness_reward_mean']:.3f} "
                  f"exact={step_metrics['exact_match']:.3f}", flush=True)

        return step_metrics

    def evaluate_apibank(
        self,
        eval_samples=None,
        num_eval: int = None,
        vllm_tp: int = 8,
        vllm_mem: float = 0.4,
        vllm_eager: bool = False,
        max_model_len: int = 3072,
    ) -> dict:
        """
        Evaluate on API-Bank eval split using the persistent vLLM engine (greedy).

        All ranks call this together — vllm_generate handles weight offload/sync
        and all_gather internally.
        """
        from .apibank_reward import compute_reward as apibank_reward
        from .vllm_generator import vllm_generate

        if eval_samples is None:
            from .apibank_data_loader import APIBankDataLoader
            loader = APIBankDataLoader()
            eval_samples = loader.get_eval_samples()

        if num_eval is not None:
            eval_samples = eval_samples[:num_eval]

        if self.is_main_process():
            print(f"[Eval] {len(eval_samples)} samples with vLLM...", flush=True)

        trajs, _, _ = vllm_generate(
            samples=eval_samples,
            model_path=self.config.model_name_or_path,
            G=1,
            temperature=0.0,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=vllm_tp,
            gpu_memory_utilization=vllm_mem,
            dtype=self.config.model_dtype,
            enforce_eager=vllm_eager,
            device=self.device,
            world_size=self.config.world_size,
            ds_model=self.policy.model,
        )

        rewards, exact_matches, parse_oks = [], [], []
        for traj in trajs:
            text_for_reward = traj.reward_text if traj.reward_text else traj.output_text
            info = apibank_reward(text_for_reward, traj.gold_tool_calls)
            rewards.append(info.reward)
            exact_matches.append(int(info.exact_match))
            parse_oks.append(int(info.parse_ok))

        result = {
            "reward_mean":      float(np.mean(rewards)) if rewards else 0.0,
            "reward_std":       float(np.std(rewards))  if rewards else 0.0,
            "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
            "parse_ok_rate":    float(np.mean(parse_oks))     if parse_oks     else 0.0,
            "num_evaluated":    len(eval_samples),
        }
        if self.is_main_process():
            print(f"[Eval] reward={result['reward_mean']:.3f} "
                  f"exact={result['exact_match_rate']:.3f} "
                  f"parse_ok={result['parse_ok_rate']:.3f}", flush=True)
        return result

    # =========================================================================
    # rlla_by_domain training (ToolRL-aligned, supports airline/retail/api_bank/bamboogle)
    # =========================================================================

    def train_step_rlla_toolrl(
        self,
        samples: list,          # List[RllaSample]
        step: int = 0,
        vllm_tp: int = 8,
        vllm_mem: float = 0.85,
        vllm_eager: bool = False,
        max_model_len: int = 3072,
    ) -> Optional[dict]:
        """
        ToolRL-style GRPO step for rlla_by_domain data.

        Identical pipeline to train_step_apibank_toolrl, but:
          - reward uses rlla_reward.compute_score(output, ground_truth)
          - supports both tool_call and response task types
        """
        import torch.distributed as _dist
        from .vllm_generator import vllm_generate
        from .rlla_reward import compute_score as rlla_reward

        rank       = self.config.local_rank
        world_size = self.config.world_size
        G          = self.config.num_samples_per_prompt
        B          = len(samples)

        if self.is_main_process():
            print(f"\n[Step {step}] B={B} G={G} -> {B*G} trajs", flush=True)

        # Phase A: vLLM generation
        all_trajs, all_olps, all_uids = vllm_generate(
            samples=samples,
            model_path=self.config.model_name_or_path,
            G=G,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=vllm_tp,
            gpu_memory_utilization=vllm_mem,
            dtype=self.config.model_dtype,
            enforce_eager=vllm_eager,
            device=self.device,
            world_size=world_size,
            ds_model=self.policy.model,
            ref_model=self.policy.reference_model,
        )

        if not all_trajs:
            if self.is_main_process():
                print(f"  No trajectories, skipping step {step}")
            return None

        if self.is_main_process():
            print(f"  Generated {len(all_trajs)} trajectories", flush=True)

        # Phase B: rewards (ToolRL compute_rewards)
        # token_level_rewards = token_level_scores - kl_coef * (old_lp - ref_lp)
        # score is placed at the last response token, 0 elsewhere (ToolRL convention).
        # We store per-traj token_level_reward tensors for advantage computation.
        all_reward_infos: list = []
        kl_coef = getattr(self.config, "kl_coef", 0.001)

        seq_rewards:          List[float]         = []   # for logging
        traj_kl_means:        List[float]         = []   # for logging
        token_level_rewards:  List[torch.Tensor]  = []   # (R,) per traj

        for traj, old_lp in zip(all_trajs, all_olps):
            output = traj.reward_text if traj.reward_text else traj.output_text
            ground_truth = traj.gold_tool_calls
            if isinstance(ground_truth, list):
                import json as _json
                ground_truth = "<tool_call>\n" + "\n".join(
                    _json.dumps(tc) for tc in ground_truth) + "\n</tool_call>"
            info = rlla_reward(output, ground_truth, step=step)
            all_reward_infos.append(info)

            R = len(old_lp) if old_lp is not None and len(old_lp) > 0 else 1

            # token_level_scores: reward at last token, 0 elsewhere (ToolRL style)
            token_scores = torch.zeros(R, dtype=torch.float32)
            token_scores[-1] = info.reward

            # KL penalty: low_var_kl(old_lp, ref_lp) = exp(ref-old) - (ref-old) - 1
            # Mirrors ToolRL core_algos.kl_penalty(..., kl_penalty='low_var_kl')
            if traj.ref_log_probs is not None and old_lp is not None and len(old_lp) > 0:
                min_len = min(R, len(traj.ref_log_probs))
                kl_tok  = torch.zeros(R, dtype=torch.float32)
                kl_diff = traj.ref_log_probs[:min_len].float() - old_lp[:min_len].float()
                kl_tok[:min_len] = torch.clamp(torch.exp(kl_diff) - kl_diff - 1.0, -10.0, 10.0)
                kl_mean = kl_tok[:min_len].mean().item()
            else:
                kl_tok  = torch.zeros(R, dtype=torch.float32)
                kl_mean = 0.0

            # token_level_reward = token_level_scores - kl_coef * kl  (ToolRL)
            tok_reward = token_scores - kl_coef * kl_tok

            token_level_rewards.append(tok_reward)
            traj_kl_means.append(kl_mean)

            # sequence-level score for logging = sum of token_level_reward
            seq_rewards.append(tok_reward.sum().item())

        if self.is_main_process():
            exact    = sum(1 for i in all_reward_infos if i.exact_match)
            fmt_mean = float(np.mean([i.format_score      for i in all_reward_infos]))
            cor_mean = float(np.mean([i.correctness_score for i in all_reward_infos]))
            kl_mean  = float(np.mean(traj_kl_means))
            print(f"  rewards: mean={np.mean(seq_rewards):.3f} "
                  f"std={np.std(seq_rewards):.3f} "
                  f"min={np.min(seq_rewards):.3f} max={np.max(seq_rewards):.3f} "
                  f"exact={exact}/{len(all_trajs)} "
                  f"fmt={fmt_mean:.3f} correct={cor_mean:.3f} kl={kl_mean:.4f}", flush=True)
            if all_trajs:
                debug_idx = next(
                    (i for i, info in enumerate(all_reward_infos) if info.parse_ok), 0)
                t = all_trajs[debug_idx]
                print(f"  [debug] output=\n{t.reward_text or t.output_text}", flush=True)
                print(f"  [debug] gt=\n{t.gold_tool_calls}", flush=True)
                print(f"  [debug] reward_info: parse_ok={all_reward_infos[debug_idx].parse_ok} "
                      f"task_type={all_reward_infos[debug_idx].task_type} "
                      f"reward={seq_rewards[debug_idx]:.3f}", flush=True)

        # Phase C: GRPO advantage (ToolRL compute_grpo_outcome_advantage)
        # Extract scalar score per traj = sum(token_level_rewards) (ToolRL convention:
        # reward is at last token, so sum == that scalar reward minus KL penalty sum).
        traj_scores = [tok_r.sum().item() for tok_r in token_level_rewards]
        advantages = self._compute_grpo_advantage(traj_scores, all_uids)

        if self.is_main_process():
            print(f"  adv: mean={advantages.mean():.3f} std={advantages.std():.3f}",
                  flush=True)

        # Phase D: PPO clip loss (token-level, ToolRL compute_policy_loss)
        self.policy.model.train()
        ppo_metrics = self.policy.compute_grpo_loss_toolrl(
            trajectories=all_trajs,
            advantages=advantages,
            old_log_probs=all_olps,
            token_level_rewards=token_level_rewards,
            accelerator=self.accelerator,
        )

        fmt_scores = [i.format_score      for i in all_reward_infos]
        cor_scores = [i.correctness_score for i in all_reward_infos]

        step_metrics = {
            "loss":               ppo_metrics.get("pg_loss", 0.0),
            "pg_loss":            ppo_metrics.get("pg_loss", 0.0),
            "pg_clipfrac":        ppo_metrics.get("pg_clipfrac", 0.0),
            "kl":                 ppo_metrics.get("kl", 0.0),
            "entropy":            ppo_metrics.get("entropy", 0.0),
            "grad_norm":          ppo_metrics.get("grad_norm", 0.0),
            "reward_mean":        float(np.mean(seq_rewards)),
            "reward_max":         float(np.max(seq_rewards)),
            "reward_min":         float(np.min(seq_rewards)),
            "reward_std":         float(np.std(seq_rewards)),
            "format_reward_mean": float(np.mean(fmt_scores)),
            "correctness_reward_mean": float(np.mean(cor_scores)),
            "exact_match":        float(sum(1 for i in all_reward_infos if i.exact_match)
                                         / max(len(all_reward_infos), 1)),
            "parse_ok":           float(sum(1 for i in all_reward_infos if i.parse_ok)
                                         / max(len(all_reward_infos), 1)),
            "num_trajs":          len(all_trajs),
        }

        if self.is_main_process():
            print(f"[Step {step}] "
                  f"pg_loss={step_metrics['pg_loss']:.4f} "
                  f"clip={step_metrics['pg_clipfrac']:.3f} "
                  f"kl={step_metrics['kl']:.4f} "
                  f"reward={step_metrics['reward_mean']:.3f} "
                  f"fmt={step_metrics['format_reward_mean']:.3f} "
                  f"correct={step_metrics['correctness_reward_mean']:.3f} "
                  f"exact={step_metrics['exact_match']:.3f}", flush=True)

        return step_metrics

    def train_rlla(
        self,
        data_dir: str,
        max_samples: int = None,
    ):
        """
        ToolRL-style GRPO training on rlla_by_domain data.

        Args:
            data_dir:    path to one domain dir, e.g. "data/rlla_by_domain/airline"
            max_samples: cap training samples (debug)
        """
        from .rlla_data_loader import RllaDataLoader

        loader = RllaDataLoader(data_dir=data_dir, max_samples=max_samples)

        B = getattr(self.config, "batch_size_per_gpu", 64)
        G = self.config.num_samples_per_prompt

        if self.is_main_process():
            print(f"\n{'='*60}")
            print(f"[rlla ToolRL+vLLM] Training started")
            print(f"  {loader.stats()}")
            print(f"  steps={self.config.num_steps_per_task}  B={B}  G={G}  B*G={B*G}")
            print(f"  lr={self.config.learning_rate}  clip={self.config.clip_range}  kl={self.config.kl_coef}")
            print(f"{'='*60}\n")

        train_samples = loader.get_train_samples()
        eval_samples  = loader.get_eval_samples()

        epoch_pool: list = []
        epoch_num = 0

        def next_batch() -> list:
            nonlocal epoch_pool, epoch_num
            batch = []
            for _ in range(B):
                if not epoch_pool:
                    epoch_pool = train_samples.copy()
                    random.shuffle(epoch_pool)
                    epoch_num += 1
                    if self.is_main_process():
                        print(f"  [Epoch {epoch_num}] {len(epoch_pool)} samples")
                batch.append(epoch_pool.pop(0))
            return batch

        early_stop_flag = False
        recent_rewards: list = []
        patience_counter = 0
        patience  = getattr(self.config, "early_stopping_patience", 10)
        threshold = getattr(self.config, "early_stopping_threshold", 3.5)

        vllm_tp    = getattr(self.config, "vllm_tensor_parallel_size", self.config.world_size)
        vllm_mem   = getattr(self.config, "vllm_gpu_memory_utilization", 0.85)
        vllm_eager = getattr(self.config, "vllm_enforce_eager", False)
        max_model_len = (getattr(self.config, "max_prompt_length", 4096)
                         + self.config.max_new_tokens)

        for step in range(self.config.num_steps_per_task):
            batch   = next_batch()
            metrics = self.train_step_rlla_toolrl(
                batch, step,
                vllm_tp=vllm_tp,
                vllm_mem=vllm_mem,
                vllm_eager=vllm_eager,
                max_model_len=max_model_len,
            )

            if self.is_main_process() and metrics:
                self.metrics.log_step(0, step, metrics)

                if patience > 0 and metrics.get("reward_mean") is not None:
                    recent_rewards.append(metrics["reward_mean"])
                    if len(recent_rewards) > 5:
                        recent_rewards.pop(0)
                    if len(recent_rewards) >= 5:
                        avg = sum(recent_rewards) / len(recent_rewards)
                        if avg >= threshold:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"\n[Early Stop] reward={avg:.3f} >= {threshold}")
                                early_stop_flag = True
                        else:
                            patience_counter = 0

            if self.config.world_size > 1:
                import torch.distributed as _dist
                flag_t = torch.tensor([1 if early_stop_flag else 0],
                                      dtype=torch.long, device=self.device)
                _dist.broadcast(flag_t, src=0)
                early_stop_flag = flag_t.item() == 1

            if early_stop_flag:
                break

            ckpt_interval = getattr(self.config, "checkpoint_interval", 100)
            if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                domain_name = Path(data_dir).name
                self._save_step_checkpoint(0, domain_name, step)

            skip_eval = getattr(self.config, "skip_intermediate_eval", False)
            if not skip_eval and step > 0 and step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate_rlla(
                    eval_samples,
                    vllm_tp=vllm_tp, vllm_mem=vllm_mem,
                    vllm_eager=vllm_eager, max_model_len=max_model_len,
                )
                if self.is_main_process():
                    self.metrics.log_eval(0, step, eval_metrics)
                    print(f"[Eval step {step}] "
                          f"exact={eval_metrics.get('exact_match_rate', 0):.3f} "
                          f"reward={eval_metrics.get('reward_mean', 0):.3f}")

        # 保存最终模型，供 sequential 训练传递给下一个 domain
        final_model_dir = Path(self.config.log_dir) / "model"
        if self.is_main_process():
            print(f"\n[rlla] Saving final model to {final_model_dir}...")
        unwrapped = self.accelerator.unwrap_model(self.policy.model)
        self.policy.save_checkpoint(str(final_model_dir), model_override=unwrapped)
        self.accelerator.wait_for_everyone()

        if self.is_main_process():
            final_step = step if self.config.num_steps_per_task > 0 else 0
            print(f"\n[rlla] Training complete at step {final_step}")
            print(f"[rlla] Final model saved to {final_model_dir}")
            self.metrics.save()
            self.metrics.close()

    def evaluate_rlla(
        self,
        eval_samples: list = None,
        num_eval: int = None,
        vllm_tp: int = 8,
        vllm_mem: float = 0.4,
        vllm_eager: bool = False,
        max_model_len: int = 3072,
    ) -> dict:
        from .rlla_reward import compute_score as rlla_reward
        from .vllm_generator import vllm_generate

        if num_eval is not None:
            eval_samples = eval_samples[:num_eval]

        if self.is_main_process():
            print(f"[Eval] {len(eval_samples)} samples with vLLM...", flush=True)

        trajs, _, _ = vllm_generate(
            samples=eval_samples,
            model_path=self.config.model_name_or_path,
            G=1,
            temperature=0.0,
            max_new_tokens=self.config.max_new_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=vllm_tp,
            gpu_memory_utilization=vllm_mem,
            dtype=self.config.model_dtype,
            enforce_eager=vllm_eager,
            device=self.device,
            world_size=self.config.world_size,
            ds_model=self.policy.model,
        )

        rewards, exact_matches, parse_oks = [], [], []
        for traj in trajs:
            output = traj.reward_text if traj.reward_text else traj.output_text
            ground_truth = traj.gold_tool_calls
            if isinstance(ground_truth, list):
                import json as _json
                ground_truth = "<tool_call>\n" + "\n".join(
                    _json.dumps(tc) for tc in ground_truth) + "\n</tool_call>"
            info = rlla_reward(output, ground_truth)
            rewards.append(info.reward)
            exact_matches.append(int(info.exact_match))
            parse_oks.append(int(info.parse_ok))

        result = {
            "reward_mean":      float(np.mean(rewards))        if rewards else 0.0,
            "reward_std":       float(np.std(rewards))         if rewards else 0.0,
            "exact_match_rate": float(np.mean(exact_matches))  if exact_matches else 0.0,
            "parse_ok_rate":    float(np.mean(parse_oks))      if parse_oks else 0.0,
            "num_evaluated":    len(eval_samples),
        }
        if self.is_main_process():
            print(f"[Eval] reward={result['reward_mean']:.3f} "
                  f"exact={result['exact_match_rate']:.3f} "
                  f"parse_ok={result['parse_ok_rate']:.3f}", flush=True)
        return result

