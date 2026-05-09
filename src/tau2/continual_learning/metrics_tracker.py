"""Metrics tracking and logging for GRPO continual learning."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .config import GRPOConfig


class MetricsTracker:
    """Track and visualize training metrics for continual learning."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.metrics: dict[str, list[dict]] = defaultdict(list)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Progress bar
        self.pbar = None
        self.use_progress_bar = bool(config.use_progress_bar and TQDM_AVAILABLE)

        # W&B state
        self.use_wandb = config.wandb_project is not None
        self.wandb = None
        self._wandb_inited = False

    def init_wandb(self):
        """Initialize Weights & Biases (call on main process only)."""
        if not self.use_wandb or self._wandb_inited:
            return

        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            self.use_wandb = False
            return

        # mode 处理：None 不覆盖，让 wandb 自己走环境变量/默认逻辑
        if self.config.wandb_mode is not None:
            # wandb.init(mode=...) 支持 "online"/"offline"/"disabled"
            mode = self.config.wandb_mode
        else:
            mode = None

        init_kwargs = dict(
            project=self.config.wandb_project,
            config=self.config.to_dict(),
            name=self.config.wandb_run_name or f"grpo_cl_{self.config.cl_algorithm}",
        )
        if self.config.wandb_entity:
            init_kwargs["entity"] = self.config.wandb_entity
        if self.config.wandb_tags:
            init_kwargs["tags"] = self.config.wandb_tags
        if mode is not None:
            init_kwargs["mode"] = mode

        wandb.init(**init_kwargs)
        self.wandb = wandb
        self._wandb_inited = True

        # 预定义指标，让 wandb 自动生成图表
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        wandb.define_metric("transfer/*")
        wandb.define_metric("buffer/*")
        wandb.define_metric("generalization/*")

    def log_step(self, task_idx: int, step: int, metrics: dict[str, Any]):
        log_entry = {
            "task_idx": task_idx,
            "step": step,
            "global_step": self._compute_global_step(task_idx, step),
            **metrics
        }
        self.metrics[f"task_{task_idx}/train"].append(log_entry)

        # wandb
        if self.use_wandb and self._wandb_inited:
            self.wandb.log({"global_step": log_entry["global_step"], **{f"train/{k}": v for k, v in metrics.items()}})

        # tqdm / print
        if self.use_progress_bar and self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": f"{metrics.get('loss', 0):.4f}",
                "reward": f"{metrics.get('reward_mean', 0):.3f}",
                "kl": f"{metrics.get('kl_div', 0):.4f}",
            })
        elif step % self.config.log_interval == 0:
            metrics_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in list(metrics.items())[:5]
            ])
            print(f"[Task {task_idx}] Step {step:3d}/{self.config.num_steps_per_task} | {metrics_str}")

    def log_eval(self, task_idx: int, step: int, metrics: dict[str, Any]):
        log_entry = {
            "task_idx": task_idx,
            "step": step,
            "global_step": self._compute_global_step(task_idx, step),
            **metrics
        }
        self.metrics[f"task_{task_idx}/eval"].append(log_entry)

        if self.use_wandb and self._wandb_inited:
            self.wandb.log({"global_step": log_entry["global_step"], **{f"eval/{k}": v for k, v in metrics.items()}})

        print(f"[EVAL] Task {task_idx} Step {step} | "
              f"Reward: {metrics.get('reward_mean', 0):.3f}±{metrics.get('reward_std', 0):.3f} | "
              f"Pass: {metrics.get('pass_rate', 0):.1%} | "
              f"Tool Acc: {metrics.get('tool_accuracy', 0):.1%}")

    def log_transfer(self, current_task_idx: int, transfer_metrics: dict[str, Any]):
        log_entry = {"task_idx": current_task_idx, **transfer_metrics}
        self.metrics["transfer"].append(log_entry)

        if self.use_wandb and self._wandb_inited:
            self.wandb.log({f"transfer/{k}": v for k, v in transfer_metrics.items()}, step=current_task_idx)

        print(f"\n=== Transfer Metrics after Task {current_task_idx} ===")
        for k, v in transfer_metrics.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except Exception:
                print(f"  {k}: {v}")
        print()

    def log_buffer_stats(self, task_idx: int, buffer_stats: dict[str, Any]):
        log_entry = {"task_idx": task_idx, **buffer_stats}
        self.metrics["buffer"].append(log_entry)

        if self.use_wandb and self._wandb_inited:
            self.wandb.log({f"buffer/{k}": v for k, v in buffer_stats.items()}, step=task_idx)

    def log_generalization(self, phase: str, metrics: dict[str, Any]):
        """
        记录泛化性测试结果。

        Args:
            phase: "zero_shot" 或 "forward_transfer"
            metrics: 评估指标
        """
        log_entry = {"phase": phase, **metrics}
        self.metrics["generalization"].append(log_entry)

        if self.use_wandb and self._wandb_inited:
            self.wandb.log({f"generalization/{phase}/{k}": v for k, v in metrics.items()})

        print(f"\n=== Generalization ({phase}) ===")
        for k, v in metrics.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except Exception:
                print(f"  {k}: {v}")
        print()

    def _compute_global_step(self, task_idx: int, step: int) -> int:
        return task_idx * self.config.num_steps_per_task + step

    def start_task_progress(self, task_idx: int, domain: str):
        if self.use_progress_bar:
            self.pbar = tqdm(
                total=self.config.num_steps_per_task,
                desc=f"Task {task_idx} ({domain})",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
            )

    def close_task_progress(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    def save(self):
        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(dict(self.metrics), f, indent=2, ensure_ascii=False)
        print(f"Saved metrics to {metrics_file}")

    def close(self):
        self.close_task_progress()
        self.save()
        if self.use_wandb and self._wandb_inited:
            self.wandb.finish()
            self._wandb_inited = False
