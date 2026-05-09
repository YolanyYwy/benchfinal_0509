"""
Data loader for rlla_by_domain format (output of convert_tau2_to_rlla.py).

Each record:
{
  "data_source": "rlla",
  "prompt": [{"role": "system", ...}, {"role": "user", ...}],
  "ability": "tool_use",
  "reward_model": {"style": "rule", "ground_truth": "<think>...</think>\n<tool_call>..."},
  "extra_info": {"index": 0, "split": "train", "domain": "airline", "task_id": "2", "step": 0}
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RllaSample:
    id:           str
    domain:       str
    task_id:      str
    step:         int
    messages:     List[dict]   # [system, user] — ready for apply_chat_template
    ground_truth: str          # expected model output for reward computation

    def to_prompt_messages(self) -> List[dict]:
        return self.messages


class RllaDataLoader:
    """
    Loads one domain's rlla_by_domain data.

    Usage:
        loader = RllaDataLoader("data/rlla_by_domain/airline")
        for sample in loader.get_train_samples():
            ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seed     = seed

        self.train_samples: List[RllaSample] = []
        self.eval_samples:  List[RllaSample] = []

        self._load(max_samples)

    def _load(self, max_samples: Optional[int]):
        train_path = self.data_dir / "train.json"
        test_path  = self.data_dir / "test.json"

        self.train_samples = self._read(train_path, max_samples)
        self.eval_samples  = self._read(test_path,  None)

        print(f"[RllaDataLoader] {self.data_dir.name}: "
              f"{len(self.train_samples)} train / {len(self.eval_samples)} eval")

    def _read(self, path: Path, max_samples: Optional[int]) -> List[RllaSample]:
        if not path.exists():
            return []
        records = json.load(open(path, encoding="utf-8"))
        if max_samples is not None:
            rng = random.Random(self.seed)
            rng.shuffle(records)
            records = records[:max_samples]
        samples = []
        for rec in records:
            ei = rec.get("extra_info", {})
            samples.append(RllaSample(
                id=f"{ei.get('domain','rlla')}-{ei.get('task_id','0')}-{ei.get('step',0)}",
                domain=ei.get("domain", "rlla"),
                task_id=str(ei.get("task_id", "")),
                step=int(ei.get("step", 0)),
                messages=rec["prompt"],
                ground_truth=rec["reward_model"]["ground_truth"],
            ))
        return samples

    def get_train_samples(self) -> List[RllaSample]:
        return self.train_samples

    def get_eval_samples(self) -> List[RllaSample]:
        return self.eval_samples

    def stats(self) -> str:
        return (f"{self.data_dir.name}: "
                f"{len(self.train_samples)} train / {len(self.eval_samples)} eval")
