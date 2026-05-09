"""API-Bank dataset loader for single-turn tool-use RL training."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class APIBankSample:
    """A single API-Bank training sample."""
    id: str                          # e.g. "level-1-42"
    system: str                      # system prompt with available tools
    user: str                        # user query (may include dialogue history)
    gold_tool_calls: List[dict]      # [{"name": "...", "parameters": {...}}, ...]
    level: int                       # 1, 2, or 3
    source_file: str = ""            # original file name for debugging

    # Replacement for the verbose Output Format block in the original system prompt.
    # We want the model to output ONLY a <tool_call> block, no <think> or <response>.
    _FORMAT_OVERRIDE = (
        "\n\n**Output Format**\n"
        "```plaintext\n"
        "<tool_call>\n"
        "{\"name\": \"Tool name\", \"parameters\": {\"param\": \"value\"}}\n"
        "</tool_call>\n"
        "```\n"
        "Output ONLY the <tool_call> block. Do NOT include <think>, <response>, "
        "or any other text outside the block."
    )

    def to_prompt_messages(self) -> List[dict]:
        """Return [system, user] message list for apply_chat_template.

        Replaces the original Output Format section so the model outputs only
        a bare <tool_call>...</tool_call> block with no <think> wrapper.
        """
        # Strip the original Output Format / Important Notes sections and append ours
        sys = self.system
        for marker in ["**Output Format**", "**Important Notes**", "**Steps for Each Turn**"]:
            idx = sys.find(marker)
            if idx != -1:
                sys = sys[:idx].rstrip()
                break
        sys = sys + self._FORMAT_OVERRIDE
        return [
            {"role": "system", "content": sys},
            {"role": "user",   "content": self.user},
        ]


class APIBankDataLoader:
    """
    Loads API-Bank data from level-{1,2,3}-api_processed.json files.

    Each JSON record has:
        system  : str   — role description + available tools
        user    : str   — user query (possibly multi-turn history condensed to one message)
        answer  : list  — gold tool calls, e.g. [{"name": "...", "parameters": {...}}]
        other   : dict  — metadata (file, id)

    This loader flattens all levels into a single list and exposes
    train / eval splits.
    """

    DATA_DIR = Path(__file__).resolve().parents[3] / "ToolRL" / "benchmarks" / "API-Bank"

    def __init__(
        self,
        levels: List[int] = None,
        data_dir: Optional[Path] = None,
        train_ratio: float = 0.9,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            levels: which difficulty levels to load, default [1, 2, 3]
            data_dir: override default data directory
            train_ratio: fraction of data used for training
            seed: random seed for reproducible splits
            max_samples: cap total samples (useful for debugging)
        """
        self.levels = levels or [1, 2, 3]
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
        self.train_ratio = train_ratio
        self.seed = seed

        self.all_samples: List[APIBankSample] = []
        self.train_samples: List[APIBankSample] = []
        self.eval_samples: List[APIBankSample] = []

        self._load(max_samples)
        self._split()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, max_samples: Optional[int]):
        for level in self.levels:
            path = self.data_dir / f"level-{level}-api_processed.json"
            if not path.exists():
                raise FileNotFoundError(f"API-Bank file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)

            for rec in records:
                answer = rec.get("answer", [])
                # answer is already a list of dicts
                if isinstance(answer, str):
                    try:
                        answer = json.loads(answer)
                    except json.JSONDecodeError:
                        answer = []

                other = rec.get("other", {})
                sample_id = f"level-{level}-{other.get('id', len(self.all_samples))}"

                self.all_samples.append(APIBankSample(
                    id=sample_id,
                    system=rec["system"],
                    user=rec["user"],
                    gold_tool_calls=answer,
                    level=level,
                    source_file=other.get("file", ""),
                ))

            print(f"[APIBankDataLoader] Loaded {len(records)} samples from level-{level}")

        if max_samples is not None:
            rng = random.Random(self.seed)
            rng.shuffle(self.all_samples)
            self.all_samples = self.all_samples[:max_samples]

        print(f"[APIBankDataLoader] Total samples: {len(self.all_samples)}")

    def _split(self):
        rng = random.Random(self.seed)
        samples = self.all_samples.copy()
        rng.shuffle(samples)

        n_train = int(len(samples) * self.train_ratio)
        self.train_samples = samples[:n_train]
        self.eval_samples  = samples[n_train:]

        print(f"[APIBankDataLoader] Split: {len(self.train_samples)} train / {len(self.eval_samples)} eval")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_train_samples(self) -> List[APIBankSample]:
        return self.train_samples

    def get_eval_samples(self) -> List[APIBankSample]:
        return self.eval_samples

    def get_train_iterator(self, shuffle: bool = True) -> Iterator[APIBankSample]:
        samples = self.train_samples.copy()
        if shuffle:
            random.shuffle(samples)
        return iter(samples)

    def sample_train_batch(self, batch_size: int) -> List[APIBankSample]:
        if batch_size >= len(self.train_samples):
            return random.choices(self.train_samples, k=batch_size)
        return random.sample(self.train_samples, batch_size)

    def __len__(self) -> int:
        return len(self.all_samples)

    def stats(self) -> dict:
        by_level = {}
        for s in self.all_samples:
            by_level[s.level] = by_level.get(s.level, 0) + 1
        return {
            "total": len(self.all_samples),
            "train": len(self.train_samples),
            "eval":  len(self.eval_samples),
            "by_level": by_level,
        }
