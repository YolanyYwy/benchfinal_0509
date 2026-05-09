"""
Reward computation aligned with ToolRL rlla.py compute_score().

Supports both:
  - tool_call tasks: ground_truth contains <tool_call>...</tool_call>
  - response tasks:  ground_truth contains <response>...</response>

Total reward range: [-3.0, 4.0]
  format_score:      [0.0,  1.0]
  correctness_score: [-3.0, 3.0]
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

TOOL_MAX  =  3.0
TOOL_MIN  = -3.0
FMT_MAX   =  1.0
FMT_MIN   =  0.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RllaRewardInfo:
    reward:            float
    format_score:      float
    correctness_score: float
    parse_ok:          bool
    exact_match:       bool
    task_type:         str    # "tool_call" | "response" | "unknown"
    details:           dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — exact mirrors of ToolRL
# ---------------------------------------------------------------------------

def _match_score(list1: list, list2: list) -> float:
    """Frequency-aware similarity, order-insensitive. Mirrors ToolRL match_score()."""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    c1, c2 = Counter(list1), Counter(list2)
    intersection = sum(min(c1[k], c2[k]) for k in c1.keys() & c2.keys())
    union = len(list1) + len(list2) - intersection
    return intersection / union if union > 0 else 0.0


def _extract_model_output(solution_str: str) -> str:
    """
    Strip chat template wrapper from model output.
    Mirrors ToolRL's extraction logic for Qwen/Llama formats.
    """
    # Qwen3 thinking mode: <|im_start|>assistant\n<think>...</think>\n...
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant")[-1].strip()
    # Llama format
    if "[/INST]" in solution_str:
        solution_str = solution_str.split("[/INST]")[-1].strip()
    return solution_str


def _parse_tool_calls(text: str) -> Tuple[List[dict], bool]:
    """
    Extract tool calls from <tool_call>...</tool_call> block.
    Returns (tool_calls, parse_ok).
    Mirrors ToolRL's split-on-tag + per-line JSON approach.
    """
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return [], False

    block = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    if not block:
        return [], False

    tool_calls: List[dict] = []

    # Strategy 1: entire block is one JSON object (multi-line)
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            tool_calls = [obj]
        elif isinstance(obj, list):
            tool_calls = [x for x in obj if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Strategy 2: one JSON per line (ToolRL default)
    if not tool_calls:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    tool_calls.append(obj)
            except json.JSONDecodeError:
                pass

    # Strategy 3: brace-matching for multi-line objects
    if not tool_calls:
        depth, start = 0, None
        for i, ch in enumerate(block):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(block[start:i+1])
                        if isinstance(obj, dict):
                            tool_calls.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

    # Normalise keys
    normalised = []
    for tc in tool_calls:
        if "name" not in tc:
            continue
        name = tc["name"]
        if not isinstance(name, str):
            name = str(name)
        normalised.append({
            "name": name,
            "parameters": tc.get("parameters", tc.get("arguments", {})),
        })

    return normalised, len(normalised) > 0


def _parse_response(text: str) -> Tuple[str, bool]:
    """Extract content from <response>...</response>."""
    m = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), True
    return "", False


# ---------------------------------------------------------------------------
# Format score — mirrors customize_format_reward_func()
# ---------------------------------------------------------------------------

def _format_score(model_output: str, task_type: str) -> float:
    """
    Full score (1.0) if output matches expected format for task_type.
    Partial (0.5) if block present but not strict.
    0.0 otherwise.
    """
    if task_type == "tool_call":
        # strict: ^<think>...</think>\n<tool_call>\n...\n</tool_call>$
        strict = re.search(
            r"<think>.*?</think>\s*<tool_call>\n.+\n</tool_call>",
            model_output.strip(), re.DOTALL
        )
        if strict:
            return FMT_MAX
        # partial: has tool_call block
        if re.search(r"<tool_call>.*?</tool_call>", model_output, re.DOTALL):
            return 0.5
        return FMT_MIN

    elif task_type == "response":
        # strict: <think>...</think>\n<response>...</response>
        strict = re.search(
            r"<think>.*?</think>\s*<response>.+</response>",
            model_output.strip(), re.DOTALL
        )
        if strict:
            return FMT_MAX
        if re.search(r"<response>.*?</response>", model_output, re.DOTALL):
            return 0.5
        return FMT_MIN

    return FMT_MIN


# ---------------------------------------------------------------------------
# Correctness score — mirrors compute_tool_call_reward()
# ---------------------------------------------------------------------------

def _tool_correctness(gold: List[dict], pred: List[dict]) -> Tuple[float, bool]:
    """
    Mirrors ToolRL compute_tool_call_reward().
    Returns (score_in_[TOOL_MIN, TOOL_MAX], exact_match).
    """
    if gold == pred:
        return TOOL_MAX, True
    if not pred:
        return TOOL_MIN, False

    gt_names = [t["name"] for t in gold]
    pd_names = [t["name"] for t in pred]
    score = _match_score(gt_names, pd_names)

    local_max = 1.0
    used_pred: set = set()

    for gt_tool in gold:
        gt_name   = gt_tool["name"]
        gt_params = gt_tool.get("parameters", {})
        if not isinstance(gt_params, dict):
            gt_params = {}
        local_max += 1.0 + len(gt_params)

        best = 0.0
        best_idx = -1
        for i, pd_tool in enumerate(pred):
            if i in used_pred or pd_tool.get("name") != gt_name:
                continue
            pd_params = pd_tool.get("parameters", {})
            if not isinstance(pd_params, dict):
                pd_params = {}
            param_score = _match_score(list(gt_params.keys()), list(pd_params.keys()))
            val_score   = sum(
                1.0 for k, v in gt_params.items()
                if k in pd_params and str(pd_params[k]) == str(v)
            )
            total = param_score + val_score
            if total > best:
                best = total
                best_idx = i

        if best_idx >= 0:
            used_pred.add(best_idx)
        score += best

    normalized = (TOOL_MAX - TOOL_MIN) * score / local_max + TOOL_MIN
    return max(TOOL_MIN, min(TOOL_MAX, normalized)), False


def _response_correctness(gold_text: str, pred_text: str) -> Tuple[float, bool]:
    """
    Simple string match for <response> tasks.
    Exact match → TOOL_MAX, partial → proportional, no match → TOOL_MIN.
    """
    gold = gold_text.strip().lower()
    pred = pred_text.strip().lower()
    if not pred:
        return TOOL_MIN, False
    if gold == pred:
        return TOOL_MAX, True
    # partial: gold is substring of pred or vice versa
    if gold in pred or pred in gold:
        return TOOL_MAX * 0.5, False
    # word overlap
    gold_words = set(gold.split())
    pred_words = set(pred.split())
    if gold_words and pred_words:
        overlap = len(gold_words & pred_words) / max(len(gold_words), len(pred_words))
        score = (TOOL_MAX - TOOL_MIN) * overlap + TOOL_MIN
        return max(TOOL_MIN, min(TOOL_MAX, score)), False
    return TOOL_MIN, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(solution_str: str, ground_truth: str, step: int = 0) -> RllaRewardInfo:
    """
    Main reward function — aligned with ToolRL rlla.compute_score().

    Args:
        solution_str:  full model output (may include chat template tokens)
        ground_truth:  expected output from reward_model.ground_truth field
                       e.g. "<think>...</think>\n<tool_call>\n...\n</tool_call>"
                       or   "<think>...</think>\n<response>answer</response>"
        step:          training step (unused currently, kept for API compat)

    Returns:
        RllaRewardInfo with .reward in [-3.0, 4.0]
    """
    model_output = _extract_model_output(solution_str)

    # Determine task type from ground_truth
    if "<tool_call>" in ground_truth:
        task_type = "tool_call"
    elif "<response>" in ground_truth:
        task_type = "response"
    else:
        task_type = "unknown"

    fmt = _format_score(model_output, task_type)

    if task_type == "tool_call":
        gold_tools, gold_ok = _parse_tool_calls(ground_truth)
        pred_tools, pred_ok = _parse_tool_calls(model_output)

        if not gold_ok:
            # malformed ground_truth — give format score only
            return RllaRewardInfo(
                reward=fmt, format_score=fmt, correctness_score=0.0,
                parse_ok=False, exact_match=False, task_type=task_type,
                details={"reason": "bad_ground_truth"},
            )

        if not pred_ok:
            return RllaRewardInfo(
                reward=fmt + TOOL_MIN, format_score=fmt,
                correctness_score=TOOL_MIN,
                parse_ok=False, exact_match=False, task_type=task_type,
                details={"reason": "parse_failed"},
            )

        correctness, exact = _tool_correctness(gold_tools, pred_tools)
        return RllaRewardInfo(
            reward=fmt + correctness, format_score=fmt,
            correctness_score=correctness,
            parse_ok=True, exact_match=exact, task_type=task_type,
            details={"gold": gold_tools, "pred": pred_tools},
        )

    elif task_type == "response":
        gold_text, _ = _parse_response(ground_truth)
        pred_text, pred_ok = _parse_response(model_output)

        if not pred_ok:
            return RllaRewardInfo(
                reward=fmt + TOOL_MIN, format_score=fmt,
                correctness_score=TOOL_MIN,
                parse_ok=False, exact_match=False, task_type=task_type,
                details={"reason": "no_response_tag"},
            )

        correctness, exact = _response_correctness(gold_text, pred_text)
        return RllaRewardInfo(
            reward=fmt + correctness, format_score=fmt,
            correctness_score=correctness,
            parse_ok=True, exact_match=exact, task_type=task_type,
            details={"gold": gold_text, "pred": pred_text},
        )

    else:
        return RllaRewardInfo(
            reward=TOOL_MIN, format_score=FMT_MIN, correctness_score=TOOL_MIN,
            parse_ok=False, exact_match=False, task_type=task_type,
            details={"reason": "unknown_task_type"},
        )
