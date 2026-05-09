"""
API-Bank reward computation — aligned with ToolRL rlla.py.

Output format expected (no <think> block):
    <tool_call>
    {"name": "...", "parameters": {...}}
    </tool_call>

Reward design mirrors ToolRL:
  - Exact match          → max reward (3.0)
  - Partial match        → proportional in [-3.0, 3.0]
  - Format bonus         → [0.0, 1.0]  (well-formed <tool_call> block)
  - Parse failure        → min_possible (-3.0) for correctness, format still scored

Total range: [-3.0, 4.0]
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOOL_MAX =  3.0
TOOL_MIN = -3.0
FORMAT_MAX = 1.0
FORMAT_MIN = 0.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class RewardInfo:
    reward: float
    correctness: float
    format_score: float
    pred_tool_calls: List[dict]
    gold_tool_calls: List[dict]
    parse_ok: bool
    exact_match: bool
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_score(list1: list, list2: list) -> float:
    """Frequency-aware similarity (order-insensitive). Mirrors ToolRL match_score()."""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    c1, c2 = Counter(list1), Counter(list2)
    intersection = sum(min(c1[k], c2[k]) for k in c1.keys() & c2.keys())
    union = len(list1) + len(list2) - intersection
    return intersection / union if union > 0 else 0.0


def _tool_correctness(gold: List[dict], pred: List[dict]) -> Tuple[float, bool, dict]:
    """
    Mirrors ToolRL compute_tool_call_reward().
    Returns (normalized_score_in_[TOOL_MIN,TOOL_MAX], exact_match, details).
    """
    if gold == pred:
        return TOOL_MAX, True, {"reason": "exact_match"}

    if not pred:
        return TOOL_MIN, False, {"reason": "empty_pred"}

    gt_names = [t["name"] for t in gold]
    pd_names = [t["name"] for t in pred]
    score = _match_score(gt_names, pd_names)

    local_max = 1.0
    used_pred: set = set()

    per_tool = []
    for gt_tool in gold:
        gt_name   = gt_tool["name"]
        gt_params = gt_tool.get("parameters", {})
        local_max += 1.0 + len(gt_params)   # ToolRL: 1 + len(params) keys + values

        best_score = 0.0
        best_idx   = -1

        for i, pd_tool in enumerate(pred):
            if i in used_pred or pd_tool.get("name") != gt_name:
                continue
            pd_params = pd_tool.get("parameters", {})

            param_score = _match_score(
                list(gt_params.keys()), list(pd_params.keys())
            )
            val_score = sum(
                1.0 for k, v in gt_params.items()
                if k in pd_params and str(pd_params[k]) == str(v)
            )
            total = param_score + val_score
            if total > best_score:
                best_score = total
                best_idx   = i

        if best_idx >= 0:
            used_pred.add(best_idx)
        score += best_score
        per_tool.append({"name": gt_name, "score": best_score})

    normalized = (TOOL_MAX - TOOL_MIN) * score / local_max + TOOL_MIN
    normalized = max(TOOL_MIN, min(TOOL_MAX, normalized))
    return normalized, False, {"per_tool": per_tool, "raw": score, "max": local_max}


def _format_score(output_text: str) -> float:
    """
    Mirrors ToolRL customize_format_reward_func() for tool-call-only answers.

    Full score (1.0) requires the strict pattern:
        ^<tool_call>\n...\n</tool_call>$
    Partial score (0.5) if <tool_call>...</tool_call> is present but not strict.
    0.0 otherwise.
    """
    strict = re.search(
        r"^<tool_call>\n.+\n</tool_call>$",
        output_text.strip(), re.DOTALL
    )
    if strict and output_text.strip().count("<tool_call>") == 1:
        return FORMAT_MAX

    # partial: block present but not perfectly formatted
    if re.search(r"<tool_call>.*?</tool_call>", output_text, re.DOTALL):
        return 0.5

    return FORMAT_MIN


# ---------------------------------------------------------------------------
# Parser — mirrors ToolRL's split-on-tag approach
# ---------------------------------------------------------------------------

def parse_agent_output(output_text: str) -> Tuple[List[dict], bool]:
    """
    Parse LLM output into (tool_calls, parse_ok).

    Expected format (ToolRL style, no think block):
        <tool_call>
        {"name": "...", "parameters": {...}}
        {"name": "...", "parameters": {...}}
        </tool_call>

    Returns parse_ok=False if <tool_call> or </tool_call> is missing.
    """
    if "<tool_call>" not in output_text or "</tool_call>" not in output_text:
        return [], False

    block = output_text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    if not block:
        return [], False

    tool_calls: List[dict] = []

    # Strategy 1: entire block is a single multi-line JSON object
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            tool_calls.append(obj)
        elif isinstance(obj, list):
            tool_calls.extend(x for x in obj if isinstance(x, dict))
    except json.JSONDecodeError:
        pass

    # Strategy 2: one JSON object per line (ToolRL default)
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

    # Strategy 3: find all top-level {...} blobs (handles multi-line objects)
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
                    blob = block[start:i+1]
                    try:
                        obj = json.loads(blob)
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
        normalised.append({
            "name": tc["name"],
            "parameters": tc.get("parameters", tc.get("arguments", {})),
        })

    return normalised, len(normalised) > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(output_text: str, gold_tool_calls: List[dict]) -> RewardInfo:
    """
    Main reward function — aligned with ToolRL rlla.py compute_score().

    Args:
        output_text:     raw LLM output string
        gold_tool_calls: list of gold tool call dicts

    Returns:
        RewardInfo with scalar .reward in [-3.0, 4.0]
    """
    # Defensive: normalise gold
    if isinstance(gold_tool_calls, str):
        try:
            gold_tool_calls = json.loads(gold_tool_calls)
        except Exception:
            gold_tool_calls = []
    parsed_gold = []
    for g in gold_tool_calls:
        if isinstance(g, dict):
            parsed_gold.append(g)
        elif isinstance(g, str):
            try:
                parsed_gold.append(json.loads(g))
            except Exception:
                pass
    gold_tool_calls = parsed_gold

    fmt = _format_score(output_text)
    pred_tool_calls, parse_ok = parse_agent_output(output_text)

    if not parse_ok:
        return RewardInfo(
            reward=fmt + TOOL_MIN,   # format partial credit + min correctness
            correctness=TOOL_MIN,
            format_score=fmt,
            pred_tool_calls=[],
            gold_tool_calls=gold_tool_calls,
            parse_ok=False,
            exact_match=False,
            details={"reason": "parse_failed"},
        )

    correctness, exact, details = _tool_correctness(gold_tool_calls, pred_tool_calls)
    total = correctness + fmt

    return RewardInfo(
        reward=total,
        correctness=correctness,
        format_score=fmt,
        pred_tool_calls=pred_tool_calls,
        gold_tool_calls=gold_tool_calls,
        parse_ok=True,
        exact_match=exact,
        details=details,
    )
