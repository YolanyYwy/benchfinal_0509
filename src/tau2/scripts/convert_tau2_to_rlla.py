"""
Convert tau2 domain tasks + API-Bank + Bamboogle → rlla_4k format for ToolRL-style training.

Output format (per record):
{
  "data_source": "rlla",
  "prompt": [{"role": "system", ...}, {"role": "user", ...}],
  "ability": "tool_use",
  "reward_model": {"style": "rule", "ground_truth": "<think>...</think>\n<tool_call>\n...\n</tool_call>"},
  "extra_info": {"index": i, "split": "train", "domain": "airline", "task_id": "2", "step": 0}
}

Supported domains:
  - airline   (uses real db + tools, actions field)
  - retail    (uses real db + tools, actions field)
  - api_bank  (direct conversion, answer field)
  - bamboogle (mock search obs, QA format)

delivery / instore use expected_states (no actions) → skipped.

Usage:
  python convert_tau2_to_rlla.py --output_dir data/rlla_unified
  python convert_tau2_to_rlla.py --domains airline retail --output_dir data/rlla_unified
  python convert_tau2_to_rlla.py --max_steps 2 --train_ratio 0.9
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# ── constants ─────────────────────────────────────────────────────────────────
MAX_STEPS = 999        # no limit — all steps in each task
TRAIN_RATIO = 0.9
SEED = 42

THINK_TEMPLATE = "<think> I will call {name} to proceed. </think>"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_json(obj: Any) -> Any:
    """Recursively convert Pydantic models / dataclasses to plain dicts."""
    if hasattr(obj, "model_dump"):
        return _to_json(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(i) for i in obj]
    return obj


def _build_system_prompt(tools_obj, domain_name: str) -> str:
    """Build a system prompt listing all available tools with their signatures."""
    lines = [
        f"You are a helpful assistant for the {domain_name} domain. "
        "Use the available tools to complete the user's request.\n",
        "**Available Tools**",
        "In your response, you can use the following tools:",
    ]
    for idx, (name, fn) in enumerate(tools_obj.tools.items(), 1):
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        params = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation
            ann_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "")
            )
            params[pname] = {"type": ann_str, "description": pname}
        lines.append(f"{idx}. Name: {name}")
        if doc:
            lines.append(f"   Description: {doc}")
        lines.append(f"   Parameters: {json.dumps(params)}")

    lines += [
        "",
        "**Output Format**",
        "```plaintext",
        "<think> Your thoughts and reasoning </think>",
        "<tool_call>",
        '{"name": "Tool name", "parameters": {"param": "value"}}',
        "</tool_call>",
        "```",
        "",
        "**Important Notes**",
        "1. Always include <think> before your tool call or response.",
        "2. Output ONLY the <tool_call> block after <think>. Do NOT include any other text.",
        "3. Refer to previous <tool_call> and <obs> in the history.",
    ]
    return "\n".join(lines)


def _build_user_prompt(user_scenario: dict, history: List[Tuple[dict, Any]],
                       next_action: dict = None) -> str:
    """Build the user message with full dialogue history.

    next_action: the ground-truth action for this step — used to build a
    reasoning hint so the model can infer the correct tool call from the obs.
    """
    instructions = user_scenario.get("instructions", {})
    reason = instructions.get("reason_for_call", "")
    known = instructions.get("known_info", "")
    task_instr = instructions.get("task_instructions", "")

    user_text = reason
    if known:
        user_text += f"\n{known}"
    if task_instr:
        user_text += f"\n{task_instr}"

    hist_lines = ""
    for action, obs in history:
        tool_line = json.dumps({"name": action["name"],
                                "parameters": action.get("arguments", {})})
        obs_str = json.dumps(_to_json(obs), ensure_ascii=False)
        hist_lines += f"<tool_call>\n{tool_line}\n</tool_call>\n"
        hist_lines += f"<obs> {obs_str} </obs>\n"

    prompt = (
        "**Dialogue Records History**\n"
        f"<user> {user_text.strip()} </user>\n"
        + (f"\n{hist_lines}" if hist_lines else "")
    )

    # Add a reasoning hint derived from the obs so the model can infer
    # the correct next tool call rather than guessing blindly.
    if next_action is not None:
        hint = _build_step_hint(history, next_action)
        if hint:
            prompt += f"\n**Current Step**\n{hint}\n"

    return prompt


def _build_step_hint(history: List[Tuple[dict, Any]], next_action: dict) -> str:
    """
    Build a natural-language hint that bridges the obs to the next tool call.
    The hint tells the agent *what it already knows* and *what to do next*,
    making the ground-truth action inferable from the context.
    """
    name = next_action["name"]
    args = next_action.get("arguments", {})

    if not history:
        # step 0 — no obs yet, hint is just the first action to take
        if "user_id" in args:
            return (f"You need to look up the user's account first. "
                    f"Call {name} with user_id={args['user_id']}.")
        return f"Start by calling {name}."

    # Collect what we know from previous obs
    last_action, last_obs = history[-1]
    last_obs_dict = _to_json(last_obs) if not isinstance(last_obs, dict) else last_obs

    # get_reservation_details — hint which reservation to check and why
    if name == "get_reservation_details":
        res_id = args.get("reservation_id", "")
        # Find where this reservation_id came from in the obs history
        for _, obs in history:
            obs_d = _to_json(obs) if not isinstance(obs, dict) else obs
            reservations = obs_d.get("reservations", [])
            if res_id in reservations:
                already_checked = [
                    a.get("arguments", {}).get("reservation_id")
                    for a, _ in history
                    if a["name"] == "get_reservation_details"
                ]
                remaining = [r for r in reservations if r not in already_checked]
                return (
                    f"You have the user's reservation list: {reservations}. "
                    f"You have already checked: {already_checked}. "
                    f"Now check the next one: {res_id}."
                )
        return f"Check reservation {res_id} to find the relevant booking."

    # cancel_reservation / update_* — hint what was found in obs
    if name in ("cancel_reservation", "cancel_flight_segment"):
        res_id = args.get("reservation_id", "")
        return (
            f"Based on the reservation details you retrieved, "
            f"reservation {res_id} matches the user's request. "
            f"Proceed to call {name}."
        )

    if name.startswith("update_"):
        param_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return (
            f"You have confirmed the relevant reservation. "
            f"Now call {name} with {param_str}."
        )

    if name == "send_certificate":
        return (
            f"The task requires sending a certificate. "
            f"Call {name} with {args}."
        )

    # Generic fallback
    param_str = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"Based on the information gathered, call {name}({param_str})."


def _make_record(system: str, user: str, ground_truth: str,
                 domain: str, task_id: str, step: int,
                 global_idx: int, split: str) -> dict:
    return {
        "data_source": "rlla",
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "ability": "tool_use",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index":   global_idx,
            "split":   split,
            "domain":  domain,
            "task_id": task_id,
            "step":    step,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Domain converters
# ─────────────────────────────────────────────────────────────────────────────

def convert_airline(data_dir: Path, max_steps: int) -> List[dict]:
    from AGentCL.domains.airline.data_model import FlightDB
    from AGentCL.domains.airline.tools import AirlineTools

    db_raw = json.load(open(data_dir / "db.json", encoding="utf-8"))
    tasks  = json.load(open(data_dir / "train_tasks.json", encoding="utf-8"))

    db    = FlightDB(**db_raw)
    tools = AirlineTools(db)
    sys_prompt = _build_system_prompt(tools, "airline")

    records = []
    for task in tasks:
        actions = task["evaluation_criteria"].get("actions", [])
        if not actions:
            continue
        actions = actions[:max_steps]

        # fresh db per task
        task_db    = FlightDB(**db_raw)
        task_tools = AirlineTools(task_db)

        history: List[Tuple[dict, Any]] = []
        for step_idx, action in enumerate(actions):
            name = action["name"]
            args = action.get("arguments", {})

            # execute first to validate args — skip step if tool call fails
            try:
                obs = task_tools.use_tool(name, **args)
            except Exception as e:
                print(f"  [skip] task={task['id']} step={step_idx} {name}({args}): {e}")
                break  # history is broken, no point continuing this task

            # build sample AFTER validating (state at this step = before execution)
            user_prompt = _build_user_prompt(task["user_scenario"], history,
                                             next_action=action)
            tool_line   = json.dumps({"name": name, "parameters": args},
                                     ensure_ascii=False)
            think       = THINK_TEMPLATE.format(name=name)
            ground_truth = f"{think}\n<tool_call>\n{tool_line}\n</tool_call>"

            records.append(_make_record(
                system=sys_prompt, user=user_prompt,
                ground_truth=ground_truth,
                domain="airline", task_id=task["id"], step=step_idx,
                global_idx=len(records), split="train",
            ))

            history.append((action, obs))

    print(f"[airline] {len(records)} step-samples from {len(tasks)} tasks")
    return records


def convert_retail(data_dir: Path, max_steps: int) -> List[dict]:
    from AGentCL.domains.retail.data_model import RetailDB
    from AGentCL.domains.retail.tools import RetailTools

    db_raw = json.load(open(data_dir / "db.json", encoding="utf-8"))
    tasks  = json.load(open(data_dir / "train_tasks.json", encoding="utf-8"))

    db    = RetailDB(**db_raw)
    tools = RetailTools(db)
    sys_prompt = _build_system_prompt(tools, "retail")

    records = []
    for task in tasks:
        actions = task["evaluation_criteria"].get("actions", [])
        if not actions:
            continue
        actions = actions[:max_steps]

        task_db    = RetailDB(**db_raw)
        task_tools = RetailTools(task_db)

        history: List[Tuple[dict, Any]] = []
        for step_idx, action in enumerate(actions):
            name = action["name"]
            args = action.get("arguments", {})

            # execute first to validate args — skip step if tool call fails
            try:
                obs = task_tools.use_tool(name, **args)
            except Exception as e:
                print(f"  [skip] task={task['id']} step={step_idx} {name}({args}): {e}")
                break  # history is broken, no point continuing this task

            user_prompt  = _build_user_prompt(task["user_scenario"], history,
                                              next_action=action)
            tool_line    = json.dumps({"name": name, "parameters": args},
                                      ensure_ascii=False)
            think        = THINK_TEMPLATE.format(name=name)
            ground_truth = f"{think}\n<tool_call>\n{tool_line}\n</tool_call>"

            records.append(_make_record(
                system=sys_prompt, user=user_prompt,
                ground_truth=ground_truth,
                domain="retail", task_id=task["id"], step=step_idx,
                global_idx=len(records), split="train",
            ))

            history.append((action, obs))

    print(f"[retail] {len(records)} step-samples from {len(tasks)} tasks")
    return records


def convert_api_bank(data_dir: Path, levels: List[int] = None) -> List[dict]:
    """Convert API-Bank level-{1,2,3}-api_processed.json."""
    levels = levels or [1, 2, 3]
    records = []

    SYS_HEADER = (
        "You are a helpful multi-turn dialogue assistant capable of leveraging "
        "tool calls to solve user tasks and provide structured chat responses.\n\n"
    )
    FORMAT_FOOTER = (
        "\n\n**Output Format**\n"
        "```plaintext\n"
        "<think> Your thoughts and reasoning </think>\n"
        "<tool_call>\n"
        '{"name": "Tool name", "parameters": {"param": "value"}}\n'
        "</tool_call>\n"
        "```\n"
        "Output ONLY the <tool_call> block after <think>. "
        "Do NOT include <response> or any other text."
    )

    for level in levels:
        path = data_dir / f"level-{level}-api_processed.json"
        if not path.exists():
            print(f"[api_bank] WARNING: {path} not found, skipping")
            continue
        data = json.load(open(path, encoding="utf-8"))

        for rec in data:
            answer = rec.get("answer", [])
            if not answer:
                continue

            # normalise: level-2 has a single dict, level-1/3 have a list
            if isinstance(answer, dict):
                answer = [answer]
            elif isinstance(answer, list):
                # each element might itself be a dict or a JSON string
                normalised = []
                for item in answer:
                    if isinstance(item, dict):
                        normalised.append(item)
                    elif isinstance(item, str):
                        try:
                            normalised.append(json.loads(item))
                        except json.JSONDecodeError:
                            pass
                answer = normalised
            if not answer:
                continue

            # build ground_truth from answer list
            tool_lines = "\n".join(json.dumps(tc, ensure_ascii=False) for tc in answer)
            think = (
                "<think> I should use the appropriate tool with proper "
                "parameters to respond to the user's need. </think>"
            )
            ground_truth = f"{think}\n<tool_call>\n{tool_lines}\n</tool_call>"

            # strip original Output Format section from system, add ours
            sys_content = rec["system"]
            for marker in ["**Output Format**", "**Important Notes**",
                           "**Steps for Each Turn**"]:
                idx = sys_content.find(marker)
                if idx != -1:
                    sys_content = sys_content[:idx].rstrip()
                    break
            sys_content = SYS_HEADER + sys_content.lstrip() + FORMAT_FOOTER

            records.append({
                "data_source": "rlla",
                "prompt": [
                    {"role": "system", "content": sys_content},
                    {"role": "user",   "content": rec["user"]},
                ],
                "ability": "tool_use",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index":   len(records),
                    "split":   "train",
                    "domain":  f"api_bank_level{level}",
                    "task_id": str(rec.get("other", {}).get("id", len(records))),
                    "step":    0,
                },
            })

    print(f"[api_bank] {len(records)} samples from levels {levels}")
    return records


def convert_bamboogle(data_path: Path) -> List[dict]:
    """Convert Bamboogle QA → mock-search multi-turn format."""
    data = json.load(open(data_path, encoding="utf-8"))

    SYS = (
        "You are a helpful multi-turn dialogue assistant capable of leveraging "
        "tool calls to solve user tasks and provide structured chat responses.\n\n"
        "**Available Tools**\n"
        "1. Name: Search\n"
        "   Description: Search the answer for a specific query using Google.\n"
        '   Parameters: {"query": {"type": "string", "description": "The query to search for."}}\n\n'
        "**Output Format**\n"
        "```plaintext\n"
        "<think> Your thoughts and reasoning </think>\n"
        "<tool_call>\n"
        '{"name": "Search", "parameters": {"query": "your query"}}\n'
        "</tool_call>\n"
        "or, if you have the answer:\n"
        "<think> ... </think>\n"
        "<response> your answer </response>\n"
        "```\n"
        "Always include <think>. Refer to previous <obs> in the history."
    )

    records = []
    for i, rec in enumerate(data):
        question    = rec["Question"]
        gold_answer = rec["Answer"]

        # mock search obs containing the gold answer
        mock_obs = (
            f"Search results for: {question}\n"
            f"1. Reference Encyclopedia\n"
            f"   - Snippet: The answer is {gold_answer}. "
            f"This is well-documented in reference sources.\n"
            f"2. Knowledge Base\n"
            f"   - Snippet: {gold_answer} is the commonly accepted answer."
        )
        tool_call_json = json.dumps(
            {"name": "Search", "parameters": {"query": question}},
            ensure_ascii=False,
        )

        user_content = (
            "**Dialogue Records History**\n"
            f"<user> {question} </user>\n\n"
            f"<think> I should search for the answer. </think>\n"
            f"<tool_call>\n{tool_call_json}\n</tool_call>\n\n"
            f"<obs> {mock_obs} </obs>\n\n"
            "<user> Based on the search results, please give a succinct final answer. </user>"
        )

        ground_truth = (
            f"<think> Based on the search results, the answer is {gold_answer}. </think>\n"
            f"<response> {gold_answer} </response>"
        )

        records.append({
            "data_source": "rlla",
            "prompt": [
                {"role": "system", "content": SYS},
                {"role": "user",   "content": user_content},
            ],
            "ability": "tool_use",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "index":   i,
                "split":   "train",
                "domain":  "bamboogle",
                "task_id": str(i),
                "step":    0,
            },
        })

    print(f"[bamboogle] {len(records)} samples")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Split + save
# ─────────────────────────────────────────────────────────────────────────────

def save_domain(records: List[dict], output_dir: Path,
                train_ratio: float, seed: int, domain_name: str):
    """Split and save one domain's records into train.json / test.json."""
    rng = random.Random(seed)
    rng.shuffle(records)

    n_train = int(len(records) * train_ratio)
    train   = records[:n_train]
    test    = records[n_train:]

    for i, r in enumerate(train):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "train"
    for i, r in enumerate(test):
        r["extra_info"]["index"] = i
        r["extra_info"]["split"] = "test"

    domain_dir = output_dir / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    with open(domain_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(domain_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    stats = {"total": len(records), "train": len(train), "test": len(test)}
    with open(domain_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"  [{domain_name}] total={len(records)}  train={len(train)}  test={len(test)}"
          f"  → {domain_dir}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert tau2 / API-Bank / Bamboogle → rlla_4k training format"
    )
    parser.add_argument(
        "--domains", nargs="+",
        default=["airline", "retail", "api_bank", "bamboogle"],
        choices=["airline", "retail", "api_bank", "bamboogle"],
        help="Which datasets to include",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT / "data" / "rlla_by_domain"),
    )
    parser.add_argument("--max_steps",   type=int,   default=MAX_STEPS)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument(
        "--api_bank_levels", nargs="+", type=int, default=[1, 2, 3],
    )
    args = parser.parse_args()

    tau2_dir       = ROOT / "data" / "tau2" / "domains"
    apibank_dir    = ROOT / "ToolRL" / "benchmarks" / "API-Bank"
    bamboogle_path = ROOT / "ToolRL" / "benchmarks" / "Bamboogle" / "data.json"
    output_dir     = Path(args.output_dir)

    print(f"\n{'='*50}")
    print(f"Output root: {output_dir}")
    print(f"{'='*50}")

    all_stats = {}

    if "airline" in args.domains:
        records = convert_airline(tau2_dir / "airline", args.max_steps)
        all_stats["airline"] = save_domain(
            records, output_dir, args.train_ratio, args.seed, "airline")

    if "retail" in args.domains:
        records = convert_retail(tau2_dir / "retail", args.max_steps)
        all_stats["retail"] = save_domain(
            records, output_dir, args.train_ratio, args.seed, "retail")

    if "api_bank" in args.domains:
        records = convert_api_bank(apibank_dir, args.api_bank_levels)
        all_stats["api_bank"] = save_domain(
            records, output_dir, args.train_ratio, args.seed, "api_bank")

    if "bamboogle" in args.domains:
        records = convert_bamboogle(bamboogle_path)
        all_stats["bamboogle"] = save_domain(
            records, output_dir, args.train_ratio, args.seed, "bamboogle")

    # summary stats
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    total = sum(v["total"] for v in all_stats.values())
    print(f"\nDone. {total} total samples across {len(all_stats)} domains.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
