"""
滑动窗口轨迹评估器 - 用于评估 instore/delivery/ota 等复杂任务
参考 vitabench 实现
"""
import json
import copy
from typing import List, Tuple, Optional

import requests

from AGentCL.data_model.message import Message, AssistantMessage, UserMessage
from AGentCL.data_model.simulation import RewardInfo
from AGentCL.data_model.tasks import Task
from AGentCL.config import USER_LLM_API_BASE, USER_LLM_API_KEY, USER_LLM_MODEL


# 滑动窗口评估 Prompt 模板
SLIDING_WINDOW_EVAL_PROMPT_CN = """# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- 由于对话轮次较多，我们采用滑动窗口评估法，即每次可见若干轮对话，rubric状态会跨窗口保留
- 你正在评估第 {window_idx} 个窗口（本次任务总共 {total_windows} 个窗口）

# 任务
- 基于当前窗口的对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在此窗口中完成了该目标
- 你也可以将true再次更新为false，当且仅当assistant在此窗口中推翻了之前的正确结论

# 注意事项
- 所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准
- 查询类tool返回的结果仅对assistant可见，不代表assistant对用户推荐的内容
- 对于订单类rubric，必须确认assistant是否真的完成了下单操作
- 如果当前窗口没有涉及某个规则，保持其原有状态不变

# 格式要求
回复JSON数组，每个元素包含：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

示例回复：
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<复述规则>",
      "justification": "<状态变化的简要解释>",
      "meetExpectation": true
  }}
]
```"""


class TrajectoryEvaluator:
    """
    滑动窗口轨迹评估器 - 用于评估 instore/delivery/ota 等使用 expected_states 的任务
    """

    def __init__(
        self,
        api_base: str = None,
        api_key: str = None,
        model: str = None,
    ):
        self.api_base = api_base or USER_LLM_API_BASE
        self.api_key = api_key or USER_LLM_API_KEY
        self.model = model or USER_LLM_MODEL

    def calculate_reward(
        self,
        task: Task,
        messages: List[Message],
        window_size: int = 10,
        overlap: int = 2,
    ) -> RewardInfo:
        """
        使用滑动窗口评估轨迹

        Args:
            task: 任务定义
            messages: 完整对话消息列表
            window_size: 窗口大小
            overlap: 窗口重叠

        Returns:
            RewardInfo: 包含 reward 和详细评估结果
        """
        # 检查是否有评估标准
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )

        # 提取 rubrics
        rubrics = self._extract_rubrics(task)
        if not rubrics:
            return RewardInfo(
                reward=1.0,
                info={"note": "No rubrics to evaluate"},
            )

        # 初始化 rubric 状态
        rubric_states = self._initialize_rubric_states(rubrics)

        # 创建滑动窗口
        windows = self._create_sliding_windows(messages, window_size, overlap)

        # 获取用户指令
        user_instruction = self._get_user_instruction(task)

        print(f"[TrajectoryEvaluator] Starting evaluation: {len(rubrics)} rubrics, {len(windows)} windows, {len(messages)} messages", flush=True)

        # 并行评估所有窗口（每个窗口独立调用 API，不依赖前一个窗口的状态）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        step = window_size - overlap
        window_results = [None] * len(windows)

        def evaluate_window_task(i, window):
            window_start_idx = i * step
            print(f"[TrajectoryEvaluator] Evaluating window {i+1}/{len(windows)}...", flush=True)
            result = self._evaluate_window(
                user_instruction=user_instruction,
                window=window,
                current_states=copy.deepcopy(rubric_states),  # 每个窗口从初始状态开始
                window_idx=i + 1,
                total_windows=len(windows),
                window_start_idx=window_start_idx,
            )
            print(f"[TrajectoryEvaluator] Window {i+1}/{len(windows)} done.", flush=True)
            return i, result

        with ThreadPoolExecutor(max_workers=min(len(windows), 8)) as executor:
            futures = [executor.submit(evaluate_window_task, i, window) for i, window in enumerate(windows)]
            for future in as_completed(futures):
                i, result = future.result()
                window_results[i] = result

        # 合并所有窗口的结果：对每个 rubric，取所有窗口中 meetExpectation=True 的最终状态
        for i, window_result in enumerate(window_results):
            if window_result is None:
                continue
            for rubric_key, state in window_result.items():
                if rubric_key in rubric_states and state.get("meetExpectation", False):
                    rubric_states[rubric_key]["meetExpectation"] = True
                    rubric_states[rubric_key]["justification"] = state.get("justification", "")

        # 计算最终 reward
        met_count = sum(1 for state in rubric_states.values() if state["meetExpectation"])
        total_count = len(rubric_states)
        rubric_score = met_count / total_count if total_count > 0 else 0.0

        # 调试信息
        print(f"[TrajectoryEvaluator] rubrics_met={met_count}/{total_count}, score={rubric_score:.2f}")
        if met_count == 0 and total_count > 0:
            print(f"[TrajectoryEvaluator] WARNING: No rubrics met! Rubric states:")
            for k, v in rubric_states.items():
                print(f"  {k}: {v['rubric'][:50]}... -> {v['meetExpectation']}")

        # 全部满足才给 1.0，否则按比例给分
        all_met = met_count == total_count and total_count > 0
        reward = 1.0 if all_met else rubric_score

        return RewardInfo(
            reward=reward,
            reward_basis=None,
            info={
                "evaluation_method": "sliding_window",
                "num_windows": len(windows),
                "window_size": window_size,
                "rubrics_met": met_count,
                "rubrics_total": total_count,
                "rubric_score": rubric_score,
                "rubric_states": rubric_states,
            },
        )

    def _extract_rubrics(self, task: Task) -> List[str]:
        """从任务中提取 rubrics。

        优先使用 overall_rubrics（它是 state_rubrics 的汇总版本）。
        只有当 overall_rubrics 不存在时才回退到 state_rubrics。
        两者不叠加，避免近义重复导致分数被稀释。
        """
        ec = task.evaluation_criteria
        if ec is None:
            return []

        # 优先使用 overall_rubrics
        overall_rubrics = getattr(ec, 'overall_rubrics', None)
        if overall_rubrics:
            # 去重但保持顺序
            seen = set()
            rubrics = []
            for rubric in overall_rubrics:
                if rubric not in seen:
                    seen.add(rubric)
                    rubrics.append(rubric)
            return rubrics

        # 回退：从 expected_states 中提取 state_rubrics
        rubrics = []
        seen = set()
        expected_states = getattr(ec, 'expected_states', None)
        if expected_states:
            for state in expected_states:
                state_rubrics = state.get('state_rubrics', []) if isinstance(state, dict) else getattr(state, 'state_rubrics', [])
                if state_rubrics:
                    for rubric in state_rubrics:
                        if rubric not in seen:
                            seen.add(rubric)
                            rubrics.append(rubric)

        return rubrics

    def _initialize_rubric_states(self, rubrics: List[str]) -> dict:
        """初始化 rubric 状态"""
        states = {}
        for i, rubric in enumerate(rubrics):
            states[f"rubric_{i}"] = {
                "rubric": rubric,
                "justification": "Not evaluated yet",
                "meetExpectation": False
            }
        return states

    def _create_sliding_windows(
        self,
        messages: List[Message],
        window_size: int,
        overlap: int
    ) -> List[List[Message]]:
        """创建滑动窗口"""
        if len(messages) <= window_size:
            return [messages]

        windows = []
        step = window_size - overlap
        i = 0

        while i < len(messages):
            window = messages[i:i + window_size]
            if window:
                windows.append(window)
            if i + window_size >= len(messages):
                break
            i += step

        return windows

    def _get_user_instruction(self, task: Task) -> str:
        """获取用户指令"""
        if hasattr(task, 'user_scenario') and task.user_scenario:
            if hasattr(task.user_scenario, 'instructions'):
                instructions = task.user_scenario.instructions
                if hasattr(instructions, 'task_instructions'):
                    return instructions.task_instructions
                elif isinstance(instructions, dict):
                    return instructions.get('task_instructions', str(instructions))
            return str(task.user_scenario)
        return str(task.description) if hasattr(task, 'description') else ""

    def _evaluate_window(
        self,
        user_instruction: str,
        window: List[Message],
        current_states: dict,
        window_idx: int,
        total_windows: int,
        window_start_idx: int,
    ) -> dict:
        """评估单个窗口"""
        # 格式化窗口内容
        window_content = self._format_window_content(window, window_start_idx)
        current_rubrics_str = json.dumps(list(current_states.values()), ensure_ascii=False, indent=2)

        # 构建 prompt
        system_prompt = SLIDING_WINDOW_EVAL_PROMPT_CN.format(
            user_instruction=user_instruction,
            window_idx=window_idx,
            total_windows=total_windows,
        )

        user_prompt = f"""# Input
<window_content>
{window_content}
</window_content>

<current_rubrics>
{current_rubrics_str}
</current_rubrics>"""

        try:
            # 调用紫东太初 API（SSE streaming）
            assistant_content = self._call_streaming_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            # Debug: 打印 API 返回内容的前 500 字符
            print(f"[TrajectoryEvaluator] Window {window_idx} API response length={len(assistant_content)}, "
                  f"preview: {assistant_content[:500]}", flush=True)

            # 解析结果并更新状态
            updated_states = copy.deepcopy(current_states)
            result_data = self._extract_json(assistant_content)

            if result_data:
                print(f"[TrajectoryEvaluator] Window {window_idx} parsed {len(result_data)} rubric results", flush=True)
                for result in result_data:
                    rubric_idx = self._resolve_rubric_idx(result, updated_states)
                    if rubric_idx and rubric_idx in updated_states:
                        updated_states[rubric_idx]["justification"] = result.get("justification", "")
                        updated_states[rubric_idx]["meetExpectation"] = result.get("meetExpectation", False)
            else:
                print(f"[TrajectoryEvaluator] Window {window_idx} WARNING: _extract_json returned None!", flush=True)
                print(f"[TrajectoryEvaluator] Full API response:\n{assistant_content}", flush=True)

            return updated_states

        except Exception as e:
            print(f"Warning: Failed to evaluate window {window_idx}: {e}")
            return current_states

    def _resolve_rubric_idx(self, result: dict, states: dict) -> str:
        """从 result dict 中提取 rubric_idx，兼容模型各种格式变体"""
        import re
        # 尝试多种可能的 key 名（模型可能截断或拼错）
        rubric_idx = ""
        for key in ("rubric_idx", "rub_idx", "rubric_index", "idx", "index", "id"):
            if key in result:
                rubric_idx = str(result[key])
                break

        if not rubric_idx:
            return ""

        # 已经能直接匹配
        if rubric_idx in states:
            return rubric_idx

        # "5" -> "rubric_5"
        if not rubric_idx.startswith("rubric_"):
            normalized = f"rubric_{rubric_idx}"
            if normalized in states:
                return normalized

        # "rubric5" -> "rubric_5" (缺少下划线)
        m = re.match(r'rubric(\d+)', rubric_idx)
        if m:
            normalized = f"rubric_{m.group(1)}"
            if normalized in states:
                return normalized

        return rubric_idx

    def _call_streaming_api(self, messages: list, temperature: float = 0.0) -> str:
        """Call zidongtaichu API with SSE streaming, return full text."""
        import time
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        max_retries = 8
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
            if resp.status_code == 403 or resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[TrajectoryEvaluator] API rate limited ({resp.status_code}), retry {attempt+1}/{max_retries} after {wait}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()

            content_parts = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

            return "".join(content_parts)

        # 所有重试都失败
        resp.raise_for_status()

    def _format_window_content(self, window: List[Message], start_idx: int) -> str:
        """格式化窗口内容"""
        lines = []
        for i, msg in enumerate(window):
            global_idx = start_idx + i + 1
            role = msg.role if hasattr(msg, 'role') else 'unknown'
            content = msg.content or ""

            # 处理 tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_strs = []
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False) if hasattr(tc, 'arguments') else "{}"
                    tool_strs.append(f"{tc.name}({args_str})")
                if tool_strs:
                    content = (content + " " if content else "") + "; ".join(tool_strs)

            if content:
                lines.append(f"[{global_idx}] {role}: {content}")

        return "\n".join(lines)

    def _fix_json_str(self, text: str) -> str:
        """尝试修复 gpt_oss_120b 常见的 JSON 格式错误"""
        import re

        # 1. 修复缺少左引号: "key":value" -> "key":"value"
        text = re.sub(r'":([a-zA-Z_])', r'":"\1', text)

        # 2. 修复 key 缺少左引号: {  key": -> { "key":
        #    以及换行后缺少左引号:  \n   key": -> \n   "key":
        text = re.sub(r'([{\[,]\s*\n?\s*)([a-zA-Z_][a-zA-Z_0-9]*)"(\s*:)', r'\1"\2"\3', text)

        # 3. 修复截断的 key 名: "rub_idx" -> "rubric_idx" (常见截断)
        text = re.sub(r'"rub_idx"', '"rubric_idx"', text)

        # 4. 修复尾部不完整的 JSON 对象 — 截断到最后一个完整对象
        #    找到最后一个 } 并在其后关闭数组
        last_brace = text.rfind('}')
        last_bracket = text.rfind(']')
        if last_brace > last_bracket:
            # JSON 数组被截断了，在最后一个 } 后补 ]
            text = text[:last_brace + 1] + '\n]'

        return text

    def _extract_json(self, text: str) -> Optional[List[dict]]:
        """从文本中提取 JSON，带容错处理"""
        import re

        if not text or not text.strip():
            print("[_extract_json] Empty or whitespace-only text", flush=True)
            return None

        # 尝试提取 ```json ... ``` 块
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
            result = self._try_parse_json_array(raw)
            if result is not None:
                print(f"[_extract_json] Successfully parsed from ```json``` block, got {len(result)} items", flush=True)
                return result

        # 尝试直接解析
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            raw = text[start:end + 1]
            result = self._try_parse_json_array(raw)
            if result is not None:
                print(f"[_extract_json] Successfully parsed from [...] block, got {len(result)} items", flush=True)
                return result

        # 最后兜底：找 [ 到文本末尾，尝试修复截断
        if start != -1:
            raw = text[start:]
            fixed = self._fix_json_str(raw)
            result = self._try_parse_json_array(fixed)
            if result is not None:
                print(f"[_extract_json] Successfully parsed after fixing truncated JSON, got {len(result)} items", flush=True)
                return result

        print(f"[_extract_json] Failed to extract JSON. Text preview: {text[:200]}", flush=True)
        return None

    def _try_parse_json_array(self, raw: str) -> Optional[List[dict]]:
        """尝试解析 JSON 数组，失败则修复后重试"""
        # 直接解析
        try:
            result = json.loads(raw)
            if isinstance(result, list) and result:
                return result
            elif isinstance(result, list):
                print("[_try_parse_json_array] Parsed empty list", flush=True)
                return None
            else:
                print(f"[_try_parse_json_array] Parsed non-list type: {type(result)}", flush=True)
        except json.JSONDecodeError as e:
            print(f"[_try_parse_json_array] Direct parse failed: {e}", flush=True)

        # 修复后解析
        try:
            fixed = self._fix_json_str(raw)
            result = json.loads(fixed)
            if isinstance(result, list) and result:
                print("[_try_parse_json_array] Successfully parsed after fixing", flush=True)
                return result
        except json.JSONDecodeError as e:
            print(f"[_try_parse_json_array] Fixed parse failed: {e}", flush=True)

        # 逐对象提取：通过大括号配对分割，能处理截断的 JSON
        parsed = self._extract_objects_by_brace_matching(raw)
        if not parsed:
            parsed = self._extract_objects_by_brace_matching(self._fix_json_str(raw))
        if parsed:
            print(f"[_try_parse_json_array] Extracted {len(parsed)} objects by brace matching", flush=True)
            return parsed

        print("[_try_parse_json_array] All parsing methods failed", flush=True)
        return None

    def _extract_objects_by_brace_matching(self, text: str) -> Optional[List[dict]]:
        """通过大括号配对提取 JSON 对象，能处理截断的数组"""
        parsed = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                in_string = False
                escape = False
                j = i
                while j < len(text):
                    ch = text[j]
                    if escape:
                        escape = False
                        j += 1
                        continue
                    if ch == '\\':
                        escape = True
                        j += 1
                        continue
                    if ch == '"':
                        in_string = not in_string
                    elif not in_string:
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                obj_str = text[i:j + 1]
                                try:
                                    obj = json.loads(obj_str)
                                    if isinstance(obj, dict):
                                        parsed.append(obj)
                                except json.JSONDecodeError:
                                    try:
                                        obj = json.loads(self._fix_json_str(obj_str))
                                        if isinstance(obj, dict):
                                            parsed.append(obj)
                                    except:
                                        pass
                                i = j + 1
                                break
                    j += 1
                else:
                    # 没找到配对的 }，最后一个对象被截断，跳过
                    break
            else:
                i += 1
        return parsed if parsed else None
