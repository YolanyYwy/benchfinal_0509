# 分域 Reward 计算方案设计

## 1. 背景与需求

### 1.1 问题描述

AGentCL 项目中有 6 个领域（domain），它们的任务复杂度和评估方式存在显著差异：

| 领域 | 任务特点 | 评估方式建议 |
|------|----------|--------------|
| **airline** | 单一任务（如取消预订、查询航班） | 任务完成度评估 |
| **telecom** | 单一任务（如套餐变更、账单查询） | 任务完成度评估 |
| **retail** | 单一任务（如商品退换、订单查询） | 任务完成度评估 |
| **delivery** | 复合任务（点餐+配送+时间约束） | 滑动轨迹评估 |
| **instore** | 复合任务（多商品选购+支付） | 滑动轨迹评估 |
| **ota** | 复合任务（酒店+机票+景点+天气判断） | 滑动轨迹评估 |

### 1.2 核心需求

1. **airline/telecom/retail**：使用现有的任务完成度评估（基于 `evaluation_criteria` 中的 `actions`、`nl_assertions` 等）
2. **delivery/instore/ota**：使用滑动窗口轨迹评估（参考 vitabench 实现），因为这些任务包含多个子任务（rubrics）

---

## 2. 数据结构分析

### 2.1 airline/telecom/retail 任务结构

```json
{
  "id": "1",
  "evaluation_criteria": {
    "actions": [
      {"action_id": "1_0", "name": "get_user_details", "arguments": {...}},
      {"action_id": "1_1", "name": "get_reservation_details", "arguments": {...}}
    ],
    "communicate_info": [],
    "nl_assertions": ["Agent should not approve the cancellation."]
  }
}
```

**特点**：
- 有明确的 `actions` 列表（期望的工具调用）
- 有 `nl_assertions`（自然语言断言）
- 任务相对简单，可以通过检查最终状态判断是否完成

### 2.2 delivery/instore/ota 任务结构

```json
{
  "id": "10711001",
  "user_scenario": {
    "instructions": {
      "task_instructions": "It's pouring rain outside... order some mild rice noodles... Absolutely avoid fried foods and those high in purine... You have surgery at 1:30 PM and need a one-hour nap..."
    }
  },
  "evaluation_criteria": {
    "expected_states": [
      {
        "required_orders": [...],
        "optional_orders": [],
        "state_rubrics": [
          "The rice noodle restaurant must support Dine-in available",
          "The rice noodle product must not be gold soup flavor",
          "The rice noodle product must not contain fried side dishes or high-purine ingredients",
          "The delivery address should be Yunnan University Affiliated Hospital...",
          "The delivery time should be around 2025-06-21 12:00:00..."
        ]
      }
    ],
    "overall_rubrics": [
      "The rice noodle restaurant must support Dine-in available",
      "The rice noodle product must not be gold soup flavor",
      "The rice noodle product must not contain fried side dishes or high-purine ingredients",
      "The delivery address should be Yunnan University Affiliated Hospital...",
      "The delivery time should be around 2025-06-21 12:00:00..."
    ]
  }
}
```

**特点**：
- 任务指令复杂，包含多个子任务
- 有条件分支（如天气判断）
- 需要多步骤完成（查询→选择→预订→确认）
- **已有预定义的 `state_rubrics` 和 `overall_rubrics`**，可直接用于滑动窗口评估
- 多个 `expected_states` 表示任务的多个阶段

---

## 3. 滑动窗口评估方案（参考 vitabench）

### 3.1 核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                        完整对话轨迹                              │
│  [M1] [M2] [M3] [M4] [M5] [M6] [M7] [M8] [M9] [M10] [M11] [M12] │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Window 1│          │ Window 2│          │ Window 3│
    │ M1-M10  │          │ M9-M12  │          │   ...   │
    └─────────┘          └─────────┘          └─────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Rubric  │    →     │ Rubric  │    →     │ Rubric  │
    │ States  │  更新    │ States  │  更新    │ States  │
    └─────────┘          └─────────┘          └─────────┘
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 10 | 每个窗口包含的消息数 |
| `overlap` | 2 | 相邻窗口重叠的消息数 |
| `step` | 8 | 窗口滑动步长 (window_size - overlap) |

### 3.3 Rubric 状态管理

```python
# 初始状态
rubric_states = {
    "rubric_0": {
        "rubric": "用户成功预订了酒店",
        "justification": "Not evaluated yet",
        "meetExpectation": False
    },
    "rubric_1": {
        "rubric": "预订了正确的房型（一间大床房+一间双床房）",
        "justification": "Not evaluated yet",
        "meetExpectation": False
    },
    # ...
}

# 每个窗口评估后更新状态
# 状态可以从 False → True（完成子任务）
# 也可以从 True → False（推翻之前的结论）
```

---

## 4. 实现方案

### 4.1 文件结构

```
src/tau2/continual_learning/
├── reward_oracle.py              # 现有文件，需要修改
├── evaluator_trajectory.py       # 新增：滑动窗口评估器
└── prompts/
    └── sliding_window_eval.yaml  # 新增：评估 prompt 模板
```

**注意**：由于 delivery/instore/ota 任务已包含预定义的 `state_rubrics` 和 `overall_rubrics`，不需要 `rubric_generator.py`。

### 4.2 核心类设计

#### 4.2.1 DomainAwareRewardOracle

```python
class DomainAwareRewardOracle:
    """分域感知的 Reward Oracle"""

    # 使用任务完成度评估的领域
    TASK_COMPLETION_DOMAINS = {"airline", "telecom", "retail"}

    # 使用滑动轨迹评估的领域
    TRAJECTORY_EVAL_DOMAINS = {"delivery", "instore", "ota"}

    def __init__(
        self,
        evaluation_type: str = "ALL",
        task_order: Optional[list[str]] = None,
        # 滑动窗口参数
        window_size: int = 10,
        overlap: int = 2,
        # LLM 评估器配置
        llm_evaluator: str = "gpt-4.1",
        llm_evaluator_api_base: str = "https://api.lingleap.com/v1",
        llm_evaluator_api_key: str = "sk-xxx",
    ):
        self.evaluation_type = EvaluationType(evaluation_type.lower())
        self.window_size = window_size
        self.overlap = overlap
        self.llm_evaluator = llm_evaluator
        self.llm_evaluator_api_base = llm_evaluator_api_base
        self.llm_evaluator_api_key = llm_evaluator_api_key

        # 初始化轨迹评估器
        self.trajectory_evaluator = TrajectoryEvaluator(
            llm_evaluator=llm_evaluator,
            api_base=llm_evaluator_api_base,
            api_key=llm_evaluator_api_key,
        )

    def compute_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
        solo_mode: bool = False
    ) -> RewardInfo:
        """根据领域选择不同的评估方式"""

        if domain in self.TASK_COMPLETION_DOMAINS:
            # airline/telecom/retail: 使用任务完成度评估
            return self._compute_task_completion_reward(
                task, trajectory, domain, solo_mode
            )
        elif domain in self.TRAJECTORY_EVAL_DOMAINS:
            # delivery/instore/ota: 使用滑动轨迹评估
            return self._compute_trajectory_reward(
                task, trajectory, domain
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _compute_task_completion_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str,
        solo_mode: bool
    ) -> RewardInfo:
        """任务完成度评估（现有逻辑）"""
        # 使用现有的 evaluate_simulation 函数
        simulation = self._trajectory_to_simulation(task, trajectory)
        return evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=self.evaluation_type,
            solo_mode=solo_mode,
            domain=domain
        )

    def _compute_trajectory_reward(
        self,
        task: Task,
        trajectory: Trajectory,
        domain: str
    ) -> RewardInfo:
        """滑动轨迹评估"""
        # 1. 生成 rubrics（如果任务没有预定义）
        rubrics = self._get_or_generate_rubrics(task, domain)

        # 2. 使用滑动窗口评估
        return self.trajectory_evaluator.calculate_reward(
            task=task,
            messages=trajectory.messages,
            rubrics=rubrics,
            window_size=self.window_size,
            overlap=self.overlap,
        )
```

#### 4.2.2 TrajectoryEvaluator

```python
class TrajectoryEvaluator:
    """滑动窗口轨迹评估器"""

    def __init__(
        self,
        llm_evaluator: str,
        api_base: str,
        api_key: str,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = llm_evaluator

    def calculate_reward(
        self,
        task: Task,
        messages: List[Message],
        rubrics: List[str],
        window_size: int = 10,
        overlap: int = 2,
    ) -> RewardInfo:
        """
        使用滑动窗口评估轨迹

        Args:
            task: 任务定义
            messages: 完整对话消息列表
            rubrics: 评估标准列表（子任务）
            window_size: 窗口大小
            overlap: 窗口重叠

        Returns:
            RewardInfo: 包含 reward 和详细评估结果
        """
        # 1. 初始化 rubric 状态
        rubric_states = self._initialize_rubric_states(rubrics)

        # 2. 创建滑动窗口
        windows = self._create_sliding_windows(messages, window_size, overlap)

        # 3. 逐窗口评估
        window_evaluations = []
        for i, window in enumerate(windows):
            window_start_idx = i * (window_size - overlap)
            rubric_states, eval_info = self._evaluate_window(
                task=task,
                window=window,
                current_states=rubric_states,
                window_idx=i + 1,
                total_windows=len(windows),
                window_start_idx=window_start_idx,
            )
            window_evaluations.append(eval_info)

        # 4. 计算最终 reward
        nl_rubric_checks = self._convert_states_to_checks(rubric_states)

        all_met = all(check.met for check in nl_rubric_checks)
        rubric_score = sum(1.0 if check.met else 0.0 for check in nl_rubric_checks) / len(nl_rubric_checks)
        reward = 1.0 if all_met else rubric_score  # 可以选择全部完成才给1，或者按比例给分

        return RewardInfo(
            reward=reward,
            nl_rubrics=nl_rubric_checks,
            reward_breakdown={"rubric_score": rubric_score},
            info={
                "evaluation_method": "sliding_window",
                "num_windows": len(windows),
                "window_size": window_size,
                "rubrics_met": sum(1 for c in nl_rubric_checks if c.met),
                "rubrics_total": len(nl_rubric_checks),
            },
            window_evaluations=window_evaluations,
        )

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

    def _evaluate_window(
        self,
        task: Task,
        window: List[Message],
        current_states: dict,
        window_idx: int,
        total_windows: int,
        window_start_idx: int,
    ) -> Tuple[dict, dict]:
        """评估单个窗口"""
        # 格式化窗口内容
        window_content = self._format_window_content(window, window_start_idx)
        current_rubrics_str = json.dumps(list(current_states.values()), ensure_ascii=False, indent=2)

        # 构建 prompt
        system_prompt = SLIDING_WINDOW_EVAL_PROMPT.format(
            user_instruction=task.instructions or str(task.user_scenario),
            window_idx=window_idx,
            total_windows=total_windows,
        )

        user_prompt = f"""
# Input
<window_content>
{window_content}
</window_content>

<current_rubrics>
{current_rubrics_str}
</current_rubrics>
"""

        # 调用 LLM 评估
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        assistant_content = response.choices[0].message.content

        # 解析结果并更新状态
        updated_states = copy.deepcopy(current_states)
        try:
            # 提取 JSON
            result_data = self._extract_json(assistant_content)
            if result_data:
                for result in result_data:
                    rubric_idx = result.get("rubric_idx")
                    if rubric_idx and rubric_idx in updated_states:
                        updated_states[rubric_idx]["justification"] = result.get("justification", "")
                        updated_states[rubric_idx]["meetExpectation"] = result.get("meetExpectation", False)
        except Exception as e:
            print(f"Warning: Failed to parse LLM response for window {window_idx}: {e}")

        eval_info = {
            "window_idx": window_idx,
            "assistant_response": assistant_content,
        }

        return updated_states, eval_info

    def _format_window_content(self, window: List[Message], start_idx: int) -> str:
        """格式化窗口内容"""
        lines = []
        for i, msg in enumerate(window):
            global_idx = start_idx + i + 1
            role = msg.role
            content = msg.content or ""

            # 处理 tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_strs = []
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    tool_strs.append(f"{tc.name}({args_str})")
                if tool_strs:
                    content = (content + " " if content else "") + "; ".join(tool_strs)

            if content:
                lines.append(f"[{global_idx}] {role}: {content}")

        return "\n".join(lines)
```

#### 4.2.3 RubricGenerator

```python
class RubricGenerator:
    """动态生成评估 Rubrics"""

    def __init__(self, llm_model: str, api_base: str, api_key: str):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = llm_model

    def generate_rubrics(self, task: Task, domain: str) -> List[str]:
        """
        根据任务指令生成评估 rubrics

        对于 delivery/instore/ota 这类复杂任务，
        将任务指令分解为多个可独立评估的子任务
        """
        task_instructions = task.user_scenario.instructions.task_instructions

        prompt = f"""
请分析以下任务指令，将其分解为多个独立的评估标准（rubrics）。
每个 rubric 应该是一个可以独立判断是否完成的子任务。

任务领域: {domain}
任务指令:
{task_instructions}

请输出 JSON 格式的 rubrics 列表，每个 rubric 应该是一个简洁的陈述句，描述需要完成的具体目标。

示例输出格式:
```json
[
    "用户成功预订了酒店",
    "预订的房型符合要求（一间大床房+一间双床房）",
    "预订日期正确（8月1日至8月3日，共3晚）",
    "根据天气情况选择了正确的景点",
    "购买了正确数量和类型的景点门票"
]
```

请输出 rubrics:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        content = response.choices[0].message.content

        # 解析 JSON
        try:
            # 提取 JSON 部分
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                rubrics = json.loads(json_match.group(1))
            else:
                rubrics = json.loads(content)
            return rubrics
        except:
            # 如果解析失败，返回默认 rubric
            return ["任务整体完成"]
```

### 4.3 Prompt 模板

```yaml
# prompts/sliding_window_eval.yaml
name: sliding_window_eval_template

chinese: |-
  # 用户完整指令
  {user_instruction}

  # 背景说明
  - 这是一个 user 与 assistant 之间的对话场景
  - assistant 可以调用工具获取信息和完成操作，工具返回结果以 tool 开头
  - 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点 rubric
  - 采用滑动窗口评估法，每次可见若干轮对话，rubric 状态会跨窗口保留
  - 你正在评估第 {window_idx} 个窗口（共 {total_windows} 个窗口）

  # 任务
  - 基于当前窗口的对话内容，更新得分点 rubric 的状态
  - 可以将状态由 false 更新为 true（当 assistant 完成了该目标）
  - 也可以将 true 更新为 false（当 assistant 推翻了之前的正确结论）

  # 注意事项
  - 所有评估以 assistant 的回复及工具调用是否完成 rubric 中的目标为准
  - 查询类工具返回的结果仅对 assistant 可见，不代表对用户推荐的内容
  - 对于订单类 rubric，必须确认 assistant 是否真的完成了下单操作

  # 格式要求
  回复 JSON 数组，每个元素包含：
  - rubric_idx: 规则标识符
  - rubric: 规则复述
  - justification: 状态变化解释
  - meetExpectation: 更新后的状态 (true/false)

english: |-
  # User Complete Instruction
  {user_instruction}

  # Background
  - This is a conversation between user and assistant
  - Assistant can call tools to retrieve information and complete operations
  - You need to evaluate whether the user instruction has been completed
  - Using sliding window evaluation, rubric status is preserved across windows
  - You are evaluating window {window_idx} (out of {total_windows} windows)

  # Task
  - Update rubric status based on the conversation in the current window
  - Update from false to true when assistant completed the goal
  - Update from true to false when assistant overturned a previous conclusion

  # Format
  Reply with JSON array, each element containing:
  - rubric_idx: Rubric identifier
  - rubric: Restatement of the rubric
  - justification: Explanation of status change
  - meetExpectation: Updated status (true/false)
```

---

## 5. 配置更新

### 5.1 GRPOConfig 新增参数

```python
@dataclass
class GRPOConfig:
    # ... 现有参数 ...

    # 分域评估配置
    use_domain_aware_reward: bool = True  # 是否启用分域评估

    # 滑动窗口参数（用于 delivery/instore/ota）
    trajectory_window_size: int = 10
    trajectory_overlap: int = 2

    # 评估器 LLM 配置
    evaluator_model: str = "gpt-4.1"
    evaluator_api_base: str = "https://api.lingleap.com/v1"
    evaluator_api_key: str = "sk-xxx"

    # Reward 计算方式
    # "binary": 全部 rubric 完成才给 1.0，否则 0.0
    # "proportional": 按完成的 rubric 比例给分
    trajectory_reward_mode: str = "proportional"
```

### 5.2 命令行参数

```python
# train_grpo_cl.py
parser.add_argument("--use_domain_aware_reward", action="store_true", default=True)
parser.add_argument("--trajectory_window_size", type=int, default=10)
parser.add_argument("--trajectory_overlap", type=int, default=2)
parser.add_argument("--evaluator_model", type=str, default="gpt-4.1")
parser.add_argument("--evaluator_api_base", type=str, default="https://api.lingleap.com/v1")
parser.add_argument("--evaluator_api_key", type=str, default="sk-xxx")
parser.add_argument("--trajectory_reward_mode", type=str, default="proportional",
                    choices=["binary", "proportional"])
```

---

## 6. 使用示例

### 6.1 训练命令

```bash
accelerate launch -m tau2.scripts.train_grpo_cl \
    --model_name_or_path /path/to/Qwen3-4B \
    --user_model gpt-4.1 \
    --user_api_base https://api.lingleap.com/v1 \
    --user_api_key sk-xxx \
    --no_local_user_model \
    --use_domain_aware_reward \
    --trajectory_window_size 10 \
    --trajectory_overlap 2 \
    --evaluator_model gpt-4.1 \
    --evaluator_api_base https://api.lingleap.com/v1 \
    --evaluator_api_key sk-xxx \
    --trajectory_reward_mode proportional \
    --task_order airline retail telecom delivery instore ota \
    --seed 42
```

### 6.2 评估输出示例

**airline 领域（任务完成度评估）**：
```
[Step 0] Task 1/4: task_0
  [✓] Sample 1: reward=1.000, termination=agent_stop, messages=8
      Action Match: 3/3, NL Assertions: 1/1
  [✗] Sample 2: reward=0.000, termination=max_steps, messages=50
      Action Match: 1/3, NL Assertions: 0/1
```

**ota 领域（滑动轨迹评估）**：
```
[Step 0] Task 1/4: D0812006
  Processing window 1/5 with 10 messages
  Processing window 2/5 with 10 messages
  Processing window 3/5 with 10 messages
  Processing window 4/5 with 10 messages
  Processing window 5/5 with 6 messages
  [✓] Sample 1: reward=0.800, termination=agent_stop, messages=46
      Rubrics: 4/5 met
      - ✓ 用户成功预订了酒店
      - ✓ 预订的房型符合要求
      - ✓ 预订日期正确
      - ✓ 根据天气选择了正确的景点
      - ✗ 购买了正确的门票（老人票）
```

---

## 7. 实现步骤

### Phase 1: 基础框架
1. [ ] 创建 `evaluator_trajectory.py`
2. [ ] 创建 `rubric_generator.py`
3. [ ] 创建 prompt 模板文件

### Phase 2: 集成
4. [ ] 修改 `reward_oracle.py`，添加 `DomainAwareRewardOracle`
5. [ ] 更新 `GRPOConfig` 添加新参数
6. [ ] 更新 `train_grpo_cl.py` 添加命令行参数

### Phase 3: 测试与优化
7. [ ] 单元测试各评估器
8. [ ] 集成测试完整训练流程
9. [ ] 调优滑动窗口参数

---

## 8. 注意事项

1. **API 成本**：滑动窗口评估需要多次调用 LLM，会增加 API 成本。建议：
   - 使用较小的窗口数量
   - 缓存相同任务的 rubrics
   - 考虑使用更便宜的模型进行评估

2. **评估一致性**：LLM 评估可能存在不一致性，建议：
   - 使用 temperature=0
   - 设置随机种子
   - 多次评估取平均（如果成本允许）

3. **Rubric 质量**：动态生成的 rubrics 质量影响评估准确性，建议：
   - 为常见任务预定义 rubrics
   - 定期检查生成的 rubrics 质量
   - 考虑人工审核重要任务的 rubrics

4. **性能优化**：
   - 批量处理多个轨迹的评估
   - 异步调用 API
   - 缓存重复的评估结果
