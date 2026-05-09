# 多 API Keys 实现总结

## 修改的文件

### 1. `src/tau2/continual_learning/config.py`
添加了 `user_api_keys` 配置项，支持传入多个 API keys：

```python
# 多卡训练时，每个卡使用不同的 API key（可选）
user_api_keys: Optional[list[str]] = None  # 如果提供，则按 rank 分配 key
```

### 2. `src/AGentCL/user/user_simulator.py`
修改 `GPT41UserSimulator` 构造函数，支持传入自定义 API key：

```python
def __init__(
    self,
    tools: Optional[list[Tool]] = None,
    instructions: Optional[UserInstructions] = None,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    api_key: Optional[str] = None,  # 新增：允许传入自定义 API key
):
    ...
    self.api_key = api_key if api_key is not None else USER_LLM_API_KEY
```

### 3. `src/tau2/continual_learning/policy_model.py`
添加了两个方法：

#### `_get_api_key_for_rank()`
根据当前 rank 获取对应的 API key：

```python
def _get_api_key_for_rank(self) -> str:
    """根据当前 rank 获取对应的 API key."""
    import os
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # 如果配置了多个 API keys，则按 rank 分配
    if self.config.user_api_keys is not None and len(self.config.user_api_keys) > 0:
        # 循环使用 keys（如果 rank 数量超过 key 数量）
        key_index = rank % len(self.config.user_api_keys)
        api_key = self.config.user_api_keys[key_index]
        if rank == 0:
            print(f"[API Key] Rank {rank} using key index {key_index} (total {len(self.config.user_api_keys)} keys)")
        return api_key
    else:
        # 使用默认的单个 key
        return self.config.user_api_key
```

#### 修改 `_create_user()`
在创建 user simulator 时传入对应的 API key：

```python
# 根据 rank 选择 API key
api_key = self._get_api_key_for_rank()

# Use GPT41UserSimulator which calls zidongtaichu API via SSE streaming
user = GPT41UserSimulator(
    instructions=task.user_scenario,
    tools=user_tools,
    api_key=api_key,  # 传入对应的 key
)
```

### 4. `src/tau2/scripts/train_grpo_cl.py`
添加命令行参数：

```python
parser.add_argument("--user_api_keys", type=str, nargs="+", default=None,
                    help="Multiple API keys for multi-GPU training (one key per rank)")
```

并在创建 config 时传入：

```python
config = GRPOConfig(
    ...
    user_api_keys=args.user_api_keys,
    ...
)
```

### 5. `src/tau2/scripts/train.sh`
添加环境变量支持：

```bash
# 多卡训练时，每个卡使用不同的 API key（可选）
# 格式：用空格分隔的多个 key，例如 "key1 key2 key3 key4 key5 key6 key7 key8"
# 如果不设置，则所有卡使用 USER_API_KEY
USER_API_KEYS=${USER_API_KEYS:-""}

# 添加多个 API keys（如果设置了）
if [ -n "${USER_API_KEYS}" ]; then
    CMD_ARGS+=(--user_api_keys ${USER_API_KEYS})
fi
```

## 工作原理

1. **Key 分配规则**：`key_index = rank % num_keys`
   - Rank 0 使用第 1 个 key
   - Rank 1 使用第 2 个 key
   - ...
   - 如果 GPU 数量 > key 数量，会循环使用

2. **流程**：
   ```
   train.sh
     → 设置 USER_API_KEYS 环境变量
     → 传递给 train_grpo_cl.py
     → 创建 GRPOConfig(user_api_keys=[...])
     → PolicyModel._get_api_key_for_rank() 根据 rank 选择 key
     → GPT41UserSimulator(api_key=selected_key) 使用对应的 key
   ```

3. **兼容性**：
   - 如果不设置 `USER_API_KEYS`，系统会使用单个 `USER_API_KEY`
   - 完全向后兼容，不影响现有代码

## 使用方法

### 方法 1: 环境变量（推荐）

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
bash src/tau2/scripts/train.sh
```

### 方法 2: 命令行

```bash
USER_API_KEYS="key1 key2 key3 key4" bash src/tau2/scripts/train.sh
```

### 方法 3: 使用提供的脚本

```bash
# 编辑 scripts/train_with_multi_keys.sh，填入你的 API keys
bash scripts/train_with_multi_keys.sh
```

### 方法 4: 修改 train.sh

编辑 `src/tau2/scripts/train.sh`，找到：

```bash
USER_API_KEYS=${USER_API_KEYS:-""}
```

修改为：

```bash
USER_API_KEYS=${USER_API_KEYS:-"key1 key2 key3 key4 key5 key6 key7 key8"}
```

## 验证

运行测试脚本：

```bash
python test_multi_keys.py
```

预期输出：

```
============================================================
测试 1: 单个 API Key
============================================================
Rank 0: using default key -> default_key
...
[PASS] 单个 key 测试通过

============================================================
测试 2: 多个 API Keys (8 个)
============================================================
Rank 0: key index 0 -> key_1
Rank 1: key index 1 -> key_2
...
[PASS] 多个 keys 测试通过

============================================================
测试 3: Keys 数量 (4) < Ranks 数量 (8)
============================================================
Rank 0: key index 0 -> key_1
Rank 4: key index 0 -> key_1  # 循环使用
...
[PASS] 循环使用 keys 测试通过

============================================================
[PASS] 所有测试通过!
============================================================
```

## 训练时的日志

训练开始时，Rank 0 会打印 key 分配信息：

```
[API Key] Rank 0 using key index 0 (total 8 keys)
```

## 优势

1. **避免流量限制**：每个 GPU 使用不同的 key，分散 API 请求
2. **灵活配置**：支持任意数量的 keys，自动循环使用
3. **向后兼容**：不影响现有单 key 配置
4. **易于使用**：只需设置环境变量即可

## 注意事项

1. **Key 数量**：建议 key 数量 >= GPU 数量，以获得最佳效果
2. **格式**：多个 keys 用空格分隔
3. **安全性**：不要将 API keys 提交到 git 仓库
4. **测试**：建议先用少量 steps 测试配置是否正确

## 故障排查

### 问题：仍然出现流量限制

**检查**：
- 确认 keys 数量是否足够
- 检查每个 key 的配额
- 查看日志确认 keys 是否正确分配

### 问题：训练启动失败

**检查**：
- 确认 keys 格式正确（空格分隔）
- 确认每个 key 都有效
- 查看 API 错误日志

### 问题：某些 rank 失败

**可能原因**：
- 某个 key 无效或已过期
- 某个 key 达到流量限制

**解决**：
- 替换失败的 key
- 增加更多 keys
