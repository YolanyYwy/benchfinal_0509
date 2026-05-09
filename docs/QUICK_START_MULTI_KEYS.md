# 快速开始：8 卡训练使用多个 API Keys

## 1 分钟快速配置

### 步骤 1: 准备你的 API Keys

准备 8 个（或更多）紫东太初 API keys。如果只有 4 个，系统会自动循环使用。

### 步骤 2: 设置环境变量

```bash
# 方式 A: 直接在命令行设置（推荐用于测试）
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 方式 B: 写入脚本（推荐用于生产）
# 编辑 scripts/train_with_multi_keys.sh
```

### 步骤 3: 运行训练

```bash
# 如果使用方式 A
bash src/tau2/scripts/train.sh

# 如果使用方式 B
bash scripts/train_with_multi_keys.sh
```

## 完整示例

### 示例 1: 8 个 Keys，8 个 GPUs

```bash
#!/bin/bash

# 设置 8 个不同的 API keys
export USER_API_KEYS="vp8ggmuy102xmtpcyf9enr3g \
                      your_key_2_here \
                      your_key_3_here \
                      your_key_4_here \
                      your_key_5_here \
                      your_key_6_here \
                      your_key_7_here \
                      your_key_8_here"

# 其他配置
export PHASE="2"
export NUM_STEPS=100
export CL_ALGORITHM="replay"
export WANDB_PROJECT="my-experiment"

# 运行训练
bash src/tau2/scripts/train.sh
```

**Key 分配**：
- GPU 0 → key1
- GPU 1 → key2
- GPU 2 → key3
- GPU 3 → key4
- GPU 4 → key5
- GPU 5 → key6
- GPU 6 → key7
- GPU 7 → key8

### 示例 2: 4 个 Keys，8 个 GPUs（循环使用）

```bash
#!/bin/bash

# 只有 4 个 keys，系统会自动循环使用
export USER_API_KEYS="key1 key2 key3 key4"

export PHASE="2"
export NUM_STEPS=100

bash src/tau2/scripts/train.sh
```

**Key 分配**：
- GPU 0, 4 → key1
- GPU 1, 5 → key2
- GPU 2, 6 → key3
- GPU 3, 7 → key4

### 示例 3: 使用提供的脚本模板

```bash
# 1. 编辑脚本
vim scripts/train_with_multi_keys.sh

# 2. 修改这一行，填入你的 keys
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 3. 运行
bash scripts/train_with_multi_keys.sh
```

## 验证配置

### 方法 1: 运行测试脚本

```bash
python test_multi_keys.py
```

预期看到：
```
[PASS] 所有测试通过!
```

### 方法 2: 查看训练日志

训练开始时，应该看到：

```
[API Key] Rank 0 using key index 0 (total 8 keys)
```

这表示配置成功。

## 常见配置场景

### 场景 1: 开发测试（2 GPUs）

```bash
export USER_API_KEYS="key1 key2"
export NUM_STEPS=10  # 少量 steps 快速测试
bash src/tau2/scripts/train.sh
```

### 场景 2: 小规模实验（4 GPUs）

```bash
export USER_API_KEYS="key1 key2 key3 key4"
export NUM_STEPS=50
export TASK_ORDER="airline retail"  # 只训练 2 个 domains
bash src/tau2/scripts/train.sh
```

### 场景 3: 完整训练（8 GPUs）

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
export NUM_STEPS=100
export TASK_ORDER="airline retail telecom instore ota delivery"
bash src/tau2/scripts/train.sh
```

## 环境变量完整列表

```bash
# ========== API Keys（必需）==========
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# ========== 训练阶段 ==========
export PHASE="2"  # 1=zero-shot, 1.5=single-domain, 2=training, 3=eval, all=full

# ========== 模型配置 ==========
export MODEL_PATH="/home/houzhiyan/Qwen3-4B"

# ========== 训练超参数 ==========
export NUM_STEPS=100
export BATCH_SIZE=1
export LEARNING_RATE=5e-7
export KL_COEF=0.5
export NUM_SAMPLES=2
export SEED=42

# ========== 持续学习配置 ==========
export CL_ALGORITHM="replay"  # sequential, replay, ewc, fusion, etc.
export REPLAY_RATIO=0.3

# ========== 任务配置 ==========
export TASK_ORDER="airline retail telecom instore ota delivery"

# ========== 日志配置 ==========
export OUTPUT_LOG_DIR="logs/my_experiment"
export WANDB_PROJECT="my-wandb-project"

# ========== 评估配置 ==========
export PASS_AT_K=4
export NUM_EVAL_SAMPLES=5
export NUM_EVAL_TASKS=20

# ========== 其他配置 ==========
export SKIP_INTERMEDIATE_EVAL="true"  # 跳过中间评估，加快训练
export CHECKPOINT_INTERVAL=5  # 每 5 步保存一次 checkpoint
```

## 故障排查速查表

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 仍然出现流量限制 | Keys 数量不足 | 增加更多 keys |
| 训练启动失败 | Keys 格式错误 | 检查是否用空格分隔 |
| 某些 rank 失败 | 某个 key 无效 | 替换失败的 key |
| 看不到 key 分配日志 | 配置未生效 | 确认环境变量已设置 |
| API 调用超时 | 网络问题 | 检查网络连接 |

## 下一步

1. **监控训练**：使用 wandb 查看训练进度
   ```bash
   # 访问 https://wandb.ai/your-username/your-project
   ```

2. **查看日志**：
   ```bash
   tail -f logs/run_logs/train_phase2_*.log
   ```

3. **断点续训**：
   ```bash
   export RESUME_FROM_TASK=2
   bash src/tau2/scripts/train.sh
   ```

4. **评估模型**：
   ```bash
   export PHASE="3"
   export RESUME_FROM="logs/my_experiment/checkpoints/final"
   bash src/tau2/scripts/train.sh
   ```

## 获取帮助

- 查看详细文档：`docs/multi_api_keys_usage.md`
- 查看实现细节：`docs/multi_api_keys_implementation.md`
- 运行测试：`python test_multi_keys.py`

## 最佳实践

1. **测试先行**：先用少量 steps 测试配置
2. **监控流量**：观察每个 key 的使用情况
3. **备用 keys**：准备额外的 keys 以防某个失效
4. **日志检查**：定期检查训练日志，确保没有 API 错误
5. **安全存储**：将 keys 存储在安全的地方，不要提交到 git
