# 多 API Keys 支持 - 完整指南

## 概述

为了解决 8 卡联训时单个 API key 流量限制的问题，我们实现了多 API keys 支持。每个 GPU 可以使用不同的 API key，从而分散 API 请求压力。

## 快速开始

### 最简单的方式

```bash
# 1. 设置你的 API keys（用空格分隔）
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 2. 运行训练
bash src/tau2/scripts/train.sh
```

就这么简单！

## 详细使用方法

### 方法 1: 环境变量（推荐）

```bash
# 设置多个 keys
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 运行训练
bash src/tau2/scripts/train.sh
```

### 方法 2: 命令行一行搞定

```bash
USER_API_KEYS="key1 key2 key3 key4" bash src/tau2/scripts/train.sh
```

### 方法 3: 使用配置文件

```bash
# 1. 复制模板
cp configs/api_keys.txt.template configs/api_keys.txt

# 2. 编辑文件，填入你的 keys
vim configs/api_keys.txt

# 3. 测试 keys 是否有效
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 4. 生成配置并运行
python scripts/manage_api_keys.py --generate --keys-file configs/api_keys.txt --output api_keys.sh
source api_keys.sh
bash src/tau2/scripts/train.sh
```

### 方法 4: 使用提供的脚本模板

```bash
# 1. 编辑脚本
vim scripts/train_with_multi_keys.sh

# 2. 修改 USER_API_KEYS 这一行
export USER_API_KEYS="your_key1 your_key2 ..."

# 3. 运行
bash scripts/train_with_multi_keys.sh
```

## Key 分配规则

系统会自动根据 GPU rank 分配 API key：

```
key_index = rank % num_keys
```

### 示例 1: 8 个 Keys，8 个 GPUs

```
GPU 0 → key1
GPU 1 → key2
GPU 2 → key3
GPU 3 → key4
GPU 4 → key5
GPU 5 → key6
GPU 6 → key7
GPU 7 → key8
```

### 示例 2: 4 个 Keys，8 个 GPUs（循环使用）

```
GPU 0, 4 → key1
GPU 1, 5 → key2
GPU 2, 6 → key3
GPU 3, 7 → key4
```

## 工具和脚本

### 1. 测试脚本

验证配置是否正确：

```bash
python test_multi_keys.py
```

### 2. API Keys 管理工具

测试 keys 是否有效：

```bash
# 测试多个 keys
python scripts/manage_api_keys.py --test --keys key1 key2 key3 key4

# 从文件加载并测试
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 测试并生成配置
python scripts/manage_api_keys.py --test --generate --keys-file configs/api_keys.txt --output api_keys.sh
```

### 3. 训练脚本模板

```bash
bash scripts/train_with_multi_keys.sh
```

## 验证配置

### 方法 1: 运行测试

```bash
python test_multi_keys.py
```

预期输出：
```
[PASS] 所有测试通过!
```

### 方法 2: 查看训练日志

训练开始时，应该看到：

```
[API Key] Rank 0 using key index 0 (total 8 keys)
```

### 方法 3: 测试 API keys

```bash
python scripts/manage_api_keys.py --test --keys key1 key2 key3
```

## 完整配置示例

```bash
#!/bin/bash

# ========== API Keys 配置 ==========
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# ========== 训练配置 ==========
export PHASE="2"                    # 训练阶段
export NUM_STEPS=100                # 每个任务的训练步数
export CL_ALGORITHM="replay"        # 持续学习算法
export BATCH_SIZE=1                 # 每个 GPU 的 batch size
export LEARNING_RATE=5e-7           # 学习率
export KL_COEF=0.5                  # KL 系数
export REPLAY_RATIO=0.3             # Replay 比例
export NUM_SAMPLES=2                # 每个 prompt 的样本数
export SEED=42                      # 随机种子

# ========== 任务配置 ==========
export TASK_ORDER="airline retail telecom instore ota delivery"

# ========== 日志配置 ==========
export OUTPUT_LOG_DIR="logs/qwen3_4b_multi_keys"
export WANDB_PROJECT="qwen3-4b-cl"

# ========== 评估配置 ==========
export PASS_AT_K=4
export NUM_EVAL_SAMPLES=5
export NUM_EVAL_TASKS=20

# ========== 其他配置 ==========
export SKIP_INTERMEDIATE_EVAL="true"
export CHECKPOINT_INTERVAL=5

# ========== 运行训练 ==========
bash src/tau2/scripts/train.sh
```

## 文件结构

```
AGentCL/
├── configs/
│   └── api_keys.txt.template      # API keys 配置模板
├── docs/
│   ├── QUICK_START_MULTI_KEYS.md  # 快速开始指南
│   ├── multi_api_keys_usage.md    # 详细使用文档
│   └── multi_api_keys_implementation.md  # 实现细节
├── scripts/
│   ├── train_with_multi_keys.sh   # 训练脚本模板
│   └── manage_api_keys.py         # API keys 管理工具
├── test_multi_keys.py             # 配置测试脚本
└── src/
    ├── tau2/
    │   ├── continual_learning/
    │   │   ├── config.py          # 添加了 user_api_keys 配置
    │   │   ├── policy_model.py    # 添加了 _get_api_key_for_rank()
    │   │   └── grpo_trainer.py
    │   └── scripts/
    │       ├── train.sh           # 支持 USER_API_KEYS 环境变量
    │       └── train_grpo_cl.py   # 添加了 --user_api_keys 参数
    └── AGentCL/
        └── user/
            └── user_simulator.py  # GPT41UserSimulator 支持自定义 api_key
```

## 常见场景

### 场景 1: 开发测试（2 GPUs）

```bash
export USER_API_KEYS="key1 key2"
export NUM_STEPS=10
bash src/tau2/scripts/train.sh
```

### 场景 2: 小规模实验（4 GPUs）

```bash
export USER_API_KEYS="key1 key2 key3 key4"
export NUM_STEPS=50
export TASK_ORDER="airline retail"
bash src/tau2/scripts/train.sh
```

### 场景 3: 完整训练（8 GPUs）

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
export NUM_STEPS=100
export TASK_ORDER="airline retail telecom instore ota delivery"
bash src/tau2/scripts/train.sh
```

## 故障排查

### 问题 1: 仍然出现流量限制

**可能原因**：
- Keys 数量不足
- 某些 keys 已达到流量限制

**解决方案**：
```bash
# 1. 测试所有 keys
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 2. 增加更多有效的 keys
# 3. 检查每个 key 的流量配额
```

### 问题 2: 训练启动失败

**检查清单**：
- [ ] Keys 格式正确（空格分隔）
- [ ] 每个 key 都有效
- [ ] 环境变量已正确设置

```bash
# 验证环境变量
echo $USER_API_KEYS

# 测试 keys
python scripts/manage_api_keys.py --test --keys $USER_API_KEYS
```

### 问题 3: 某些 rank 失败

**诊断步骤**：
```bash
# 1. 查看日志
tail -f logs/run_logs/train_phase2_*.log

# 2. 检查哪个 rank 失败
grep "Rank.*failed" logs/run_logs/train_phase2_*.log

# 3. 测试对应的 key
python scripts/manage_api_keys.py --test --keys <failed_key>
```

### 问题 4: 看不到 key 分配日志

**可能原因**：配置未生效

**解决方案**：
```bash
# 1. 确认环境变量
echo $USER_API_KEYS

# 2. 确认传递给训练脚本
bash -x src/tau2/scripts/train.sh 2>&1 | grep USER_API_KEYS

# 3. 检查 Python 配置
python -c "
import os
keys = os.environ.get('USER_API_KEYS', '').split()
print(f'Found {len(keys)} keys')
"
```

## 最佳实践

### 1. 安全管理 Keys

```bash
# ✓ 好的做法
# 将 keys 存储在单独的文件中
echo "key1 key2 key3" > ~/.api_keys
chmod 600 ~/.api_keys
export USER_API_KEYS=$(cat ~/.api_keys)

# ✗ 不好的做法
# 不要将 keys 硬编码在脚本中并提交到 git
```

### 2. 测试先行

```bash
# 在正式训练前，先测试配置
python test_multi_keys.py
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
```

### 3. 监控使用情况

```bash
# 定期检查训练日志
tail -f logs/run_logs/train_phase2_*.log | grep -E "(API|Rate|Error)"

# 使用 wandb 监控
# 访问 https://wandb.ai/your-username/your-project
```

### 4. 准备备用 Keys

```bash
# 准备比 GPU 数量更多的 keys
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8 backup1 backup2"
```

### 5. 分阶段测试

```bash
# 阶段 1: 测试 keys
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 阶段 2: 小规模测试（少量 steps）
export NUM_STEPS=5
bash src/tau2/scripts/train.sh

# 阶段 3: 完整训练
export NUM_STEPS=100
bash src/tau2/scripts/train.sh
```

## 技术细节

### 实现原理

1. **配置层**：`GRPOConfig` 添加 `user_api_keys` 字段
2. **分配层**：`PolicyModel._get_api_key_for_rank()` 根据 rank 选择 key
3. **使用层**：`GPT41UserSimulator` 接收并使用分配的 key

### Key 分配算法

```python
def _get_api_key_for_rank(self) -> str:
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if self.config.user_api_keys:
        key_index = rank % len(self.config.user_api_keys)
        return self.config.user_api_keys[key_index]
    else:
        return self.config.user_api_key
```

### 向后兼容

- 如果不设置 `USER_API_KEYS`，系统使用单个 `USER_API_KEY`
- 完全兼容现有代码和配置

## 相关文档

- [快速开始指南](docs/QUICK_START_MULTI_KEYS.md) - 1 分钟快速配置
- [详细使用文档](docs/multi_api_keys_usage.md) - 完整的使用说明
- [实现细节](docs/multi_api_keys_implementation.md) - 技术实现细节

## 获取帮助

### 运行测试

```bash
# 配置测试
python test_multi_keys.py

# API keys 测试
python scripts/manage_api_keys.py --test --keys key1 key2 key3
```

### 查看日志

```bash
# 训练日志
tail -f logs/run_logs/train_phase2_*.log

# 查找错误
grep -i error logs/run_logs/train_phase2_*.log
```

### 常用命令

```bash
# 查看环境变量
env | grep USER_API

# 测试单个 key
curl -X POST https://cloud.zidongtaichu.com/maas/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt_oss_120b","messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# 查看 GPU 使用情况
nvidia-smi

# 查看训练进程
ps aux | grep train_grpo_cl.py
```

## 总结

通过多 API keys 支持，你可以：

✓ 避免单个 key 的流量限制
✓ 提高训练稳定性
✓ 灵活配置 key 数量
✓ 自动循环使用 keys
✓ 保持向后兼容

开始使用：

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
bash src/tau2/scripts/train.sh
```

就这么简单！
