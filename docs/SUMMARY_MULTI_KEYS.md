# 多 API Keys 功能实现总结

## 🎯 问题

8 卡联训时，所有 GPU 共用一个 API key 会导致：
- API 流量限制错误（403/429）
- 训练中断
- 无法充分利用多卡性能

## ✅ 解决方案

实现了**一卡一 key**的机制：
- 每个 GPU rank 使用不同的 API key
- 自动循环分配（如果 GPU 数量 > key 数量）
- 完全向后兼容

## 📝 修改的文件

### 核心代码修改（5 个文件）

1. **src/tau2/continual_learning/config.py**
   - 添加 `user_api_keys: Optional[list[str]]` 配置项

2. **src/AGentCL/user/user_simulator.py**
   - `GPT41UserSimulator.__init__()` 添加 `api_key` 参数

3. **src/tau2/continual_learning/policy_model.py**
   - 添加 `_get_api_key_for_rank()` 方法
   - 修改 `_create_user()` 传递对应的 key

4. **src/tau2/scripts/train_grpo_cl.py**
   - 添加 `--user_api_keys` 命令行参数

5. **src/tau2/scripts/train.sh**
   - 添加 `USER_API_KEYS` 环境变量支持

### 新增文件（8 个）

#### 文档
1. **docs/README_MULTI_KEYS.md** - 完整指南
2. **docs/QUICK_START_MULTI_KEYS.md** - 快速开始
3. **docs/multi_api_keys_usage.md** - 详细使用说明
4. **docs/multi_api_keys_implementation.md** - 实现细节

#### 工具和脚本
5. **test_multi_keys.py** - 配置测试脚本
6. **scripts/manage_api_keys.py** - API keys 管理工具
7. **scripts/train_with_multi_keys.sh** - 训练脚本模板
8. **configs/api_keys.txt.template** - Keys 配置模板

#### 安全配置
9. **.gitignore** - 添加 API keys 相关规则

## 🚀 快速使用

### 最简单的方式

```bash
# 1. 设置你的 API keys
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 2. 运行训练
bash src/tau2/scripts/train.sh
```

### 使用配置文件

```bash
# 1. 创建配置文件
cp configs/api_keys.txt.template configs/api_keys.txt

# 2. 编辑文件，填入你的 keys
vim configs/api_keys.txt

# 3. 测试 keys
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 4. 生成配置
python scripts/manage_api_keys.py --generate --keys-file configs/api_keys.txt --output api_keys.sh

# 5. 运行训练
source api_keys.sh
bash src/tau2/scripts/train.sh
```

### 使用脚本模板

```bash
# 1. 编辑脚本
vim scripts/train_with_multi_keys.sh

# 2. 修改 USER_API_KEYS 这一行
export USER_API_KEYS="your_key1 your_key2 ..."

# 3. 运行
bash scripts/train_with_multi_keys.sh
```

## 🔍 验证配置

### 方法 1: 运行测试脚本

```bash
python test_multi_keys.py
```

预期输出：
```
[PASS] 所有测试通过!
```

### 方法 2: 测试 API keys

```bash
python scripts/manage_api_keys.py --test --keys key1 key2 key3 key4
```

### 方法 3: 查看训练日志

训练开始时应该看到：
```
[API Key] Rank 0 using key index 0 (total 8 keys)
```

## 📊 Key 分配规则

### 8 个 Keys，8 个 GPUs
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

### 4 个 Keys，8 个 GPUs（循环使用）
```
GPU 0, 4 → key1
GPU 1, 5 → key2
GPU 2, 6 → key3
GPU 3, 7 → key4
```

## 🛠️ 工具说明

### 1. test_multi_keys.py
测试配置是否正确

```bash
python test_multi_keys.py
```

### 2. manage_api_keys.py
管理和测试 API keys

```bash
# 测试 keys
python scripts/manage_api_keys.py --test --keys key1 key2 key3

# 从文件测试
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 生成配置
python scripts/manage_api_keys.py --generate --keys-file configs/api_keys.txt --output api_keys.sh
```

### 3. train_with_multi_keys.sh
训练脚本模板，包含所有配置

```bash
bash scripts/train_with_multi_keys.sh
```

## 📚 文档结构

```
docs/
├── README_MULTI_KEYS.md              # 完整指南（推荐阅读）
├── QUICK_START_MULTI_KEYS.md        # 快速开始（1 分钟配置）
├── multi_api_keys_usage.md          # 详细使用说明
└── multi_api_keys_implementation.md # 技术实现细节
```

**推荐阅读顺序**：
1. QUICK_START_MULTI_KEYS.md - 快速上手
2. README_MULTI_KEYS.md - 完整了解
3. multi_api_keys_usage.md - 深入使用
4. multi_api_keys_implementation.md - 技术细节

## 🔧 配置示例

### 完整配置

```bash
#!/bin/bash

# API Keys（必需）
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 训练配置
export PHASE="2"
export NUM_STEPS=100
export CL_ALGORITHM="replay"
export BATCH_SIZE=1
export LEARNING_RATE=5e-7
export KL_COEF=0.5
export REPLAY_RATIO=0.3
export NUM_SAMPLES=2
export SEED=42

# 任务配置
export TASK_ORDER="airline retail telecom instore ota delivery"

# 日志配置
export OUTPUT_LOG_DIR="logs/qwen3_4b_multi_keys"
export WANDB_PROJECT="qwen3-4b-cl"

# 评估配置
export PASS_AT_K=4
export NUM_EVAL_SAMPLES=5
export NUM_EVAL_TASKS=20

# 其他配置
export SKIP_INTERMEDIATE_EVAL="true"
export CHECKPOINT_INTERVAL=5

# 运行训练
bash src/tau2/scripts/train.sh
```

## ⚠️ 注意事项

### 1. 安全性
- ✅ 已添加 `.gitignore` 规则，防止 keys 被提交
- ✅ 使用 `chmod 600` 保护 keys 文件
- ❌ 不要在公开的脚本中硬编码 keys

### 2. Keys 数量
- 推荐：keys 数量 >= GPU 数量
- 最少：至少 2 个 keys（避免单点故障）
- 循环：如果 keys 不足，会自动循环使用

### 3. 测试
- 训练前先测试 keys 是否有效
- 使用少量 steps 验证配置
- 监控训练日志，确保没有 API 错误

## 🐛 故障排查

### 问题 1: 仍然出现流量限制

```bash
# 1. 测试所有 keys
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt

# 2. 增加更多 keys
# 3. 检查每个 key 的配额
```

### 问题 2: 训练启动失败

```bash
# 1. 验证环境变量
echo $USER_API_KEYS

# 2. 测试 keys
python scripts/manage_api_keys.py --test --keys $USER_API_KEYS

# 3. 查看日志
tail -f logs/run_logs/train_phase2_*.log
```

### 问题 3: 某些 rank 失败

```bash
# 1. 查看哪个 rank 失败
grep "Rank.*failed" logs/run_logs/train_phase2_*.log

# 2. 测试对应的 key
python scripts/manage_api_keys.py --test --keys <failed_key>

# 3. 替换失败的 key
```

## 📈 优势

✅ **避免流量限制**：每个 GPU 使用不同的 key
✅ **提高稳定性**：分散 API 请求压力
✅ **灵活配置**：支持任意数量的 keys
✅ **自动循环**：keys 不足时自动循环使用
✅ **向后兼容**：不影响现有单 key 配置
✅ **易于使用**：只需设置环境变量

## 🎓 最佳实践

### 1. 测试先行
```bash
python test_multi_keys.py
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
```

### 2. 分阶段测试
```bash
# 小规模测试
export NUM_STEPS=5
bash src/tau2/scripts/train.sh

# 完整训练
export NUM_STEPS=100
bash src/tau2/scripts/train.sh
```

### 3. 监控训练
```bash
# 查看日志
tail -f logs/run_logs/train_phase2_*.log

# 使用 wandb
# 访问 https://wandb.ai/your-username/your-project
```

### 4. 准备备用 Keys
```bash
# 准备比 GPU 数量更多的 keys
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8 backup1 backup2"
```

## 📞 获取帮助

### 查看文档
- 快速开始：`docs/QUICK_START_MULTI_KEYS.md`
- 完整指南：`docs/README_MULTI_KEYS.md`
- 详细使用：`docs/multi_api_keys_usage.md`
- 技术细节：`docs/multi_api_keys_implementation.md`

### 运行测试
```bash
python test_multi_keys.py
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
```

### 查看日志
```bash
tail -f logs/run_logs/train_phase2_*.log
grep -i error logs/run_logs/train_phase2_*.log
```

## 🎉 总结

通过这次实现，你现在可以：

1. **轻松配置**：只需设置 `USER_API_KEYS` 环境变量
2. **自动分配**：系统自动为每个 GPU 分配 key
3. **避免限制**：不再受单个 key 的流量限制
4. **灵活使用**：支持任意数量的 keys，自动循环
5. **安全管理**：提供工具测试和管理 keys

开始使用：

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
bash src/tau2/scripts/train.sh
```

就这么简单！🚀

---

**创建时间**：2026-03-04
**版本**：v1.0
**状态**：已完成并测试
