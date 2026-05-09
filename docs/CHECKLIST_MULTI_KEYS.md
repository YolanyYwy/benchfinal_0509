# 多 API Keys 功能 - 使用清单

## ✅ 实现完成清单

### 核心功能
- [x] 支持多个 API keys 配置
- [x] 自动按 rank 分配 keys
- [x] 循环使用 keys（当 GPU 数量 > key 数量）
- [x] 向后兼容单 key 配置
- [x] 添加 .gitignore 规则防止 keys 泄露

### 代码修改
- [x] `src/tau2/continual_learning/config.py` - 添加 `user_api_keys` 配置
- [x] `src/AGentCL/user/user_simulator.py` - 支持自定义 `api_key`
- [x] `src/tau2/continual_learning/policy_model.py` - 实现 key 分配逻辑
- [x] `src/tau2/scripts/train_grpo_cl.py` - 添加命令行参数
- [x] `src/tau2/scripts/train.sh` - 支持环境变量

### 文档
- [x] `docs/README_MULTI_KEYS.md` - 完整指南
- [x] `docs/QUICK_START_MULTI_KEYS.md` - 快速开始
- [x] `docs/multi_api_keys_usage.md` - 详细使用说明
- [x] `docs/multi_api_keys_implementation.md` - 实现细节
- [x] `docs/SUMMARY_MULTI_KEYS.md` - 功能总结

### 工具和脚本
- [x] `test_multi_keys.py` - 配置测试脚本
- [x] `scripts/manage_api_keys.py` - API keys 管理工具
- [x] `scripts/train_with_multi_keys.sh` - 训练脚本模板
- [x] `scripts/training_scenarios.sh` - 8 种训练场景
- [x] `scripts/getting_started.sh` - 交互式配置向导

### 配置模板
- [x] `configs/api_keys.txt.template` - Keys 配置模板

### 测试
- [x] 配置测试通过
- [x] Key 分配逻辑验证
- [x] 循环使用测试

---

## 🚀 快速开始（3 步）

### 步骤 1: 设置 API Keys

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
```

### 步骤 2: 验证配置

```bash
python test_multi_keys.py
```

### 步骤 3: 运行训练

```bash
bash src/tau2/scripts/train.sh
```

---

## 📋 使用前检查清单

### 准备工作
- [ ] 准备好 8 个（或更多）API keys
- [ ] 确认每个 key 都有效
- [ ] 确认 keys 有足够的流量配额

### 配置检查
- [ ] 已设置 `USER_API_KEYS` 环境变量
- [ ] Keys 格式正确（空格分隔）
- [ ] 已运行 `test_multi_keys.py` 验证配置

### 安全检查
- [ ] Keys 未硬编码在脚本中
- [ ] Keys 文件已添加到 `.gitignore`
- [ ] Keys 文件权限设置为 600

### 训练配置
- [ ] 已选择合适的训练场景
- [ ] 已设置输出目录
- [ ] 已配置 wandb（可选）

---

## 🔧 常用命令速查

### 测试配置
```bash
# 测试配置是否正确
python test_multi_keys.py

# 测试 API keys 是否有效
python scripts/manage_api_keys.py --test --keys key1 key2 key3

# 从文件测试
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
```

### 运行训练
```bash
# 方式 1: 直接设置环境变量
export USER_API_KEYS="key1 key2 key3 key4"
bash src/tau2/scripts/train.sh

# 方式 2: 命令行一行搞定
USER_API_KEYS="key1 key2 key3 key4" bash src/tau2/scripts/train.sh

# 方式 3: 使用脚本模板
bash scripts/train_with_multi_keys.sh

# 方式 4: 使用场景脚本
bash scripts/training_scenarios.sh 1  # 开发测试

# 方式 5: 交互式配置
bash scripts/getting_started.sh
```

### 监控训练
```bash
# 查看日志
tail -f logs/run_logs/train_phase2_*.log

# 查看 GPU 使用
nvidia-smi

# 查看进程
ps aux | grep train_grpo_cl.py

# 查看 wandb
# 访问 https://wandb.ai/your-username/your-project
```

### 管理 Keys
```bash
# 生成配置文件
python scripts/manage_api_keys.py --generate --keys key1 key2 key3 --output api_keys.sh

# 从文件生成配置
python scripts/manage_api_keys.py --generate --keys-file configs/api_keys.txt --output api_keys.sh

# 使用生成的配置
source api_keys.sh
bash src/tau2/scripts/train.sh
```

---

## 📊 训练场景速查

| 场景 | GPUs | Steps | Domains | 时间 | 用途 |
|------|------|-------|---------|------|------|
| 1 | 2 | 5 | 1 | 5-10分钟 | 开发测试 |
| 2 | 4 | 50 | 2 | 1-2小时 | 小规模实验 |
| 3 | 8 | 100 | 4 | 4-6小时 | 中等规模训练 |
| 4 | 8 | 100 | 6 | 8-12小时 | 完整训练 |
| 5 | 8 | 100 | 4 | 2-4小时/算法 | 对比实验 |
| 6 | 8 | - | - | 取决于剩余 | 断点续训 |
| 7 | 8 | - | 6 | 1-2小时 | 只评估 |
| 8 | 8 | 100 | 6 | 12-16小时 | 完整流程 |

使用方法：
```bash
bash scripts/training_scenarios.sh <场景编号>
```

---

## 🐛 故障排查速查

### 问题：流量限制错误

```bash
# 1. 检查 keys 数量
echo $USER_API_KEYS | wc -w

# 2. 测试所有 keys
python scripts/manage_api_keys.py --test --keys $USER_API_KEYS

# 3. 增加更多 keys
export USER_API_KEYS="$USER_API_KEYS new_key1 new_key2"
```

### 问题：训练启动失败

```bash
# 1. 验证环境变量
echo $USER_API_KEYS

# 2. 检查配置
python test_multi_keys.py

# 3. 查看详细日志
bash -x src/tau2/scripts/train.sh 2>&1 | tee debug.log
```

### 问题：某些 rank 失败

```bash
# 1. 查看失败的 rank
grep "Rank.*failed" logs/run_logs/train_phase2_*.log

# 2. 查看错误信息
grep -i error logs/run_logs/train_phase2_*.log

# 3. 测试对应的 key
# 假设 rank 2 失败，它使用第 3 个 key
python scripts/manage_api_keys.py --test --keys $(echo $USER_API_KEYS | cut -d' ' -f3)
```

### 问题：看不到 key 分配日志

```bash
# 1. 确认环境变量已设置
env | grep USER_API

# 2. 确认传递给训练脚本
bash -x src/tau2/scripts/train.sh 2>&1 | grep USER_API_KEYS

# 3. 检查 rank 0 的日志
grep "API Key" logs/run_logs/train_phase2_*.log
```

---

## 📁 文件结构速查

```
AGentCL/
├── configs/
│   └── api_keys.txt.template          # Keys 配置模板
├── docs/
│   ├── README_MULTI_KEYS.md           # 完整指南 ⭐
│   ├── QUICK_START_MULTI_KEYS.md      # 快速开始 ⭐
│   ├── SUMMARY_MULTI_KEYS.md          # 功能总结
│   ├── CHECKLIST_MULTI_KEYS.md        # 本文件
│   ├── multi_api_keys_usage.md        # 详细使用
│   └── multi_api_keys_implementation.md # 实现细节
├── scripts/
│   ├── getting_started.sh             # 交互式配置 ⭐
│   ├── training_scenarios.sh          # 8 种场景 ⭐
│   ├── train_with_multi_keys.sh       # 训练模板
│   └── manage_api_keys.py             # Keys 管理工具
├── test_multi_keys.py                 # 配置测试 ⭐
└── src/
    ├── tau2/
    │   ├── continual_learning/
    │   │   ├── config.py              # 配置（已修改）
    │   │   └── policy_model.py        # Key 分配（已修改）
    │   └── scripts/
    │       ├── train.sh               # 训练脚本（已修改）
    │       └── train_grpo_cl.py       # Python 脚本（已修改）
    └── AGentCL/
        └── user/
            └── user_simulator.py      # User simulator（已修改）
```

⭐ = 推荐首先查看

---

## 🎯 推荐工作流程

### 第一次使用

1. **阅读文档**
   ```bash
   cat docs/QUICK_START_MULTI_KEYS.md
   ```

2. **准备 Keys**
   - 复制模板：`cp configs/api_keys.txt.template configs/api_keys.txt`
   - 编辑文件：`vim configs/api_keys.txt`
   - 填入你的 keys

3. **测试配置**
   ```bash
   python test_multi_keys.py
   python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
   ```

4. **小规模测试**
   ```bash
   bash scripts/training_scenarios.sh 1  # 场景 1：开发测试
   ```

5. **正式训练**
   ```bash
   bash scripts/training_scenarios.sh 4  # 场景 4：完整训练
   ```

### 日常使用

1. **快速启动**
   ```bash
   export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
   bash src/tau2/scripts/train.sh
   ```

2. **使用场景脚本**
   ```bash
   bash scripts/training_scenarios.sh <场景编号>
   ```

3. **交互式配置**
   ```bash
   bash scripts/getting_started.sh
   ```

---

## 💡 最佳实践

### 安全管理
- ✅ 将 keys 存储在单独的文件中
- ✅ 设置文件权限：`chmod 600 configs/api_keys.txt`
- ✅ 不要将 keys 提交到 git
- ✅ 定期更换 keys

### 测试策略
- ✅ 训练前先测试配置
- ✅ 使用少量 steps 验证
- ✅ 监控训练日志
- ✅ 准备备用 keys

### 训练策略
- ✅ 从小规模开始
- ✅ 使用 wandb 监控
- ✅ 定期保存 checkpoint
- ✅ 记录实验配置

---

## 📞 获取帮助

### 查看文档
```bash
# 快速开始
cat docs/QUICK_START_MULTI_KEYS.md

# 完整指南
cat docs/README_MULTI_KEYS.md

# 详细使用
cat docs/multi_api_keys_usage.md

# 实现细节
cat docs/multi_api_keys_implementation.md
```

### 运行测试
```bash
# 配置测试
python test_multi_keys.py

# API keys 测试
python scripts/manage_api_keys.py --test --keys-file configs/api_keys.txt
```

### 查看日志
```bash
# 训练日志
tail -f logs/run_logs/train_phase2_*.log

# 查找错误
grep -i error logs/run_logs/train_phase2_*.log

# 查看 key 分配
grep "API Key" logs/run_logs/train_phase2_*.log
```

---

## ✨ 总结

你现在拥有：

✅ **完整的多 API keys 支持**
- 一卡一 key，避免流量限制
- 自动分配，无需手动配置
- 循环使用，灵活应对不同场景

✅ **丰富的工具和脚本**
- 配置测试脚本
- Keys 管理工具
- 8 种训练场景
- 交互式配置向导

✅ **详细的文档**
- 快速开始指南
- 完整使用文档
- 实现细节说明
- 故障排查指南

✅ **最佳实践**
- 安全管理 keys
- 测试驱动开发
- 监控和日志
- 断点续训

**开始使用：**

```bash
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"
bash src/tau2/scripts/train.sh
```

就这么简单！🚀

---

**文档版本**：v1.0
**创建时间**：2026-03-04
**最后更新**：2026-03-04
