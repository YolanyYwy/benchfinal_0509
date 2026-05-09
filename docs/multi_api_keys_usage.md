# 多卡训练使用多个 API Keys

## 问题背景

在 8 卡联训时，如果所有卡共用一个 API key，可能会超出 API 的最大流量限制，导致请求失败。

## 解决方案

为每个 GPU 分配不同的 API key，避免单个 key 的流量限制。

## 使用方法

### 方法 1: 通过环境变量设置

在运行训练脚本前，设置 `USER_API_KEYS` 环境变量：

```bash
# 8 个不同的 API keys（用空格分隔）
export USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8"

# 运行训练
bash src/tau2/scripts/train.sh
```

### 方法 2: 在命令行中直接指定

```bash
USER_API_KEYS="key1 key2 key3 key4 key5 key6 key7 key8" bash src/tau2/scripts/train.sh
```

### 方法 3: 修改 train.sh 脚本

编辑 `src/tau2/scripts/train.sh`，找到以下行：

```bash
USER_API_KEYS=${USER_API_KEYS:-""}
```

修改为：

```bash
USER_API_KEYS=${USER_API_KEYS:-"key1 key2 key3 key4 key5 key6 key7 key8"}
```

## Key 分配规则

- 每个 GPU rank 会自动分配一个 API key
- 分配规则：`key_index = rank % num_keys`
- 例如：
  - Rank 0 使用 key1
  - Rank 1 使用 key2
  - ...
  - Rank 7 使用 key8

## 如果 GPU 数量多于 Key 数量

系统会循环使用 keys。例如，如果有 8 个 GPU 但只有 4 个 keys：

- Rank 0, 4 使用 key1
- Rank 1, 5 使用 key2
- Rank 2, 6 使用 key3
- Rank 3, 7 使用 key4

## 完整示例

```bash
#!/bin/bash

# 设置多个 API keys
export USER_API_KEYS="vp8ggmuy102xmtpcyf9enr3g \
                      another_key_here_2 \
                      another_key_here_3 \
                      another_key_here_4 \
                      another_key_here_5 \
                      another_key_here_6 \
                      another_key_here_7 \
                      another_key_here_8"

# 其他配置
export PHASE="2"
export NUM_STEPS=100
export CL_ALGORITHM="replay"

# 运行训练
bash src/tau2/scripts/train.sh
```

## 验证

训练开始时，Rank 0 会打印 key 分配信息：

```
[API Key] Rank 0 using key index 0 (total 8 keys)
```

## 注意事项

1. **Key 格式**：多个 keys 用空格分隔
2. **引号**：如果在命令行中设置，建议用引号包裹整个字符串
3. **兼容性**：如果不设置 `USER_API_KEYS`，系统会回退到使用单个 `USER_API_KEY`
4. **安全性**：不要将 API keys 提交到 git 仓库中

## 故障排查

### 问题：仍然出现流量限制错误

**可能原因**：
- Keys 数量不足
- 某些 keys 已达到流量限制

**解决方案**：
- 增加更多 API keys
- 检查每个 key 的流量配额

### 问题：训练启动失败

**检查**：
- 确认 keys 格式正确（空格分隔）
- 确认每个 key 都有效
- 查看日志中的 API 错误信息
