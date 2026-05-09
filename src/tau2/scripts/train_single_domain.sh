#!/usr/bin/env bash

# 单任务训练脚本：在每个 domain 上单独训练模型并保存权重
# 用于计算 FWT 的 baseline：
# FWT_j = 顺序训练后性能 - 单任务训练后性能
#
# 支持多卡训练（DeepSpeed ZeRO-2）
# 支持断点续训：
#   - 从指定 domain 开始：START_FROM_DOMAIN=retail bash train_single_domain.sh
#   - 从指定 step 开始：RESUME_FROM_CHECKPOINT=logs/xxx/checkpoint_task_0_step_50 bash train_single_domain.sh

# PyTorch 内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置 PYTHONPATH
export PYTHONPATH=/home/houzhiyan/agent-ywy/agent/AGentCL/src:$PYTHONPATH

# CUDA 设备顺序
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL 配置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# PyTorch 分布式配置
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# 日志目录
LOG_DIR=logs/run_logs
mkdir -p ${LOG_DIR}

# ========== 配置参数（可通过环境变量覆盖）==========

# 模型配置
MODEL_PATH=${MODEL_PATH:-"/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct"}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"./single_domain_checkpoint"}

# 训练配置
NUM_STEPS=${NUM_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-5e-7}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-42}

# Checkpoint 保存间隔
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-101}

# 是否跳过中间评估
SKIP_INTERMEDIATE_EVAL=${SKIP_INTERMEDIATE_EVAL:-"true"}

# W&B 配置
WANDB_PROJECT=${WANDB_PROJECT:-"Agent"}
WANDB_ENTITY=${WANDB_ENTITY:-"yuwy22-tsinghua-university"}
WANDB_MODE=${WANDB_MODE:-"online"}

# 早停配置（如果需要）
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-20}
EARLY_STOPPING_THRESHOLD=${EARLY_STOPPING_THRESHOLD:-0.7}

# User API 配置
USER_API_BASE=${USER_API_BASE:-"https://cloud.zidongtaichu.com/maas/v1"}
# 支持多个 API keys（用空格分隔）或单个 key
USER_API_KEYS=${USER_API_KEYS:-""}
USER_API_KEY=${USER_API_KEY:-"vp8ggmuy102xmtpcyf9enr3g"}
USER_MODEL=${USER_MODEL:-"gpt_oss_120b"}

# Accelerate 配置文件
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/houzhiyan/agent-ywy/agent/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_8gpu_bf16.yaml"}

# 要训练的 domain 列表（不包括最后一个用于泛化测试的 domain）
DOMAINS=${DOMAINS:-"airline retail instore delivery ota telecom"}

# ========== 断点续训配置 ==========
# 从指定 domain 开始（跳过之前的 domain）
START_FROM_DOMAIN=${START_FROM_DOMAIN:-""}

# 从指定 checkpoint 恢复（用于单个 domain 内的断点续训）
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}

echo "=============================================="
echo "Single Domain Training for FWT Baseline"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Domains: ${DOMAINS}"
echo "Steps per domain: ${NUM_STEPS}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
echo "Skip intermediate eval: ${SKIP_INTERMEDIATE_EVAL}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "W&B project: ${WANDB_PROJECT:-'(disabled)'}"
echo "W&B mode: ${WANDB_MODE}"
if [ -n "${START_FROM_DOMAIN}" ]; then
    echo "Start from domain: ${START_FROM_DOMAIN}"
fi
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
echo "=============================================="

# 是否已经找到起始 domain
FOUND_START_DOMAIN=false
if [ -z "${START_FROM_DOMAIN}" ]; then
    FOUND_START_DOMAIN=true
fi

# 依次训练每个 domain
for DOMAIN in ${DOMAINS}; do
    # 检查是否需要跳过
    if [ "${FOUND_START_DOMAIN}" = "false" ]; then
        if [ "${DOMAIN}" = "${START_FROM_DOMAIN}" ]; then
            FOUND_START_DOMAIN=true
        else
            echo ""
            echo "[Skip] Domain ${DOMAIN} - already completed"
            continue
        fi
    fi

    echo ""
    echo "=============================================="
    echo "Training domain: ${DOMAIN}"
    echo "=============================================="

    LOG_FILE=${LOG_DIR}/train_single_${DOMAIN}_$(date +"%Y%m%d_%H%M%S").log
    DOMAIN_LOG_DIR=${OUTPUT_DIR}/${DOMAIN}

    # 构建命令参数
    CMD_ARGS=(
        --domain ${DOMAIN}
        --model_name_or_path ${MODEL_PATH}
        --model_dtype bfloat16
        --output_dir ${OUTPUT_DIR}
        --num_steps_per_task ${NUM_STEPS}
        --batch_size_per_gpu ${BATCH_SIZE}
        --gradient_accumulation_steps 8
        --learning_rate ${LEARNING_RATE}
        --num_samples_per_prompt ${NUM_SAMPLES}
        --kl_coef 0.05
        --clip_range 0.1
        --max_grad_norm 1.0
        --num_eval_tasks 5
        --num_eval_samples 1
        --pass_at_k 1
        --eval_interval 50
        --seed ${SEED}
        --max_new_tokens 1024
        --user_api_base ${USER_API_BASE}
        --user_model ${USER_MODEL}
        --use_flash_attention
        --gradient_checkpointing
        --checkpoint_interval ${CHECKPOINT_INTERVAL}
    )

    # 添加 API key 参数（支持多 key 或单 key）
    if [ -n "${USER_API_KEYS}" ]; then
        CMD_ARGS+=(--user_api_keys ${USER_API_KEYS})
    else
        CMD_ARGS+=(--user_api_key ${USER_API_KEY})
    fi

    # 添加可选参数
    if [ "${SKIP_INTERMEDIATE_EVAL}" = "true" ]; then
        CMD_ARGS+=(--skip_intermediate_eval)
    fi

    # 添加 early stopping 参数
    if [ "${EARLY_STOPPING_PATIENCE}" -gt 0 ] 2>/dev/null; then
        CMD_ARGS+=(--early_stopping_patience ${EARLY_STOPPING_PATIENCE})
        CMD_ARGS+=(--early_stopping_threshold ${EARLY_STOPPING_THRESHOLD})
    fi

    # 添加 W&B 参数
    if [ -n "${WANDB_PROJECT}" ]; then
        CMD_ARGS+=(--wandb_project ${WANDB_PROJECT})
    fi
    if [ -n "${WANDB_ENTITY}" ]; then
        CMD_ARGS+=(--wandb_entity ${WANDB_ENTITY})
    fi
    if [ -n "${WANDB_MODE}" ]; then
        CMD_ARGS+=(--wandb_mode ${WANDB_MODE})
    fi
    CMD_ARGS+=(--wandb_run_name single_${DOMAIN})

    # 如果是起始 domain 且有 checkpoint，添加恢复参数
    if [ "${DOMAIN}" = "${START_FROM_DOMAIN}" ] && [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        CMD_ARGS+=(--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT})
        # 只在第一个 domain 使用，之后清除
        RESUME_FROM_CHECKPOINT=""
    fi

    accelerate launch --config_file ${ACCELERATE_CONFIG} \
        /home/houzhiyan/agent-ywy/agent/AGentCL/src/tau2/scripts/train_single_domain.py \
        "${CMD_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"

    echo "Domain ${DOMAIN} completed. Log: ${LOG_FILE}"
done

echo ""
echo "=============================================="
echo "All single domain training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "To use these checkpoints for FWT calculation:"
echo "  SINGLE_DOMAIN_CHECKPOINT_DIR=${OUTPUT_DIR} bash train.sh"
echo ""
echo "To resume from a specific domain:"
echo "  START_FROM_DOMAIN=retail bash train_single_domain.sh"
echo ""
echo "To resume from a checkpoint within a domain:"
echo "  START_FROM_DOMAIN=retail RESUME_FROM_CHECKPOINT=path/to/checkpoint bash train_single_domain.sh"
echo "=============================================="
