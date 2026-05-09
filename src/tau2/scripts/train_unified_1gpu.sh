#!/usr/bin/env bash
# Unified GRPO training launcher (1-GPU).
#
# One accelerate launch, one process — avoids repeated init cost across domains.
#
# Modes:
#   MODE=single       每个 domain 前 reload 原始模型 + reset optimizer (single-domain baseline)
#   MODE=sequential   domain 间权重接力 (sequential baseline)
#   MODE=replay       启用 ReplayCL
#   MODE=ewc          启用 EWCCL
#   MODE=fusion       启用 ModelFusionCL
#   MODE=progressive  启用 ProgressiveNetsCL
#   MODE=adaptive_replay / online_ewc / ewc_pp / adaptive_fusion / dynamic_expansion
#
# Usage:
#   MODE=single bash train_unified_1gpu.sh
#   MODE=sequential DOMAINS="bamboogle api_bank airline retail" bash train_unified_1gpu.sh
#   MODE=ewc GPU=2 NUM_STEPS=300 bash train_unified_1gpu.sh
#   MODE=sequential SKIP_FIRST_N=2 MODEL_PATH=/last/finished/model/path bash train_unified_1gpu.sh

set -euo pipefail

# ========== 环境变量 ==========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export VLLM_USE_V1=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 指定 GPU
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

# ========== 模式 ==========
MODE=${MODE:-"sequential"}

# ========== 路径 ==========
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"${PROJECT_ROOT}/src/tau2/continual_learning/continual_learning/accelerate_1gpu_bf16.yaml"}
TRAIN_SCRIPT="${SCRIPT_DIR}/train_unified.py"
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/data/rlla_by_domain"}
LOG_DIR=${LOG_DIR:-"/9950backfile/ywy/logs/unified_${MODE}"}
mkdir -p "${LOG_DIR}"

# ========== 模型 ==========
MODEL_PATH=${MODEL_PATH:-"/9950backfile/qwen/Qwen2.5-3B-Instruct"}
MODEL_DTYPE=${MODEL_DTYPE:-"bfloat16"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}

# ========== GRPO 超参 ==========
NUM_STEPS=${NUM_STEPS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
KL_COEF=${KL_COEF:-0.001}
CLIP_RANGE=${CLIP_RANGE:-0.2}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-32}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-8}
PPO_EPOCHS=${PPO_EPOCHS:-1}
LR=${LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ========== 早停 ==========
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-3.5}

# ========== 日志 ==========
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
CKPT_INTERVAL=${CKPT_INTERVAL:-0}
SKIP_EVAL=${SKIP_EVAL:-"1"}
SKIP_FIRST_N=${SKIP_FIRST_N:-0}

# ========== vLLM ==========
VLLM_TP=${VLLM_TP:-1}
VLLM_MEM=${VLLM_MEM:-0.4}

# ========== W&B ==========
WANDB_PROJECT=${WANDB_PROJECT:-"rlla_unified_${MODE}"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-"online"}

# ========== Domains ==========
ALL_DOMAINS="dev_math ecommerce entertainment finance weather_geo health social_comm bamboogle api_bank toolrl_4k glaive toolace airline retail"
DOMAINS=${DOMAINS:-${ALL_DOMAINS}}

# ========== 打印配置 ==========
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
RUN_NAME="${MODE}_${TIMESTAMP}"

echo "========================================"
echo "Unified GRPO Training"
echo "  mode:       ${MODE}"
echo "  GPU:        ${GPU}"
echo "  model:      ${MODEL_PATH}"
echo "  domains:    ${DOMAINS}"
echo "  steps/dom:  ${NUM_STEPS}"
echo "  log_dir:    ${LOG_DIR}"
echo "  log_file:   ${LOG_FILE}"
echo "========================================"

# ========== 组装参数 ==========
SKIP_EVAL_FLAG=""
if [ -n "${SKIP_EVAL}" ]; then
    SKIP_EVAL_FLAG="--skip_intermediate_eval"
fi

WANDB_ENTITY_FLAG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_FLAG="--wandb_entity ${WANDB_ENTITY}"
fi

# ========== 单次 accelerate launch ==========
accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --mode "${MODE}" \
    --data_root "${DATA_ROOT}" \
    --domains ${DOMAINS} \
    --model_name_or_path "${MODEL_PATH}" \
    --model_dtype "${MODEL_DTYPE}" \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_samples_per_prompt "${NUM_SAMPLES}" \
    --batch_size_per_gpu "${BATCH_SIZE}" \
    --kl_coef "${KL_COEF}" \
    --clip_range "${CLIP_RANGE}" \
    --entropy_coeff "${ENTROPY_COEFF}" \
    --ppo_mini_batch_size "${PPO_MINI_BATCH}" \
    --ppo_micro_batch_size "${PPO_MICRO_BATCH}" \
    --ppo_epochs "${PPO_EPOCHS}" \
    --num_steps_per_task "${NUM_STEPS}" \
    --learning_rate "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --early_stopping_patience "${EARLY_STOP_PATIENCE}" \
    --early_stopping_threshold "${EARLY_STOP_THRESHOLD}" \
    --log_dir "${LOG_DIR}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --checkpoint_interval "${CKPT_INTERVAL}" \
    --vllm_tensor_parallel_size "${VLLM_TP}" \
    --vllm_gpu_memory_utilization "${VLLM_MEM}" \
    --vllm_enforce_eager \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_mode "${WANDB_MODE}" \
    --skip_first_n "${SKIP_FIRST_N}" \
    --seed 42 \
    ${WANDB_ENTITY_FLAG} \
    ${SKIP_EVAL_FLAG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Unified training done. Log: ${LOG_FILE}"
