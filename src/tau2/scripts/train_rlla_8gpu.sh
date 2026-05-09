#!/usr/bin/env bash
# ToolRL-style GRPO training on rlla_by_domain data (8-GPU DeepSpeed ZeRO-2)
#
# Usage:
#   bash train_rlla_8gpu.sh                          # train all domains
#   DOMAIN=airline bash train_rlla_8gpu.sh           # single domain
#   DOMAIN=api_bank NUM_STEPS=200 bash train_rlla_8gpu.sh
#
# 断点续训:
#   DOMAIN=airline RESUME_FROM_CHECKPOINT=logs/rlla/checkpoint_task_0_step_100 bash train_rlla_8gpu.sh

set -euo pipefail

# ========== 环境变量 ==========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/houzhiyan/agent-ywy/agent/AGentCL/src:${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# vLLM: force V0 engine (V1 spawns EngineCore subprocesses that deadlock
# when 8 DeepSpeed ranks all initialise NCCL simultaneously)
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export VLLM_USE_V1=0

# ========== 路径配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/houzhiyan/agent-ywy/agent/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_8gpu_bf16.yaml"}
TRAIN_SCRIPT="${SCRIPT_DIR}/train_rlla.py"
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/data/rlla_by_domain"}

LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/rlla"}
mkdir -p "${LOG_DIR}"

# ========== 模型 ==========
MODEL_PATH=${MODEL_PATH:-"/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct"}
MODEL_DTYPE=${MODEL_DTYPE:-"bfloat16"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}

# ========== GRPO 超参 (对齐 ToolRL run_grpo.sh) ==========
NUM_STEPS=${NUM_STEPS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-4}          # rollout.n=4
BATCH_SIZE=${BATCH_SIZE:-64}           # prompts per step per GPU
KL_COEF=${KL_COEF:-0.001}
CLIP_RANGE=${CLIP_RANGE:-0.2}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-128}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-32}
PPO_EPOCHS=${PPO_EPOCHS:-1}
LR=${LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ========== 早停 ==========
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-3.5}

# ========== 日志 ==========
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
CKPT_INTERVAL=${CKPT_INTERVAL:-100}
SKIP_EVAL=${SKIP_EVAL:-""}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}

# ========== vLLM ==========
VLLM_TP=${VLLM_TP:-1}
VLLM_MEM=${VLLM_MEM:-0.15}
VLLM_EAGER=${VLLM_EAGER:-"true"}   # must be true with weight offload (no CUDA graphs)

# ========== W&B ==========
WANDB_PROJECT=${WANDB_PROJECT:-"rlla_grpo"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-"online"}

# ========== 要训练的 domain ==========
# 可以用 DOMAIN=airline 只训练一个，或者 DOMAINS="airline retail" 训练多个
# 默认训练全部
ALL_DOMAINS="airline retail api_bank bamboogle"
DOMAINS=${DOMAINS:-${DOMAIN:-${ALL_DOMAINS}}}

# ========== 训练 ==========
for DOMAIN_NAME in ${DOMAINS}; do
    DATA_DIR="${DATA_ROOT}/${DOMAIN_NAME}"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "[WARN] Data dir not found: ${DATA_DIR}, skipping ${DOMAIN_NAME}"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/${DOMAIN_NAME}_${TIMESTAMP}.log"
    RUN_NAME="rlla_${DOMAIN_NAME}_${TIMESTAMP}"

    echo "========================================"
    echo "Training domain: ${DOMAIN_NAME}"
    echo "  data_dir:  ${DATA_DIR}"
    echo "  model:     ${MODEL_PATH}"
    echo "  steps:     ${NUM_STEPS}"
    echo "  log:       ${LOG_FILE}"
    echo "========================================"

    SKIP_EVAL_FLAG=""
    if [ -n "${SKIP_EVAL}" ]; then
        SKIP_EVAL_FLAG="--skip_intermediate_eval"
    fi

    RESUME_FLAG=""
    if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        RESUME_FLAG="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
    fi

    WANDB_ENTITY_FLAG=""
    if [ -n "${WANDB_ENTITY}" ]; then
        WANDB_ENTITY_FLAG="--wandb_entity ${WANDB_ENTITY}"
    fi

    accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        "${TRAIN_SCRIPT}" \
        --data_dir "${DATA_DIR}" \
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
        --log_dir "${LOG_DIR}/${DOMAIN_NAME}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --checkpoint_interval "${CKPT_INTERVAL}" \
        --vllm_tensor_parallel_size "${VLLM_TP}" \
        --vllm_gpu_memory_utilization "${VLLM_MEM}" \
        --vllm_enforce_eager \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${RUN_NAME}" \
        --wandb_mode "${WANDB_MODE}" \
        --seed 42 \
        ${WANDB_ENTITY_FLAG} \
        ${SKIP_EVAL_FLAG} \
        ${RESUME_FLAG} \
        2>&1 | tee "${LOG_FILE}"

    echo "[Done] ${DOMAIN_NAME} -> ${LOG_FILE}"
done

echo ""
echo "All domains done."
