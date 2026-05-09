#!/usr/bin/env bash
# ToolRL-style GRPO training on rlla_by_domain data (1-GPU)
#
# Usage:
#   bash train_rlla_1gpu.sh                          # train all domains
#   DOMAIN=toolrl_4k bash train_rlla_1gpu.sh         # single domain
#   DOMAIN=toolrl_4k NUM_STEPS=200 bash train_rlla_1gpu.sh
#   GPU=4 DOMAIN=toolrl_4k bash train_rlla_1gpu.sh   # specify which GPU
#
# 断点续训:
#   DOMAIN=airline RESUME_FROM_CHECKPOINT=logs/rlla/checkpoint_task_0_step_100 bash train_rlla_1gpu.sh

set -euo pipefail

# ========== 环境变量 ==========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/src:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 指定使用哪张卡，默认 0，可用 GPU=4 bash train_rlla_1gpu.sh 覆盖
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

export NCCL_DEBUG=WARN
export VLLM_USE_V1=0

# ========== 路径配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/haoxiangzhao/ywy/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_1gpu_bf16.yaml"}
TRAIN_SCRIPT="${SCRIPT_DIR}/train_rlla.py"
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/data/rlla_by_domain"}

LOG_DIR=${LOG_DIR:-"/9950backfile/ywy/logs/rlla"}
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
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
SKIP_EVAL=${SKIP_EVAL:-""}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}

# ========== vLLM ==========
# 单卡：DS model 和 vLLM 交替使用同一张卡
# DS model (3B bf16) ~6GB，vLLM KV cache 用剩余显存
# A800 80GB: 给 vLLM 0.5 足够；小卡(24GB)用 0.4
VLLM_TP=${VLLM_TP:-1}
VLLM_MEM=${VLLM_MEM:-0.5}
VLLM_EAGER=${VLLM_EAGER:-"true"}

# ========== W&B ==========
WANDB_PROJECT=${WANDB_PROJECT:-"rlla_grpo"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-"online"}

# ========== 要训练的 domain ==========
ALL_DOMAINS="dev_math ecommerce entertainment finance weather_geo health social_comm bamboogle api_bank toolrl_4k glaive toolace airline retail"
DOMAINS=${DOMAINS:-${ALL_DOMAINS}}

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
    echo "  GPU:       ${GPU}"
    echo "  data_dir:  ${DATA_DIR}"
    echo "  model:     ${MODEL_PATH}"
    echo "  steps:     ${NUM_STEPS}  batch=${BATCH_SIZE}  G=${NUM_SAMPLES}"
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

    # 等待 GPU 彻底释放（vLLM/DeepSpeed 残留显存）
    echo "[Cleanup] waiting for GPU memory to be released..."
    sleep 30
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('[Cleanup] GPU cache cleared')" 2>/dev/null || true
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
done

echo ""
echo "All domains done."
