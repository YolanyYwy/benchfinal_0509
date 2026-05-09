#!/usr/bin/env bash
# Sequential baseline: train domains one after another, each domain starts
# from where the previous one left off (no forgetting prevention).
#
# Usage:
#   bash train_rlla_sequential_1gpu.sh
#   DOMAINS="bamboogle api_bank airline retail" bash train_rlla_sequential_1gpu.sh
#   GPU=2 bash train_rlla_sequential_1gpu.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/src:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

export NCCL_DEBUG=WARN
export VLLM_USE_V1=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/haoxiangzhao/ywy/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_1gpu_bf16_seq.yaml"}
TRAIN_SCRIPT="${SCRIPT_DIR}/train_rlla.py"
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/data/rlla_by_domain"}

LOG_DIR=${LOG_DIR:-"/9950backfile/ywy/logs/rlla_sequential"}
mkdir -p "${LOG_DIR}"

# ========== 模型 ==========
MODEL_PATH=${MODEL_PATH:-"/9950backfile/qwen/Qwen2.5-3B-Instruct"}
BASE_MODEL_PATH="${MODEL_PATH}"
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

# ========== vLLM ==========
VLLM_TP=${VLLM_TP:-1}
VLLM_MEM=${VLLM_MEM:-0.5}

# ========== W&B ==========
WANDB_PROJECT=${WANDB_PROJECT:-"rlla_sequential"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-"online"}

# ========== 要训练的 domain ==========
ALL_DOMAINS="dev_math ecommerce entertainment finance weather_geo health social_comm bamboogle api_bank toolrl_4k glaive toolace airline retail"
DOMAINS=${DOMAINS:-${ALL_DOMAINS}}

# ========== Sequential 训练 ==========
# CURRENT_MODEL 从原始模型开始，每个 domain 训完后更新为该 domain 的 checkpoint
CURRENT_MODEL="${BASE_MODEL_PATH}"

echo "========================================"
echo "Sequential Baseline Training"
echo "  Base model: ${BASE_MODEL_PATH}"
echo "  Domains:    ${DOMAINS}"
echo "  Steps/domain: ${NUM_STEPS}"
echo "  Log dir:    ${LOG_DIR}"
echo "========================================"

for DOMAIN_NAME in ${DOMAINS}; do
    DATA_DIR="${DATA_ROOT}/${DOMAIN_NAME}"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "[WARN] Data dir not found: ${DATA_DIR}, skipping ${DOMAIN_NAME}"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/${DOMAIN_NAME}_${TIMESTAMP}.log"
    RUN_NAME="seq_${DOMAIN_NAME}_${TIMESTAMP}"
    DOMAIN_LOG_DIR="${LOG_DIR}/${DOMAIN_NAME}"

    echo "========================================"
    echo "Training domain: ${DOMAIN_NAME}"
    echo "  GPU:       ${GPU}"
    echo "  model:     ${CURRENT_MODEL}"
    echo "  data_dir:  ${DATA_DIR}"
    echo "  steps:     ${NUM_STEPS}"
    echo "  log:       ${LOG_FILE}"
    echo "========================================"

    SKIP_EVAL_FLAG=""
    if [ -n "${SKIP_EVAL}" ]; then
        SKIP_EVAL_FLAG="--skip_intermediate_eval"
    fi

    WANDB_ENTITY_FLAG=""
    if [ -n "${WANDB_ENTITY}" ]; then
        WANDB_ENTITY_FLAG="--wandb_entity ${WANDB_ENTITY}"
    fi

    accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        "${TRAIN_SCRIPT}" \
        --data_dir "${DATA_DIR}" \
        --model_name_or_path "${CURRENT_MODEL}" \
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
        --log_dir "${DOMAIN_LOG_DIR}" \
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
        2>&1 | tee "${LOG_FILE}"

    # 训完后把 checkpoint 路径传给下一个 domain
    CKPT_PATH="${DOMAIN_LOG_DIR}/model"
    if [ -d "${CKPT_PATH}" ]; then
        CURRENT_MODEL="${CKPT_PATH}"
        echo "[Done] ${DOMAIN_NAME} -> next model: ${CURRENT_MODEL}"
    else
        echo "[WARN] Checkpoint not found at ${CKPT_PATH}, next domain will use: ${CURRENT_MODEL}"
    fi

    # 等待 GPU 彻底释放（vLLM/DeepSpeed 残留显存）
    echo "[Cleanup] waiting for GPU memory to be released..."
    sleep 30
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('[Cleanup] GPU cache cleared')" 2>/dev/null || true
    # 打印当前 GPU 显存使用情况，方便排查
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
done

echo ""
echo "Sequential training done."
echo "Final model: ${CURRENT_MODEL}"
