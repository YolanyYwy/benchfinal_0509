#!/usr/bin/env bash

# API-Bank 单轮 Tool-Use GRPO 训练脚本（8卡 DeepSpeed ZeRO-2）
#
# 用法：
#   bash train_apibank_8gpu.sh
#
# 支持环境变量覆盖所有参数，例如：
#   NUM_STEPS=1000 WANDB_PROJECT=my_proj bash train_apibank_8gpu.sh
#
# 断点续训：
#   RESUME_FROM_CHECKPOINT=logs/apibank/checkpoint_task_0_step_200 bash train_apibank_8gpu.sh

set -euo pipefail

# ========== 环境变量 ==========

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/houzhiyan/agent-ywy/agent/AGentCL/src:${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0

# PyTorch 分布式
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# vLLM: force V0 engine (V1 spawns EngineCore subprocesses that deadlock
# when 8 DeepSpeed ranks all initialise NCCL simultaneously)
export VLLM_USE_V1=0


# ========== 路径配置 ==========

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/houzhiyan/agent-ywy/agent/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_8gpu_bf16.yaml"}
TRAIN_SCRIPT="${SCRIPT_DIR}/train_apibank.py"

LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/run_logs"}
mkdir -p "${LOG_DIR}"

# ========== 训练参数（可通过环境变量覆盖）==========

# 模型
MODEL_PATH=${MODEL_PATH:-"/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct"}
MODEL_DTYPE=${MODEL_DTYPE:-"bfloat16"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}      # ToolRL max_response_length=1024
TEMPERATURE=${TEMPERATURE:-1.0}             # ToolRL rollout.temperature=1.0

# API-Bank 数据
LEVELS=${LEVELS:-"1 2 3"}
MAX_SAMPLES=${MAX_SAMPLES:-""}

# GRPO 超参 — 完全对齐 ToolRL run_grpo.sh
NUM_STEPS=${NUM_STEPS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-4}               # ToolRL rollout.n=4
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64} # ToolRL 512 total / 8 GPUs = 64
KL_COEF=${KL_COEF:-0.001}                  # ToolRL algorithm.kl_ctrl.kl_coef=0.001
LEARNING_RATE=${LEARNING_RATE:-1e-6}        # ToolRL actor.optim.lr=1e-6
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}  # ToolRL actor.ppo_mini_batch_size=128
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-32} # ToolRL ppo_micro_batch_size
PPO_EPOCHS=${PPO_EPOCHS:-1}                 # ToolRL actor.ppo_epochs=1
CLIP_RANGE=${CLIP_RANGE:-0.2}              # ToolRL actor.clip_ratio=0.2
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}      # ToolRL actor.entropy_coeff=0.001
WARMUP_STEPS=${WARMUP_STEPS:-10}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# vLLM 配置 (per-rank tp=1, persistent engine with weight offload)
VLLM_TP=${VLLM_TP:-1}                          # ignored — always tp=1 per rank
VLLM_MEM=${VLLM_MEM:-0.15}                     # small KV cache, leaves room for backward
VLLM_EAGER=${VLLM_EAGER:-"true"}               # must be true with weight offload

# 早停
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-20}
EARLY_STOPPING_THRESHOLD=${EARLY_STOPPING_THRESHOLD:-3.5}  # reward 接近 4.0 时停止

# 评估 & Checkpoint
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
SKIP_INTERMEDIATE_EVAL=${SKIP_INTERMEDIATE_EVAL:-"false"}

# 输出目录
OUTPUT_LOG_DIR=${OUTPUT_LOG_DIR:-"${PROJECT_ROOT}/logs/apibank"}

# W&B
WANDB_PROJECT=${WANDB_PROJECT:-"Agent"}
WANDB_ENTITY=${WANDB_ENTITY:-"yuwy22-tsinghua-university"}
WANDB_MODE=${WANDB_MODE:-"online"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"apibank_8gpu_$(date +%Y%m%d_%H%M%S)"}

# 断点续训
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}

# 随机种子
SEED=${SEED:-42}

# ========== 打印配置 ==========

echo "=============================================="
echo "  API-Bank GRPO Training  (8-GPU, ToolRL+vLLM)"
echo "=============================================="
echo "  Model        : ${MODEL_PATH}"
echo "  Levels       : ${LEVELS}"
echo "  Steps        : ${NUM_STEPS}"
echo "  B (prompts)  : ${BATCH_SIZE_PER_GPU}  (total B*G = $((BATCH_SIZE_PER_GPU * NUM_SAMPLES)) trajs/step)"
echo "  G (rollout.n): ${NUM_SAMPLES}"
echo "  LR           : ${LEARNING_RATE}"
echo "  KL coef      : ${KL_COEF}"
echo "  clip_range   : ${CLIP_RANGE}"
echo "  ppo_mini_bsz : ${PPO_MINI_BATCH_SIZE}"
echo "  ppo_epochs   : ${PPO_EPOCHS}"
echo "  vLLM tp      : ${VLLM_TP}  mem=${VLLM_MEM}"
echo "  Output       : ${OUTPUT_LOG_DIR}"
echo "  W&B project  : ${WANDB_PROJECT:-'(disabled)'}"
echo "  W&B run      : ${WANDB_RUN_NAME}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "  Resume from  : ${RESUME_FROM_CHECKPOINT}"
fi
echo "=============================================="

# ========== 构建命令参数 ==========

CMD_ARGS=(
    --model_name_or_path      "${MODEL_PATH}"
    --model_dtype             "${MODEL_DTYPE}"
    --temperature             "${TEMPERATURE}"
    --max_new_tokens          "${MAX_NEW_TOKENS}"
    --levels                  ${LEVELS}
    --num_steps_per_task      "${NUM_STEPS}"
    --num_samples_per_prompt  "${NUM_SAMPLES}"
    --batch_size_per_gpu      "${BATCH_SIZE_PER_GPU}"
    --kl_coef                 "${KL_COEF}"
    --learning_rate           "${LEARNING_RATE}"
    --ppo_mini_batch_size     "${PPO_MINI_BATCH_SIZE}"
    --ppo_micro_batch_size    "${PPO_MICRO_BATCH_SIZE}"
    --ppo_epochs              "${PPO_EPOCHS}"
    --clip_range              "${CLIP_RANGE}"
    --entropy_coeff           "${ENTROPY_COEFF}"
    --warmup_steps            "${WARMUP_STEPS}"
    --max_grad_norm           "${MAX_GRAD_NORM}"
    --vllm_tensor_parallel_size   "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_MEM}"
    --early_stopping_patience  "${EARLY_STOPPING_PATIENCE}"
    --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}"
    --eval_interval           "${EVAL_INTERVAL}"
    --checkpoint_interval     "${CHECKPOINT_INTERVAL}"
    --log_dir                 "${OUTPUT_LOG_DIR}"
    --seed                    "${SEED}"
)

# 可选：限制样本数（调试用）
if [ -n "${MAX_SAMPLES}" ]; then
    CMD_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

# vLLM enforce_eager (must be true with weight offload)
if [ "${VLLM_EAGER}" = "true" ]; then
    CMD_ARGS+=(--vllm_enforce_eager)
fi

# 跳过中间评估
if [ "${SKIP_INTERMEDIATE_EVAL}" = "true" ]; then
    CMD_ARGS+=(--skip_intermediate_eval)
fi

# 断点续训
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# W&B
if [ -n "${WANDB_PROJECT}" ]; then
    CMD_ARGS+=(--wandb_project  "${WANDB_PROJECT}")
    CMD_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
fi
if [ -n "${WANDB_ENTITY}" ]; then
    CMD_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [ -n "${WANDB_MODE}" ]; then
    CMD_ARGS+=(--wandb_mode "${WANDB_MODE}")
fi

# ========== 启动训练 ==========

LOG_FILE="${LOG_DIR}/train_apibank_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: ${LOG_FILE}"
echo ""

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    "${CMD_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=============================================="
echo "Training complete."
echo "Checkpoints : ${OUTPUT_LOG_DIR}"
echo "Log         : ${LOG_FILE}"
echo ""
echo "To resume from a checkpoint:"
echo "  RESUME_FROM_CHECKPOINT=${OUTPUT_LOG_DIR}/checkpoint_task_0_step_XXX bash train_apibank_8gpu.sh"
echo "=============================================="
