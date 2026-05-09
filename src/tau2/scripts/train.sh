#!/usr/bin/env bash

# 持续学习训练脚本（sequential baseline / CL 训练）
# 支持分阶段运行：
#   - PHASE=1 bash train.sh      # Zero-shot 评估
#   - PHASE=1.5 bash train.sh    # 单任务 baseline 评估
#   - PHASE=2 bash train.sh      # 持续学习训练（sequential baseline 只需这个）
#   - PHASE=3 bash train.sh      # 最终评估
#   - PHASE=all bash train.sh    # 完整流程（默认）
#
# 支持断点续训：
#   - RESUME_FROM_TASK=2 bash train.sh
#   - RESUME_FROM_CHECKPOINT=logs/xxx/checkpoint_task_1_step_50 bash train.sh

set -euo pipefail

# PyTorch 内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 动态计算项目根目录，避免写死路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# CUDA 设备顺序 - 确保 rank 和 GPU 映射一致
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 指定使用哪张卡，默认 0
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

# NCCL 配置 - 解决 "Guessing device ID" 警告
export NCCL_DEBUG=WARN
export VLLM_USE_V1=0

# 日志目录
LOG_DIR=${LOG_DIR:-"/9950backfile/ywy/logs/rlla_baseline"}
mkdir -p "${LOG_DIR}"

# ========== 配置参数（可通过环境变量覆盖）==========

# 阶段选择：1, 1.5, 2, 3, all
PHASE=${PHASE:-"all"}

# 模型配置 - 本地 Qwen3-4B 模型路径
MODEL_PATH=${MODEL_PATH:-"/home/haoxiangzhao/ywy/qwen/Qwen2.5-3B-Instruct"}

# 训练配置（单卡默认值，与 train_rlla_1gpu.sh 对齐）
NUM_STEPS=${NUM_STEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
NUM_SAMPLES=${NUM_SAMPLES:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
SEED=${SEED:-42}

# CL 算法：sequential, replay, ewc, fusion 等
CL_ALGORITHM=${CL_ALGORITHM:-"sequential"}

# Task 顺序（最后一个用于泛化测试）
TASK_ORDER=${TASK_ORDER:-"bamboogle api_bank toolrl_4k airline retail"}

# 日志目录
OUTPUT_LOG_DIR=${OUTPUT_LOG_DIR:-"/9950backfile/ywy/logs/rlla_baseline"}

# pass@k 评估配置
PASS_AT_K=${PASS_AT_K:-4}
NUM_EVAL_SAMPLES=${NUM_EVAL_SAMPLES:-5}
NUM_EVAL_TASKS=${NUM_EVAL_TASKS:-20}

# 单任务 baseline checkpoint 目录（用于 FWT 计算）
SINGLE_DOMAIN_CHECKPOINT_DIR=${SINGLE_DOMAIN_CHECKPOINT_DIR:-""}

# Checkpoint 保存间隔（每隔多少 step 保存一次）
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-101}

# 是否跳过中间评估
SKIP_INTERMEDIATE_EVAL=${SKIP_INTERMEDIATE_EVAL:-"true"}

# ========== 断点续训配置 ==========
# 从第几个 task 开始（0-indexed）
RESUME_FROM_TASK=${RESUME_FROM_TASK:-0}

# 从指定 checkpoint 恢复（优先级高于 RESUME_FROM_TASK）
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-""}

# 从指定模型恢复（用于 phase 3）
RESUME_FROM=${RESUME_FROM:-""}

# 从评估阶段恢复：指定从哪个 domain 开始评估
RESUME_FROM_EVAL_DOMAIN=${RESUME_FROM_EVAL_DOMAIN:-""}

# 跳过 resume_from_task 的训练，直接从评估开始
SKIP_TRAINING_FOR_RESUME_TASK=${SKIP_TRAINING_FOR_RESUME_TASK:-"false"}

# ========== 分阶段结果文件路径 ==========
PHASE1_RESULTS=${PHASE1_RESULTS:-"${OUTPUT_LOG_DIR}/phase1_zero_shot.json"}
PHASE1_5_RESULTS=${PHASE1_5_RESULTS:-"${OUTPUT_LOG_DIR}/phase1_5_single_domain.json"}

# User API 配置（用于 User Simulator — 紫东太初 API）
USER_API_BASE=${USER_API_BASE:-"https://cloud.zidongtaichu.com/maas/v1"}
USER_API_KEY=${USER_API_KEY:-"vp8ggmuy102xmtpcyf9enr3g"}
USER_MODEL=${USER_MODEL:-"gpt_oss_120b"}

USER_API_KEYS=${USER_API_KEYS:-""}

# Accelerate 配置文件（默认单卡，多卡可通过环境变量覆盖）
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"/home/haoxiangzhao/ywy/AGentCL/src/tau2/continual_learning/continual_learning/accelerate_1gpu_bf16.yaml"}

# W&B 配置
WANDB_PROJECT=${WANDB_PROJECT:-"agent"}

# 根据 phase 设置日志文件名
LOG_FILE=${LOG_DIR}/train_phase${PHASE}_$(date +"%Y%m%d_%H%M%S").log

echo "=============================================="
echo "Continual Learning Training"
echo "=============================================="
echo "Phase: ${PHASE}"
echo "Model: ${MODEL_PATH}"
echo "CL Algorithm: ${CL_ALGORITHM}"
echo "Task Order: ${TASK_ORDER}"
echo "Steps per task: ${NUM_STEPS}"
echo "Output dir: ${OUTPUT_LOG_DIR}"
echo "Log file: ${LOG_FILE}"

case ${PHASE} in
    "1")
        echo "Mode: Zero-shot evaluation"
        ;;
    "1.5")
        echo "Mode: Single-domain baseline evaluation"
        echo "Single domain checkpoints: ${SINGLE_DOMAIN_CHECKPOINT_DIR}"
        ;;
    "2")
        echo "Mode: Continual learning training"
        echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
        echo "Skip intermediate eval: ${SKIP_INTERMEDIATE_EVAL}"
        ;;
    "3")
        echo "Mode: Final evaluation"
        ;;
    "all")
        echo "Mode: Full pipeline"
        ;;
esac

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
elif [ "${RESUME_FROM_TASK}" -gt 0 ]; then
    echo "Resume from task: ${RESUME_FROM_TASK}"
fi
if [ -n "${RESUME_FROM}" ]; then
    echo "Resume model from: ${RESUME_FROM}"
fi
echo "=============================================="

# 构建命令参数
CMD_ARGS=(
    --phase ${PHASE}
    --model_name_or_path ${MODEL_PATH}
    --model_dtype bfloat16
    --batch_size_per_gpu ${BATCH_SIZE}
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --num_steps_per_task ${NUM_STEPS}
    --learning_rate ${LEARNING_RATE}
    --kl_coef 0.01
    --clip_range 0.2
    --max_grad_norm 1.0
    --cl_algorithm ${CL_ALGORITHM}
    --task_order ${TASK_ORDER}
    --log_dir ${OUTPUT_LOG_DIR}
    --wandb_project ${WANDB_PROJECT}
    --trajectory_log_interval 1
    --num_samples_per_prompt ${NUM_SAMPLES}
    --use_flash_attention
    --gradient_checkpointing
    --max_new_tokens 1024
    --seed ${SEED}
    --pass_at_k ${PASS_AT_K}
    --num_eval_samples ${NUM_EVAL_SAMPLES}
    --num_eval_tasks ${NUM_EVAL_TASKS}
    --checkpoint_interval ${CHECKPOINT_INTERVAL}
    --user_api_base ${USER_API_BASE}
    --user_api_key ${USER_API_KEY}
    --user_model ${USER_MODEL}
)
# 添加多个 API keys（如果设置了）
if [ -n "${USER_API_KEYS}" ]; then
    CMD_ARGS+=(--user_api_keys ${USER_API_KEYS})
fi
# 添加可选参数
if [ "${SKIP_INTERMEDIATE_EVAL}" = "true" ]; then
    CMD_ARGS+=(--skip_intermediate_eval)
fi

if [ -n "${SINGLE_DOMAIN_CHECKPOINT_DIR}" ]; then
    CMD_ARGS+=(--single_domain_checkpoint_dir ${SINGLE_DOMAIN_CHECKPOINT_DIR})
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD_ARGS+=(--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT})
fi

if [ "${RESUME_FROM_TASK}" -gt 0 ]; then
    CMD_ARGS+=(--resume_from_task ${RESUME_FROM_TASK})
fi

if [ -n "${RESUME_FROM}" ]; then
    CMD_ARGS+=(--resume_from ${RESUME_FROM})
fi

# 评估阶段恢复参数
if [ -n "${RESUME_FROM_EVAL_DOMAIN}" ]; then
    CMD_ARGS+=(--resume_from_eval_domain ${RESUME_FROM_EVAL_DOMAIN})
fi

if [ "${SKIP_TRAINING_FOR_RESUME_TASK}" = "true" ]; then
    CMD_ARGS+=(--skip_training_for_resume_task)
fi

# 加载之前阶段的结果（用于 phase 1.5, 2, 3）
if [ -f "${PHASE1_RESULTS}" ]; then
    CMD_ARGS+=(--load_phase1_results ${PHASE1_RESULTS})
fi

if [ -f "${PHASE1_5_RESULTS}" ]; then
    CMD_ARGS+=(--load_phase1_5_results ${PHASE1_5_RESULTS})
fi

# 运行训练
accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train_grpo_cl.py" \
    "${CMD_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=============================================="
echo "Phase ${PHASE} completed!"
echo "Log file: ${LOG_FILE}"
echo "Output dir: ${OUTPUT_LOG_DIR}"
echo "=============================================="
echo ""
echo "分阶段运行示例："
echo "  PHASE=1 bash train.sh                    # Zero-shot 评估"
echo "  PHASE=1.5 bash train.sh                  # 单任务 baseline 评估"
echo "  PHASE=2 bash train.sh                    # 持续学习训练"
echo "  PHASE=3 bash train.sh                    # 最终评估"
echo "  PHASE=all bash train.sh                  # 完整流程"
echo ""
echo "断点续训示例："
echo "  RESUME_FROM_TASK=2 PHASE=2 bash train.sh"
echo "  RESUME_FROM_CHECKPOINT=${OUTPUT_LOG_DIR}/checkpoint_task_X_step_Y PHASE=2 bash train.sh"
echo ""
echo "从评估阶段恢复（训练完成但评估中断）："
echo "  RESUME_FROM_TASK=1 SKIP_TRAINING_FOR_RESUME_TASK=true RESUME_FROM_EVAL_DOMAIN=retail RESUME_FROM=path/to/model PHASE=2 bash train.sh"
echo ""
echo "Phase 3 加载训练后模型："
echo "  RESUME_FROM=${OUTPUT_LOG_DIR}/checkpoints/final PHASE=3 bash train.sh"
echo "=============================================="
