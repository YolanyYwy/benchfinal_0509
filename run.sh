#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ================== 基本配置 ==================
torchrun --nproc_per_node=8 -m tau2.scripts.train_grpo_cl \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 4 \
    --num_steps_per_task 100 \
    --learning_rate 1e-6 \
    --kl_coef 0.1 \
    --cl_algorithm sequential \
    --task_order instore ota delivery \
    --log_dir logs/qwen3_4b_replay \
    --wandb_project qwen3-4b-cl \
    --trajectory_log_interval 1 \
    --num_samples_per_prompt 2 \
    --gradient_checkpointing \
    --use_flash_attention \
    --max_new_tokens 1024