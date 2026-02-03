#!/bin/bash
# MS-Swift Training Script for RunPod (4x GPU with DeepSpeed + WandB)
# Optimized for 4x RTX 3090 / A5000 setup

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="output/qwen2vl_gastro_vqa"
PROJECT_NAME="gastro-vqa-qwenvl"

# Dataset paths
SIMPLE_VQA="data/formatted/simple_vqa_train.json"
GROUNDING_VQA="data/formatted/grounding_vqa.json"
COMPARISON="data/formatted/comparison.json"

# Training hyperparameters (optimized for 4x 24GB GPUs)
BATCH_SIZE=1                   # Per GPU batch size
GRADIENT_ACCUMULATION=4         # Effective batch = 4 * 2 * 8 = 64
LEARNING_RATE=1e-4
NUM_EPOCHS=1
MAX_LENGTH=4096

# LoRA config
LORA_RANK=8
LORA_ALPHA=32

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Starting Multi-GPU Training on RunPod"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Multi-GPU training with DeepSpeed ZeRO-2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model ${MODEL_NAME} \
    --dataset ${SIMPLE_VQA} ${GROUNDING_VQA} ${COMPARISON} \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules all-linear \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --use_hf true \
    --report_to wandb \
    --run_name ${PROJECT_NAME}

echo "=============================================="
echo "Training Complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=============================================="

HF_REPO="peeache/qwen2vl-gastro-vqa"  # Change for production!

# Find the latest checkpoint automatically
# Gets the most recent v*-* folder, then finds the highest checkpoint
LATEST_RUN=$(ls -dt ${OUTPUT_DIR}/v*-* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No training runs found in ${OUTPUT_DIR}"
    exit 1
fi

# Find the highest checkpoint number in that run
LATEST_CKPT=$(ls -d ${LATEST_RUN}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoints found in ${LATEST_RUN}"
    exit 1
fi

echo "=============================================="
echo "Pushing to HuggingFace: ${HF_REPO}"
echo "Checkpoint: ${LATEST_CKPT}"
echo "=============================================="

swift export \
    --adapters ${LATEST_CKPT} \
    --push_to_hub true \
    --hub_model_id ${HF_REPO} \
    --use_hf true

echo "=============================================="
echo "Pushed to: https://huggingface.co/${HF_REPO}"
echo "=============================================="