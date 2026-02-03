#!/bin/bash
# Temporary quick test - 5 steps only

set -e

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="output/qwen2vl_gastro_vqa"

# Dataset paths
SIMPLE_VQA="data/formatted/simple_vqa_train.json"
GROUNDING_VQA="data/formatted/grounding_vqa.json"
COMPARISON="data/formatted/comparison.json"

PROJECT_NAME="gastro-vqa-qwenvl"

# Quick test - 5 steps
NUM_EPOCHS=1
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
MAX_LENGTH=4096

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "TEMP RUN - 5 steps only"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model ${MODEL_NAME} \
    --dataset ${SIMPLE_VQA}\
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --save_steps 25 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_checkpointing true \
    --use_hf true \
    --report_to wandb \
    --run_name ${PROJECT_NAME}

echo "=============================================="
echo "Temp run complete!"
echo "=============================================="

HF_REPO="peeache/qwen2vl-gastro-vqa-basic"  # Change for production!

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
    --use_hf true \
    --merge_lora true

echo "=============================================="
echo "Pushed to: https://huggingface.co/${HF_REPO}"
echo "=============================================="