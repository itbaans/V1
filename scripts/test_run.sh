#!/bin/bash
# Quick test run - just a few steps to verify setup works
# Also includes HuggingFace push after training

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="output/qwen2vl_test_run"
PROJECT_NAME="gastro-vqa-test"

# HuggingFace config (set your values)
HF_REPO="peeache/qwen2vl-gastro-vqa-test"  # Change this!
# Run: huggingface-cli login first

# Dataset paths
SIMPLE_VQA="data/formatted/simple_vqa_train.json"

# Test config - just 50 steps
MAX_STEPS=50
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4

echo "=============================================="
echo "TEST RUN - Verifying Setup"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Max Steps: ${MAX_STEPS}"
echo "=============================================="

# Single GPU test run
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ${MODEL_NAME} \
    --dataset ${SIMPLE_VQA} \
    --train_type lora \
    --torch_dtype bfloat16 \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --save_steps 25 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_checkpointing true \
    --use_hf true \
    --report_to wandb \
    --run_name ${PROJECT_NAME}

echo "=============================================="
echo "Test Run Complete!"
echo "=============================================="

# Push to HuggingFace
echo "Pushing to HuggingFace: ${HF_REPO}"
swift export \
    --adapters ${OUTPUT_DIR}/checkpoint-${MAX_STEPS} \
    --push_to_hub true \
    --hub_model_id ${HF_REPO} \
    --use_hf true

echo "=============================================="
echo "Pushed to: https://huggingface.co/${HF_REPO}"
echo "=============================================="
