#!/bin/bash
# MS-Swift Training Script for QwenVL-2B on Custom Gastroenterology VQA Dataset
# This script trains Qwen2-VL-2B-Instruct model using LoRA on combined datasets:
# - Simple VQA (train split)
# - Grounding VQA
# - Comparison dataset

# Configuration
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="output/qwen2vl_gastro_vqa"

# Dataset paths (relative to project root)
SIMPLE_VQA="data/formatted/simple_vqa_train.json"
GROUNDING_VQA="data/formatted/grounding_vqa.json"
COMPARISON="data/formatted/comparison.json"

# Training hyperparameters
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LENGTH=2048
LORA_RANK=8
LORA_ALPHA=32

# Run training with LoRA
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ${MODEL_NAME} \
    --dataset ${SIMPLE_VQA} ${GROUNDING_VQA} ${COMPARISON} \
    --train_type lora \
    --dtype bfloat16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules all-linear \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --use_hf true
