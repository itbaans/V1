#!/bin/bash
# Push trained model to HuggingFace

HF_REPO="peeache/qwen2vl-gastro-vqa-basic-1"
LATEST_CKPT="output/qwen2vl_gastro_vqa/v0-20260203-190931/checkpoint-2501"

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
