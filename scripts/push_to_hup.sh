#!/bin/bash
# Push trained model to HuggingFace

OUTPUT_DIR="output/qwen2vl_gastro_vqa"
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
