OUTPUT_DIR="output/qwen2vl_test_run"
HF_REPO="peeache/qwen2vl-gastro-vqa-test"
MAX_STEPS=50
CKPT_DIR="/workspace/V1/output/qwen2vl_test_run/v6-20260203-03000/checkpoint-50"

echo "Pushing to HuggingFace: ${HF_REPO}"
swift export \
    --adapters ${CKPT_PATH} \
    --push_to_hub true \
    --hub_model_id ${HF_REPO} \
    --use_hf true
