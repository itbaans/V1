OUTPUT_DIR="output/qwen2vl_test_run"
HF_REPO="peeache/qwen2vl-gastro-vqa-test"
MAX_STEPS=50

echo "Pushing to HuggingFace: ${HF_REPO}"
swift export \
    --adapters ./${OUTPUT_DIR}/checkpoint-${MAX_STEPS} \
    --push_to_hub true \
    --hub_model_id ${HF_REPO} \
