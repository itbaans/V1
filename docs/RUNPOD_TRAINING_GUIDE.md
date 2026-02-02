# RunPod Training Setup Guide
## Training QwenVL-2B on Gastroenterology VQA Dataset

---

## 1. RunPod Pod Setup

### Recommended Pod Configuration
| Spec | Recommendation |
|------|----------------|
| GPU | 4x RTX 3090 or 4x A5000 |
| Template | `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` |
| Disk | 50GB (model ~5GB + data ~10GB + deps ~5GB + checkpoints ~5GB) |
| Volume | Optional 20GB for persistent checkpoints |

### Create Pod
1. Go to [runpod.io](https://runpod.io) → **Pods** → **+ New Pod**
2. Select **Community Cloud** or **Secure Cloud**
3. Choose **4x RTX 3090** (or your preferred config)
4. Select template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
5. Set **Container Disk**: 100GB
6. Add **Volume** (optional): 50GB for persistent checkpoint storage
7. Click **Deploy**

---

## 2. Initial Setup (Run Once)

SSH into your pod or use the web terminal:

```bash
# Update system
apt-get update && apt-get install -y git wget tmux

# Clone your repository (or upload via SCP)
cd /workspace
git clone https://github.com/YOUR_USERNAME/WeaklySupervised_MultiTaskLearning.git
cd WeaklySupervised_MultiTaskLearning

# Install MS-Swift and dependencies
pip install ms-swift[all] transformers accelerate peft
pip install qwen-vl-utils deepspeed
pip install wandb tensorboard

# Login to Weights & Biases for live graphs
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## 3. Upload Your Data

### Option A: SCP from local machine
```bash
# From your LOCAL machine (not the pod)
scp -P <PORT> -r data/formatted/ root@<POD_IP>:/workspace/WeaklySupervised_MultiTaskLearning/data/
scp -P <PORT> -r data/images/ root@<POD_IP>:/workspace/WeaklySupervised_MultiTaskLearning/data/
```

### Option B: Use rclone with cloud storage
```bash
# On the pod
pip install rclone
rclone config  # Setup Google Drive/S3/etc
rclone copy gdrive:datasets/formatted /workspace/WeaklySupervised_MultiTaskLearning/data/formatted
rclone copy gdrive:datasets/images /workspace/WeaklySupervised_MultiTaskLearning/data/images
```

### Option C: Direct download (if hosted publicly)
```bash
wget -O data.zip https://your-storage-url/data.zip
unzip data.zip -d /workspace/WeaklySupervised_MultiTaskLearning/
```

---

## 4. Run Training with Live Graphs

### Start Training in tmux (keeps running if disconnected)
```bash
cd /workspace/WeaklySupervised_MultiTaskLearning
tmux new -s training

# Run training with wandb logging
bash scripts/train_qwenvl_runpod.sh
```

### Detach from tmux: `Ctrl+B` then `D`
### Reattach later: `tmux attach -t training`

---

## 5. Monitor Training

### Weights & Biases (Real-time graphs)
- Go to [wandb.ai](https://wandb.ai) → Your project
- View live loss curves, learning rate, GPU usage

### TensorBoard (Alternative)
```bash
# In another terminal
tensorboard --logdir output/qwen2vl_gastro_vqa --port 6006 --bind_all

# Access via: http://<POD_IP>:6006
```

---

## 6. After Training

### Download Checkpoints
```bash
# From LOCAL machine
scp -P <PORT> -r root@<POD_IP>:/workspace/WeaklySupervised_MultiTaskLearning/output ./
```

### Test Inference
```bash
swift infer \
    --adapters output/qwen2vl_gastro_vqa/checkpoint-xxx \
    --stream true \
    --max_new_tokens 512
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `batch_size` or enable `gradient_checkpointing` |
| Slow training | Increase `batch_size`, use DeepSpeed ZeRO-2 |
| Connection lost | Use `tmux` - training continues in background |
| wandb not logging | Check `WANDB_API_KEY` env variable |
