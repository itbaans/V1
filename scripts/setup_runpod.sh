#!/bin/bash
# Quick setup script for RunPod environment
# Run this once when you first connect to the pod

set -e

echo "=== RunPod Environment Setup ==="

# Update and install essentials
apt-get update
apt-get install -y git wget tmux htop nvtop

# Install Python packages
pip install --upgrade pip
pip install ms-swift[all]
pip install transformers>=4.40.0 accelerate>=0.28.0 peft>=0.10.0
pip install qwen-vl-utils>=0.0.2
pip install deepspeed>=0.14.0
pip install wandb tensorboard
pip install datasets pandas Pillow tqdm

# Setup wandb (interactive login)
echo ""
echo "=== Weights & Biases Setup ==="
echo "Get your API key from: https://wandb.ai/authorize"
wandb login

# Verify GPU setup
echo ""
echo "=== GPU Information ==="
nvidia-smi

# Verify MS-Swift installation
echo ""
echo "=== MS-Swift Version ==="
swift --version

echo ""
echo "=== Setup Complete! ==="
echo "Next steps:"
echo "  1. Upload your data to /workspace/WeaklySupervised_MultiTaskLearning/data/"
echo "  2. Run: tmux new -s training"
echo "  3. Run: bash scripts/train_qwenvl_runpod.sh"
