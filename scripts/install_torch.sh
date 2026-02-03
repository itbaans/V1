#!/bin/bash
# Install PyTorch and torchvision based on detected CUDA version

set -e

echo "Detecting CUDA version..."

# Get CUDA version from nvcc or nvidia-smi
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
elif command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p')
else
    echo "Error: Could not detect CUDA version. Neither nvcc nor nvidia-smi found."
    exit 1
fi

echo "Detected CUDA version: ${CUDA_VERSION}"

# Map CUDA version to PyTorch index URL
case "${CUDA_VERSION}" in
    12.1|12.2|12.3|12.4|12.5|12.6)
        INDEX_URL="https://download.pytorch.org/whl/cu121"
        echo "Using CUDA 12.1 wheels (compatible with ${CUDA_VERSION})"
        ;;
    11.8)
        INDEX_URL="https://download.pytorch.org/whl/cu118"
        echo "Using CUDA 11.8 wheels"
        ;;
    11.7)
        INDEX_URL="https://download.pytorch.org/whl/cu117"
        echo "Using CUDA 11.7 wheels"
        ;;
    *)
        echo "Warning: CUDA ${CUDA_VERSION} - using default PyTorch (may need manual install)"
        INDEX_URL=""
        ;;
esac

echo "=============================================="
echo "Installing PyTorch, torchvision, torchaudio"
echo "=============================================="

if [ -n "$INDEX_URL" ]; then
    pip install torch torchvision torchaudio --index-url ${INDEX_URL}
else
    pip install torch torchvision torchaudio
fi

echo "=============================================="
echo "Installation complete!"
echo "=============================================="

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
