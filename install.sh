#!/usr/bin/env bash
set -euo pipefail

# 0. Create conda environment (optional)
# conda create -n hunyuanvideo python=3.12 -y
# conda activate hunyuanvideo

# 1. Install PyTorch for your CUDA version, for example CUDA 12.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install remaining dependencies
pip install -r requirements.txt

echo "Install finished. Download weights with scripts/download/download_model.sh"
