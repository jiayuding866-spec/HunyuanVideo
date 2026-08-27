#!/usr/bin/env bash
set -euo pipefail

# Download the official Diffusers conversion of HunyuanVideo-1.5 480P I2V Step-Distilled.
# In mainland China, huggingface.co may be unreachable. Uncomment the mirror line below.

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

MODEL_ID="${MODEL_ID:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled}"
LOCAL_DIR="${LOCAL_DIR:-./models/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled}"

pip install -U "huggingface_hub[cli]"
mkdir -p "$(dirname "$LOCAL_DIR")"
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"

echo "Model saved to $LOCAL_DIR"
