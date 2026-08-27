#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 480P I2V Step-Distilled: 8 or 12 steps recommended, CFG scale 1.0, flow shift 7.
# Default paths work after you download weights with scripts/download/download_model.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer_hunyuan.py \
  --model_path "${MODEL_PATH:-./models/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled}" \
  --image_path "example/guitar-man.png" \
  --prompt "$(cat example/prompt.txt)" \
  --negative_prompt "flicker, jump cut, sudden scene change, duplicated limbs, distorted hands, identity change, static frame, low quality" \
  --num_frames 121 \
  --num_inference_steps 12 \
  --fps 24 \
  --seed 42 \
  --output_folder "./output_hunyuan" \
  --enable_cpu_offload \
  --enable_low_vram_attn
