#!/usr/bin/env python3
"""HunyuanVideo-1.5 480P image-to-video inference (Diffusers, Step-Distilled)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_hunyuan_video15 import (
    HunyuanVideo15AttnProcessor2_0,
)
from diffusers.utils import export_to_video, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate video with HunyuanVideo-1.5 480P I2V Step-Distilled"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
    )
    parser.add_argument("--image_path", type=str, default="example/guitar-man.png")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A man with short gray hair plays a red electric guitar. "
            "His hands move naturally along the strings and fretboard. "
            "Subtle body movement, realistic cloth motion, a slow camera push-in, "
            "stable identity, continuous shot, cinematic natural lighting."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "flicker, jump cut, sudden scene change, duplicated limbs, "
            "distorted hands, identity change, static frame, low quality"
        ),
    )
    parser.add_argument("--output_folder", type=str, default="./output_hunyuan")
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument(
        "--enable_low_vram_attn",
        action="store_true",
        help="Broadcast key-padding mask as [B,1,1,S] instead of [B,1,S,S].",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Optional Hugging Face mirror, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()


def patch_low_vram_attention() -> None:
    """Avoid materializing a dense [B,1,S,S] padding mask (about 5GB at 121 frames)."""

    def _lowmem_attn_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
        batch_size, seq_len, heads, dim = query.shape
        attention_mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, seq_len)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    HunyuanVideo15AttnProcessor2_0.__call__ = _lowmem_attn_call


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires an NVIDIA CUDA GPU.")

    if args.enable_low_vram_attn:
        patch_low_vram_attention()

    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    image = load_image(args.image_path).convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hunyuan15_i2v_step_distilled.mp4"

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
        set_priority=True,
    ):
        result = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
        )
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    frames = result.frames[0]
    export_to_video(frames, str(output_path), fps=args.fps)

    print(f"Saved: {output_path.resolve()}")
    print(f"Frames: {len(frames)}")
    print(f"Duration at {args.fps} fps: {len(frames) / args.fps:.2f}s")
    print(f"Generation time: {elapsed:.1f}s")
    print(f"Peak allocated CUDA memory: {peak_gb:.2f}GB")


if __name__ == "__main__":
    main()
