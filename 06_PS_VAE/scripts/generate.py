#!/usr/bin/env python3
"""
Generation script for PS-VAE + DiT.

Usage:
    python scripts/generate.py \
        --psvae-checkpoint outputs/checkpoints/final.pt \
        --dit-checkpoint outputs/checkpoints/dit_final.pt \
        --prompt "A beautiful sunset over mountains" \
        --output output.png
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psvae.models import PSVAE
from psvae.diffusion import DiT, DiffusionScheduler, DDIMSampler
from psvae.diffusion.scheduler import DiffusionConfig

try:
    from transformers import T5EncoderModel, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with PS-VAE + DiT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--psvae-checkpoint", type=str, required=True)
    parser.add_argument("--dit-checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--text-encoder", type=str, default="google/flan-t5-xl")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PS-VAE
    print(f"Loading PS-VAE from {args.psvae_checkpoint}")
    psvae = PSVAE(
        encoder_type=config.model.encoder.type,
        model_size=config.model.encoder.size,
        latent_dim=config.model.latent.dim,
        image_size=config.data.image_size,
    )
    
    psvae_ckpt = torch.load(args.psvae_checkpoint, map_location="cpu")
    if "ema" in psvae_ckpt:
        psvae.load_state_dict(psvae_ckpt["ema"])
        print("Loaded EMA weights for PS-VAE")
    else:
        psvae.load_state_dict(psvae_ckpt["model"])
    
    psvae = psvae.to(device)
    psvae.eval()
    
    # Get latent shape
    spatial_size = psvae.get_latent_shape()[1]
    latent_shape = (args.num_images, spatial_size, spatial_size, config.model.latent.dim)
    print(f"Latent shape: {latent_shape}")
    
    # Load text encoder
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required for generation. Install with: pip install transformers")
    
    print(f"Loading text encoder: {args.text_encoder}")
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_dim = text_encoder.config.d_model
    
    # Load DiT
    print(f"Loading DiT from {args.dit_checkpoint}")
    dit = DiT(
        input_size=spatial_size,
        in_channels=config.model.latent.dim,
        hidden_size=config.diffusion.dit.hidden_size,
        depth=config.diffusion.dit.depth,
        num_heads=config.diffusion.dit.num_heads,
        mlp_ratio=config.diffusion.dit.mlp_ratio,
        text_dim=text_dim,
        num_text_tokens=config.diffusion.dit.num_text_tokens,
        class_dropout_prob=0.0,  # No dropout during inference
        learn_sigma=config.diffusion.dit.learn_sigma,
    )
    
    dit_ckpt = torch.load(args.dit_checkpoint, map_location="cpu")
    if "ema" in dit_ckpt:
        dit.load_state_dict(dit_ckpt["ema"])
        print("Loaded EMA weights for DiT")
    else:
        dit.load_state_dict(dit_ckpt["dit"])
    
    dit = dit.to(device)
    dit.eval()
    
    # Create scheduler and sampler
    diff_config = DiffusionConfig(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
        prediction_type=config.diffusion.prediction_type,
    )
    scheduler = DiffusionScheduler(diff_config).to(device)
    sampler = DDIMSampler(scheduler, dit)
    
    # Encode text prompt
    print(f"Prompt: {args.prompt}")
    
    with torch.no_grad():
        # Tokenize
        tokens = tokenizer(
            args.prompt,
            max_length=config.diffusion.dit.num_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Expand for batch
        input_ids = input_ids.expand(args.num_images, -1)
        attention_mask = attention_mask.expand(args.num_images, -1)
        
        # Encode
        text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state
    
    # Generate
    print(f"Generating {args.num_images} image(s) with {args.steps} steps...")
    
    with torch.no_grad():
        latents = sampler.sample(
            shape=latent_shape,
            text_embeds=text_embeds,
            text_mask=attention_mask.bool(),
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            eta=0.0,
            device=device,
            progress=True,
        )
        
        # Decode to images
        images = psvae.decode_to_image(latents)
    
    # Post-process and save
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)
    images = (images * 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    # Save images
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.num_images == 1:
        Image.fromarray(images[0]).save(output_path)
        print(f"Saved image to {output_path}")
    else:
        for i, img in enumerate(images):
            path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
            Image.fromarray(img).save(path)
            print(f"Saved image to {path}")
    
    print("Done!")


if __name__ == "__main__":
    main()

