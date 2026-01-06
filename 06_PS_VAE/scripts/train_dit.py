#!/usr/bin/env python3
"""
Training script for DiT (Diffusion Transformer) on PS-VAE latent space.

Usage:
    python scripts/train_dit.py --config configs/default.yaml --psvae-checkpoint outputs/checkpoints/final.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psvae.models import PSVAE
from psvae.diffusion import DiT, DiffusionScheduler
from psvae.diffusion.scheduler import DiffusionConfig
from psvae.training import DiTTrainer, ImageTextDataset

try:
    from transformers import T5EncoderModel, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT for T2I")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--psvae-checkpoint", type=str, required=True, help="Path to PS-VAE checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume from DiT checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-encoder", type=str, default="google/flan-t5-xl", help="Text encoder model")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set seed
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
    
    checkpoint = torch.load(args.psvae_checkpoint, map_location="cpu")
    psvae.load_state_dict(checkpoint["model"])
    psvae.eval()
    
    # Get latent spatial size
    spatial_size = psvae.get_latent_shape()[1]
    print(f"Latent spatial size: {spatial_size}x{spatial_size}")
    
    # Load text encoder
    text_encoder = None
    tokenizer = None
    text_dim = config.diffusion.dit.text_dim
    
    if TRANSFORMERS_AVAILABLE:
        print(f"Loading text encoder: {args.text_encoder}")
        tokenizer = T5Tokenizer.from_pretrained(args.text_encoder)
        text_encoder = T5EncoderModel.from_pretrained(args.text_encoder)
        text_encoder.eval()
        text_dim = text_encoder.config.d_model
        print(f"Text encoder dimension: {text_dim}")
    else:
        print("Warning: transformers not available. Text conditioning disabled.")
    
    # Create DiT
    dit = DiT(
        input_size=spatial_size,
        in_channels=config.model.latent.dim,
        hidden_size=config.diffusion.dit.hidden_size,
        depth=config.diffusion.dit.depth,
        num_heads=config.diffusion.dit.num_heads,
        mlp_ratio=config.diffusion.dit.mlp_ratio,
        text_dim=text_dim,
        num_text_tokens=config.diffusion.dit.num_text_tokens,
        class_dropout_prob=config.diffusion.dit.class_dropout_prob,
        learn_sigma=config.diffusion.dit.learn_sigma,
    )
    
    # Create diffusion scheduler
    diff_config = DiffusionConfig(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        beta_schedule=config.diffusion.beta_schedule,
        prediction_type=config.diffusion.prediction_type,
    )
    scheduler = DiffusionScheduler(diff_config)
    
    # Create dataset
    train_dataset = ImageTextDataset(
        data_dir=config.data.train_data_dir,
        image_size=config.data.image_size,
        tokenizer=tokenizer,
        max_text_length=config.diffusion.dit.num_text_tokens,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=tuple(config.training.optimizer.betas),
        eps=config.training.optimizer.eps,
    )
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * config.training.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.scheduler.warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=config.training.scheduler.min_lr / config.training.optimizer.lr,
    )
    
    # Training config
    trainer_config = {
        "lr": config.training.optimizer.lr,
        "weight_decay": config.training.optimizer.weight_decay,
        "mixed_precision": config.training.mixed_precision,
        "ema": OmegaConf.to_container(config.training.ema),
        "use_wandb": config.logging.use_wandb and WANDB_AVAILABLE,
        "log_every": config.logging.log_every,
        "save_every": config.training.save_every,
        "save_dir": os.path.join(args.output_dir, "checkpoints"),
    }
    
    # Initialize wandb
    if trainer_config["use_wandb"]:
        run_name = config.logging.run_name or f"dit_{config.diffusion.dit.depth}L"
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=OmegaConf.to_container(config),
        )
    
    # Create trainer
    trainer = DiTTrainer(
        dit=dit,
        psvae=psvae,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        text_encoder=text_encoder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=trainer_config,
        device=device,
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print(f"Starting training for {config.training.epochs} epochs")
    trainer.train(num_epochs=config.training.epochs)
    
    # Save final model
    trainer.save_checkpoint("final")
    
    if trainer_config["use_wandb"]:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()

