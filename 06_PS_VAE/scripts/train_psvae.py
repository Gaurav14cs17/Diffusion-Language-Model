#!/usr/bin/env python3
"""
Training script for PS-VAE autoencoder.

Usage:
    python scripts/train_psvae.py --config configs/default.yaml
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

from psvae.models import PSVAE, SVAE
from psvae.training import PSVAETrainer, ImageDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train PS-VAE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--svae-only", action="store_true", help="Train S-VAE only (frozen encoder)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
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
    
    # Create model
    if args.svae_only:
        print("Training S-VAE (frozen encoder)")
        model = SVAE(
            encoder_type=config.model.encoder.type,
            model_size=config.model.encoder.size,
            latent_dim=config.model.latent.dim,
            kl_weight=config.model.loss.kl_weight,
        )
    else:
        print("Training PS-VAE (trainable encoder)")
        model = PSVAE(
            encoder_type=config.model.encoder.type,
            model_size=config.model.encoder.size,
            latent_dim=config.model.latent.dim,
            image_size=config.data.image_size,
            kl_weight=config.model.loss.kl_weight,
            semantic_weight=config.model.loss.semantic_weight,
            pixel_weight=config.model.loss.pixel_weight,
            perceptual_weight=config.model.loss.perceptual_weight,
        )
    
    # Create datasets
    train_dataset = ImageDataset(
        data_dir=config.data.train_data_dir,
        image_size=config.data.image_size,
    )
    
    val_dataset = None
    if hasattr(config.data, "val_data_dir") and config.data.val_data_dir:
        val_dataset = ImageDataset(
            data_dir=config.data.val_data_dir,
            image_size=config.data.image_size,
        )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=tuple(config.training.optimizer.betas),
        eps=config.training.optimizer.eps,
    )
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * config.training.epochs
    scheduler = get_cosine_schedule_with_warmup(
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
        run_name = config.logging.run_name or f"psvae_{config.model.encoder.type}_{config.model.encoder.size}"
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=OmegaConf.to_container(config),
        )
    
    # Create trainer
    trainer = PSVAETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
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

