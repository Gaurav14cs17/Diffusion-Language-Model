"""
Training classes for PS-VAE and DiT.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Optional, Dict, Any
from tqdm import tqdm
import math
from copy import deepcopy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        
        # Create EMA model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """Update EMA parameters."""
        self.step += 1
        if self.step % self.update_every != 0:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )
    
    def get_model(self) -> nn.Module:
        """Get EMA model."""
        return self.ema_model


class PSVAETrainer:
    """
    Trainer for PS-VAE autoencoder.
    
    Trains the PS-VAE model with semantic and pixel reconstruction objectives.
    
    Args:
        model: PS-VAE model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to use
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Default config
        self.config = config or {}
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get("lr", 1e-4),
                weight_decay=self.config.get("weight_decay", 0.01),
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Mixed precision
        self.use_amp = self.config.get("mixed_precision", "no") != "no"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = torch.bfloat16 if self.config.get("mixed_precision") == "bf16" else torch.float16
        
        # EMA
        if self.config.get("ema", {}).get("enabled", False):
            self.ema = EMA(
                model,
                decay=self.config["ema"].get("decay", 0.9999),
                update_every=self.config["ema"].get("update_every", 10),
            )
        else:
            self.ema = None
        
        # Logging
        self.use_wandb = self.config.get("use_wandb", False) and WANDB_AVAILABLE
        self.log_every = self.config.get("log_every", 100)
        
        # Checkpointing
        self.save_every = self.config.get("save_every", 5000)
        self.save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_semantic_loss = 0
        total_pixel_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get images
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
            else:
                images = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(images)
                    losses = self.model.compute_loss(outputs)
                
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.model.compute_loss(outputs)
                losses["loss"].backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.ema is not None:
                self.ema.update()
            
            # Accumulate losses
            total_loss += losses["loss"].item()
            total_semantic_loss += losses.get("semantic_loss", torch.tensor(0)).item()
            total_pixel_loss += losses.get("pixel_loss", torch.tensor(0)).item()
            total_kl_loss += losses["kl_loss"].item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "sem": f"{losses.get('semantic_loss', torch.tensor(0)).item():.4f}",
                "pix": f"{losses.get('pixel_loss', torch.tensor(0)).item():.4f}",
            })
            
            # Logging
            if self.global_step % self.log_every == 0 and self.use_wandb:
                wandb.log({
                    "train/loss": losses["loss"].item(),
                    "train/semantic_loss": losses.get("semantic_loss", torch.tensor(0)).item(),
                    "train/pixel_loss": losses.get("pixel_loss", torch.tensor(0)).item(),
                    "train/kl_loss": losses["kl_loss"].item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "global_step": self.global_step,
                })
            
            # Checkpointing
            if self.global_step % self.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            self.global_step += 1
        
        num_batches = len(self.train_dataloader)
        return {
            "loss": total_loss / num_batches,
            "semantic_loss": total_semantic_loss / num_batches,
            "pixel_loss": total_pixel_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_semantic_loss = 0
        total_pixel_loss = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
            else:
                images = batch.to(self.device)
            
            outputs = self.model(images)
            losses = self.model.compute_loss(outputs)
            
            total_loss += losses["loss"].item()
            total_semantic_loss += losses.get("semantic_loss", torch.tensor(0)).item()
            total_pixel_loss += losses.get("pixel_loss", torch.tensor(0)).item()
        
        num_batches = len(self.val_dataloader)
        metrics = {
            "val/loss": total_loss / num_batches,
            "val/semantic_loss": total_semantic_loss / num_batches,
            "val/pixel_loss": total_pixel_loss / num_batches,
        }
        
        if self.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            
            if self.val_dataloader is not None:
                val_metrics = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_metrics['val/loss']:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        if self.ema is not None:
            checkpoint["ema"] = self.ema.ema_model.state_dict()
        
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        if self.ema is not None and "ema" in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint["ema"])
        
        print(f"Loaded checkpoint from {path}")


class DiTTrainer:
    """
    Trainer for Diffusion Transformer.
    
    Trains DiT for text-to-image generation in PS-VAE latent space.
    
    Args:
        dit: DiT model
        psvae: PS-VAE model (for encoding)
        scheduler: Diffusion scheduler
        train_dataloader: Training data loader
        text_encoder: Text encoder model
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to use
    """
    
    def __init__(
        self,
        dit: nn.Module,
        psvae: nn.Module,
        scheduler,
        train_dataloader: DataLoader,
        text_encoder: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.dit = dit.to(device)
        self.psvae = psvae.to(device)
        self.psvae.eval()
        for param in self.psvae.parameters():
            param.requires_grad = False
        
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.text_encoder = text_encoder
        if text_encoder is not None:
            self.text_encoder = text_encoder.to(device)
            self.text_encoder.eval()
        
        self.device = device
        self.config = config or {}
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                dit.parameters(),
                lr=self.config.get("lr", 1e-4),
                weight_decay=self.config.get("weight_decay", 0.01),
            )
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        
        # Mixed precision
        self.use_amp = self.config.get("mixed_precision", "no") != "no"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.amp_dtype = torch.bfloat16 if self.config.get("mixed_precision") == "bf16" else torch.float16
        
        # EMA
        if self.config.get("ema", {}).get("enabled", False):
            self.ema = EMA(
                dit,
                decay=self.config["ema"].get("decay", 0.9999),
                update_every=self.config["ema"].get("update_every", 10),
            )
        else:
            self.ema = None
        
        # Logging
        self.use_wandb = self.config.get("use_wandb", False) and WANDB_AVAILABLE
        self.log_every = self.config.get("log_every", 100)
        
        # Checkpointing
        self.save_every = self.config.get("save_every", 5000)
        self.save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.global_step = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.dit.train()
        
        # Get images and encode to latents
        images = batch["image"].to(self.device)
        
        with torch.no_grad():
            latents = self.psvae.encode_to_latent(images)
        
        # Get text embeddings
        if self.text_encoder is not None and "input_ids" in batch:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                text_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                text_embeds = text_outputs.last_hidden_state
        else:
            text_embeds = batch.get("text_embed")
            if text_embeds is not None:
                text_embeds = text_embeds.to(self.device)
            attention_mask = batch.get("text_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        
        # Sample timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )
        
        # Sample noise and add to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                noise_pred = self.dit(noisy_latents, timesteps, text_embeds, attention_mask)
                
                # Handle learned variance
                if noise_pred.shape[-1] == latents.shape[-1] * 2:
                    noise_pred, _ = noise_pred.chunk(2, dim=-1)
                
                loss = F.mse_loss(noise_pred, noise)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            noise_pred = self.dit(noisy_latents, timesteps, text_embeds, attention_mask)
            
            if noise_pred.shape[-1] == latents.shape[-1] * 2:
                noise_pred, _ = noise_pred.chunk(2, dim=-1)
            
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            self.optimizer.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        if self.ema is not None:
            self.ema.update()
        
        return {"loss": loss.item()}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch)
            total_loss += losses["loss"]
            
            pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})
            
            # Logging
            if self.global_step % self.log_every == 0 and self.use_wandb:
                wandb.log({
                    "train/loss": losses["loss"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "global_step": self.global_step,
                })
            
            # Checkpointing
            if self.global_step % self.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            self.global_step += 1
        
        return {"loss": total_loss / len(self.train_dataloader)}
    
    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Loss: {metrics['loss']:.4f}")
            self.save_checkpoint(f"epoch_{epoch}")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint = {
            "dit": self.dit.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        
        if self.lr_scheduler is not None:
            checkpoint["scheduler"] = self.lr_scheduler.state_dict()
        
        if self.ema is not None:
            checkpoint["ema"] = self.ema.ema_model.state_dict()
        
        path = os.path.join(self.save_dir, f"dit_{name}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.dit.load_state_dict(checkpoint["dit"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        
        if self.lr_scheduler is not None and "scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        
        if self.ema is not None and "ema" in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint["ema"])
        
        print(f"Loaded checkpoint from {path}")

