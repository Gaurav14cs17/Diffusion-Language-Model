#!/usr/bin/env python3
"""
Training validation tests using tiny open-source datasets.

Uses CIFAR-10 from torchvision as a tiny dataset to validate 
that all training code runs correctly end-to-end.

Usage:
    python tests/test_training_validation.py
    python tests/test_training_validation.py --test psvae
    python tests/test_training_validation.py --test dit
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

# Try to import torchvision for CIFAR-10
try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def create_synthetic_dataset(data_dir: str, num_images: int = 20, image_size: int = 64):
    """
    Create a synthetic dataset with random images and captions.
    
    Args:
        data_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of images
    """
    os.makedirs(data_dir, exist_ok=True)
    
    metadata = []
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        img_name = f"image_{i:04d}.png"
        img_path = os.path.join(data_dir, img_name)
        img.save(img_path)
        
        # Create caption file
        caption = f"A synthetic test image number {i}"
        txt_path = os.path.join(data_dir, f"image_{i:04d}.txt")
        with open(txt_path, "w") as f:
            f.write(caption)
        
        metadata.append({
            "image": img_name,
            "caption": caption,
        })
    
    # Also save as metadata.json for ImageTextDataset
    import json
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {num_images} synthetic images in {data_dir}")
    return data_dir


def download_cifar10_as_images(data_dir: str, num_images: int = 50, image_size: int = 64):
    """
    Download CIFAR-10 and save as individual images with captions.
    
    Args:
        data_dir: Directory to save images
        num_images: Number of images to save
        image_size: Target image size (will resize from 32x32)
    """
    if not TORCHVISION_AVAILABLE:
        print("torchvision not available, using synthetic data")
        return create_synthetic_dataset(data_dir, num_images, image_size)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # CIFAR-10 class names
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Download CIFAR-10
    try:
        cifar = datasets.CIFAR10(
            root=os.path.join(data_dir, ".cifar_cache"),
            train=True,
            download=True,
        )
    except Exception as e:
        print(f"Failed to download CIFAR-10: {e}")
        print("Using synthetic data instead")
        return create_synthetic_dataset(data_dir, num_images, image_size)
    
    # Save subset as individual images
    metadata = []
    for i in range(min(num_images, len(cifar))):
        img, label = cifar[i]
        
        # Resize to target size
        img = img.resize((image_size, image_size), Image.BILINEAR)
        
        # Save image
        img_name = f"cifar_{i:04d}.png"
        img_path = os.path.join(data_dir, img_name)
        img.save(img_path)
        
        # Create caption
        caption = f"A photo of a {classes[label]}"
        txt_path = os.path.join(data_dir, f"cifar_{i:04d}.txt")
        with open(txt_path, "w") as f:
            f.write(caption)
        
        metadata.append({
            "image": img_name,
            "caption": caption,
        })
    
    # Save metadata.json
    import json
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {len(metadata)} CIFAR-10 images in {data_dir}")
    return data_dir


def test_psvae_training(config_path: str, data_dir: str, output_dir: str) -> bool:
    """
    Test PS-VAE training script.
    
    Args:
        config_path: Path to test config
        data_dir: Path to test data
        output_dir: Path for outputs
    
    Returns:
        True if test passes
    """
    print("\n" + "="*60)
    print("Testing PS-VAE Training")
    print("="*60)
    
    from psvae.models import PSVAE
    from psvae.training import PSVAETrainer, ImageDataset
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Override data paths
    config.data.train_data_dir = data_dir
    config.data.val_data_dir = data_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create model
        print("Creating PS-VAE model...")
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
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create dataset
        print("Creating dataset...")
        train_dataset = ImageDataset(
            data_dir=config.data.train_data_dir,
            image_size=config.data.image_size,
        )
        print(f"Dataset size: {len(train_dataset)}")
        
        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=min(config.training.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last=True,
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
        
        # Training config
        trainer_config = {
            "lr": config.training.optimizer.lr,
            "weight_decay": config.training.optimizer.weight_decay,
            "mixed_precision": config.training.mixed_precision,
            "ema": OmegaConf.to_container(config.training.ema),
            "use_wandb": False,
            "log_every": config.logging.log_every,
            "save_every": 100000,  # Don't save during test
            "save_dir": os.path.join(output_dir, "checkpoints"),
        }
        
        # Create trainer
        print("Creating trainer...")
        trainer = PSVAETrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            optimizer=optimizer,
            scheduler=None,
            config=trainer_config,
            device=device,
        )
        
        # Train for a few steps
        print("Running training for 1 epoch...")
        trainer.train(num_epochs=1)
        
        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(train_dataloader))
            if isinstance(test_batch, dict):
                test_images = test_batch["image"].to(device)
            else:
                test_images = test_batch.to(device)
            
            outputs = model(test_images)
            print(f"  Input shape: {test_images.shape}")
            print(f"  Latent shape: {outputs['z'].shape}")
            print(f"  Recon pixel shape: {outputs['recon_pixel'].shape}")
        
        # Test encode/decode
        print("Testing encode/decode...")
        with torch.no_grad():
            z = model.encode_to_latent(test_images)
            recon = model.decode_to_image(z)
            print(f"  Encoded latent shape: {z.shape}")
            print(f"  Decoded image shape: {recon.shape}")
        
        # Save a test checkpoint
        print("Saving test checkpoint...")
        checkpoint_path = os.path.join(output_dir, "checkpoints", "psvae_test.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": OmegaConf.to_container(config),
        }, checkpoint_path)
        print(f"  Saved to {checkpoint_path}")
        
        print("\n✓ PS-VAE training test PASSED!")
        return True, checkpoint_path
        
    except Exception as e:
        print(f"\n✗ PS-VAE training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_dit_training(config_path: str, data_dir: str, output_dir: str, psvae_checkpoint: str) -> bool:
    """
    Test DiT training script.
    
    Args:
        config_path: Path to test config
        data_dir: Path to test data
        output_dir: Path for outputs
        psvae_checkpoint: Path to PS-VAE checkpoint
    
    Returns:
        True if test passes
    """
    print("\n" + "="*60)
    print("Testing DiT Training")
    print("="*60)
    
    from psvae.models import PSVAE
    from psvae.diffusion import DiT, DiffusionScheduler
    from psvae.diffusion.scheduler import DiffusionConfig
    from psvae.training import DiTTrainer, ImageTextDataset
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Override data paths
    config.data.train_data_dir = data_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load PS-VAE
        print(f"Loading PS-VAE from {psvae_checkpoint}...")
        psvae = PSVAE(
            encoder_type=config.model.encoder.type,
            model_size=config.model.encoder.size,
            latent_dim=config.model.latent.dim,
            image_size=config.data.image_size,
        )
        
        checkpoint = torch.load(psvae_checkpoint, map_location="cpu")
        psvae.load_state_dict(checkpoint["model"])
        psvae.eval()
        
        # Get latent spatial size
        spatial_size = psvae.get_latent_shape()[1]
        print(f"Latent spatial size: {spatial_size}x{spatial_size}")
        
        # Create DiT (without text encoder for simplicity)
        print("Creating DiT model...")
        dit = DiT(
            input_size=spatial_size,
            in_channels=config.model.latent.dim,
            hidden_size=config.diffusion.dit.hidden_size,
            depth=config.diffusion.dit.depth,
            num_heads=config.diffusion.dit.num_heads,
            mlp_ratio=config.diffusion.dit.mlp_ratio,
            text_dim=config.diffusion.dit.text_dim,
            num_text_tokens=config.diffusion.dit.num_text_tokens,
            class_dropout_prob=config.diffusion.dit.class_dropout_prob,
            learn_sigma=config.diffusion.dit.learn_sigma,
        )
        print(f"DiT created with {sum(p.numel() for p in dit.parameters()):,} parameters")
        
        # Create diffusion scheduler
        diff_config = DiffusionConfig(
            num_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            beta_schedule=config.diffusion.beta_schedule,
            prediction_type=config.diffusion.prediction_type,
        )
        scheduler = DiffusionScheduler(diff_config)
        
        # Create dataset (without tokenizer for simplicity)
        print("Creating dataset...")
        train_dataset = ImageTextDataset(
            data_dir=config.data.train_data_dir,
            image_size=config.data.image_size,
            tokenizer=None,  # No text encoder for testing
            max_text_length=config.diffusion.dit.num_text_tokens,
        )
        print(f"Dataset size: {len(train_dataset)}")
        
        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=min(config.training.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last=True,
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            dit.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
        
        # Training config
        trainer_config = {
            "lr": config.training.optimizer.lr,
            "weight_decay": config.training.optimizer.weight_decay,
            "mixed_precision": config.training.mixed_precision,
            "ema": OmegaConf.to_container(config.training.ema),
            "use_wandb": False,
            "log_every": config.logging.log_every,
            "save_every": 100000,  # Don't save during test
            "save_dir": os.path.join(output_dir, "checkpoints"),
        }
        
        # Create trainer
        print("Creating trainer...")
        trainer = DiTTrainer(
            dit=dit,
            psvae=psvae,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            text_encoder=None,  # No text encoder for testing
            optimizer=optimizer,
            lr_scheduler=None,
            config=trainer_config,
            device=device,
        )
        
        # Train for a few steps
        print("Running training for 1 epoch...")
        trainer.train(num_epochs=1)
        
        # Test forward pass
        print("Testing DiT forward pass...")
        dit.eval()
        with torch.no_grad():
            # Get a batch
            test_batch = next(iter(train_dataloader))
            test_images = test_batch["image"].to(device)
            
            # Encode to latents
            latents = psvae.encode_to_latent(test_images)
            
            # Sample timesteps
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # Forward through DiT (no text conditioning)
            noise_pred = dit(noisy_latents, timesteps, text_embeds=None, text_mask=None)
            
            print(f"  Latent shape: {latents.shape}")
            print(f"  Noise pred shape: {noise_pred.shape}")
        
        # Save test checkpoint
        print("Saving test checkpoint...")
        checkpoint_path = os.path.join(output_dir, "checkpoints", "dit_test.pt")
        torch.save({
            "dit": dit.state_dict(),
            "config": OmegaConf.to_container(config),
        }, checkpoint_path)
        print(f"  Saved to {checkpoint_path}")
        
        print("\n✓ DiT training test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ DiT training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training code with tiny dataset")
    parser.add_argument("--test", type=str, choices=["psvae", "dit", "all"], default="all",
                       help="Which test to run")
    parser.add_argument("--data-source", type=str, choices=["cifar10", "synthetic"], default="cifar10",
                       help="Data source to use")
    parser.add_argument("--num-images", type=int, default=20,
                       help="Number of images to use for testing")
    parser.add_argument("--keep-data", action="store_true",
                       help="Keep test data after running")
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "test_tiny.yaml"
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="psvae_test_")
    data_dir = os.path.join(temp_dir, "data")
    output_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test directory: {temp_dir}")
    print(f"Config: {config_path}")
    
    try:
        # Create test data
        print("\n" + "="*60)
        print("Creating Test Data")
        print("="*60)
        
        # Image size must be multiple of 14 for DINOv2 patch size
        image_size = 112
        if args.data_source == "cifar10":
            data_dir = download_cifar10_as_images(data_dir, args.num_images, image_size=image_size)
        else:
            data_dir = create_synthetic_dataset(data_dir, args.num_images, image_size=image_size)
        
        results = {}
        psvae_checkpoint = None
        
        # Run PS-VAE test
        if args.test in ["psvae", "all"]:
            passed, psvae_checkpoint = test_psvae_training(str(config_path), data_dir, output_dir)
            results["psvae"] = passed
        
        # Run DiT test
        if args.test in ["dit", "all"]:
            if psvae_checkpoint is None:
                # Need to train PS-VAE first
                print("\nTraining PS-VAE first (required for DiT)...")
                passed, psvae_checkpoint = test_psvae_training(str(config_path), data_dir, output_dir)
                if not passed:
                    print("Cannot run DiT test without PS-VAE checkpoint")
                    results["dit"] = False
                else:
                    results["dit"] = test_dit_training(str(config_path), data_dir, output_dir, psvae_checkpoint)
            else:
                results["dit"] = test_dit_training(str(config_path), data_dir, output_dir, psvae_checkpoint)
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✓ All tests PASSED!")
            return 0
        else:
            print("\n✗ Some tests FAILED!")
            return 1
            
    finally:
        # Cleanup
        if not args.keep_data:
            print(f"\nCleaning up {temp_dir}...")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nTest data kept at: {temp_dir}")


if __name__ == "__main__":
    sys.exit(main())

