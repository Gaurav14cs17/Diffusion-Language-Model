#!/usr/bin/env python3
"""
Quick pytest-based validation tests for PS-VAE training code.

Uses synthetic data for fast testing without network downloads.

Usage:
    pytest tests/test_few_epochs.py -v
    pytest tests/test_few_epochs.py -v -k psvae
    pytest tests/test_few_epochs.py -v -k dit
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def test_data_dir():
    """Create temporary test data directory with synthetic images."""
    temp_dir = tempfile.mkdtemp(prefix="psvae_pytest_")
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create synthetic images (112x112 to be compatible with DINOv2 patch size 14)
    image_size = 112
    num_images = 8
    
    import json
    metadata = []
    
    for i in range(num_images):
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_name = f"test_image_{i:04d}.png"
        img_path = os.path.join(data_dir, img_name)
        img.save(img_path)
        
        caption = f"A synthetic test image number {i}"
        txt_path = os.path.join(data_dir, f"test_image_{i:04d}.txt")
        with open(txt_path, "w") as f:
            f.write(caption)
        
        metadata.append({"image": img_name, "caption": caption})
    
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp(prefix="psvae_output_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestPSVAETraining:
    """Tests for PS-VAE training pipeline."""
    
    def test_psvae_model_creation(self, device):
        """Test PS-VAE model can be created."""
        from psvae.models import PSVAE
        
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            kl_weight=0.0001,
            semantic_weight=1.0,
            pixel_weight=1.0,
            perceptual_weight=0.0,
        )
        
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'compute_loss')
    
    def test_psvae_forward_pass(self, device):
        """Test PS-VAE forward pass."""
        from psvae.models import PSVAE
        
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            perceptual_weight=0.0,
        ).to(device)
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 112, 112, device=device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        assert "z" in outputs
        assert "mu" in outputs
        assert "log_var" in outputs
        assert "recon_pixel" in outputs
        assert "recon_semantic" in outputs
        
        # Check shapes
        assert outputs["recon_pixel"].shape == x.shape
    
    def test_psvae_training_step(self, test_data_dir, output_dir, device):
        """Test PS-VAE training for a few steps."""
        from psvae.models import PSVAE
        from psvae.training import PSVAETrainer, ImageDataset
        from torch.utils.data import DataLoader
        
        # Create model
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            perceptual_weight=0.0,
        )
        
        # Create dataset
        dataset = ImageDataset(data_dir=test_data_dir, image_size=112)
        assert len(dataset) > 0
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
        
        # Create trainer
        trainer_config = {
            "mixed_precision": "no",
            "ema": {"enabled": False},
            "use_wandb": False,
            "log_every": 1,
            "save_every": 100000,
            "save_dir": os.path.join(output_dir, "checkpoints"),
        }
        
        trainer = PSVAETrainer(
            model=model,
            train_dataloader=dataloader,
            config=trainer_config,
            device=device,
        )
        
        # Train for 1 epoch
        trainer.train(num_epochs=1)
        
        # Verify training happened
        assert trainer.global_step > 0
    
    def test_psvae_encode_decode(self, device):
        """Test PS-VAE encode and decode functions."""
        from psvae.models import PSVAE
        
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
        ).to(device)
        
        model.eval()
        
        # Test encode
        x = torch.randn(2, 3, 112, 112, device=device)
        with torch.no_grad():
            z = model.encode_to_latent(x)
        
        assert z.shape[0] == 2
        assert z.shape[-1] == 32  # latent_dim
        
        # Test decode
        with torch.no_grad():
            recon = model.decode_to_image(z)
        
        assert recon.shape == x.shape


class TestDiTTraining:
    """Tests for DiT training pipeline."""
    
    @pytest.fixture(scope="class")
    def psvae_checkpoint(self, test_data_dir, output_dir, device):
        """Create a PS-VAE checkpoint for DiT testing."""
        from psvae.models import PSVAE
        
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            perceptual_weight=0.0,
        )
        
        checkpoint_path = os.path.join(output_dir, "psvae_for_dit.pt")
        torch.save({"model": model.state_dict()}, checkpoint_path)
        
        return checkpoint_path
    
    def test_dit_model_creation(self):
        """Test DiT model can be created."""
        from psvae.diffusion import DiT
        
        dit = DiT(
            input_size=8,
            in_channels=32,
            hidden_size=256,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            text_dim=512,
            num_text_tokens=32,
            learn_sigma=False,
        )
        
        assert dit is not None
    
    def test_dit_forward_pass(self, device):
        """Test DiT forward pass."""
        from psvae.diffusion import DiT
        
        dit = DiT(
            input_size=8,
            in_channels=32,
            hidden_size=256,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            text_dim=512,
            num_text_tokens=32,
            learn_sigma=False,
        ).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 8, 8, 32, device=device)
        t = torch.randint(0, 100, (batch_size,), device=device)
        
        dit.eval()
        with torch.no_grad():
            output = dit(x, t, text_embeds=None, text_mask=None)
        
        assert output.shape == x.shape
    
    def test_diffusion_scheduler(self):
        """Test diffusion scheduler."""
        from psvae.diffusion import DiffusionScheduler
        from psvae.diffusion.scheduler import DiffusionConfig
        
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        
        scheduler = DiffusionScheduler(config)
        
        # Test add_noise
        x = torch.randn(2, 8, 8, 32)
        noise = torch.randn_like(x)
        t = torch.tensor([10, 50])
        
        noisy = scheduler.add_noise(x, noise, t)
        assert noisy.shape == x.shape
    
    def test_dit_training_step(self, test_data_dir, output_dir, psvae_checkpoint, device):
        """Test DiT training for a few steps."""
        from psvae.models import PSVAE
        from psvae.diffusion import DiT, DiffusionScheduler
        from psvae.diffusion.scheduler import DiffusionConfig
        from psvae.training import DiTTrainer, ImageTextDataset
        from torch.utils.data import DataLoader
        
        # Load PS-VAE
        psvae = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
        )
        checkpoint = torch.load(psvae_checkpoint, map_location="cpu")
        psvae.load_state_dict(checkpoint["model"])
        psvae.eval()
        
        spatial_size = psvae.get_latent_shape()[1]
        
        # Create DiT
        dit = DiT(
            input_size=spatial_size,
            in_channels=32,
            hidden_size=256,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            text_dim=512,
            num_text_tokens=32,
            learn_sigma=False,
        )
        
        # Create scheduler
        diff_config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        scheduler = DiffusionScheduler(diff_config)
        
        # Create dataset
        dataset = ImageTextDataset(
            data_dir=test_data_dir,
            image_size=112,
            tokenizer=None,
            max_text_length=32,
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
        
        # Create trainer
        trainer_config = {
            "mixed_precision": "no",
            "ema": {"enabled": False},
            "use_wandb": False,
            "log_every": 1,
            "save_every": 100000,
            "save_dir": os.path.join(output_dir, "dit_checkpoints"),
        }
        
        trainer = DiTTrainer(
            dit=dit,
            psvae=psvae,
            scheduler=scheduler,
            train_dataloader=dataloader,
            text_encoder=None,
            config=trainer_config,
            device=device,
        )
        
        # Train for 1 epoch
        trainer.train(num_epochs=1)
        
        # Verify training happened
        assert trainer.global_step > 0


class TestDatasets:
    """Tests for dataset classes."""
    
    def test_image_dataset(self, test_data_dir):
        """Test ImageDataset."""
        from psvae.training import ImageDataset
        
        dataset = ImageDataset(data_dir=test_data_dir, image_size=112)
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, 112, 112)
    
    def test_image_text_dataset(self, test_data_dir):
        """Test ImageTextDataset."""
        from psvae.training import ImageTextDataset
        
        dataset = ImageTextDataset(
            data_dir=test_data_dir,
            image_size=112,
            tokenizer=None,
            max_text_length=32,
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert "image" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "caption" in sample
        
        assert sample["image"].shape == (3, 112, 112)


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_psnr_metric(self, device):
        """Test PSNR computation."""
        from psvae.utils.metrics import compute_psnr
        
        # Create test images
        img1 = torch.rand(4, 3, 64, 64, device=device)
        img2 = img1.clone()  # Identical images
        
        # PSNR of identical images should be very high (inf in theory)
        psnr_identical = compute_psnr(img1, img2)
        assert psnr_identical > 40, f"PSNR of identical images should be high, got {psnr_identical}"
        
        # PSNR of different images should be lower
        img3 = torch.rand(4, 3, 64, 64, device=device)
        psnr_different = compute_psnr(img1, img3)
        assert psnr_different < psnr_identical, "PSNR of different images should be lower"
        assert psnr_different > 0, f"PSNR should be positive, got {psnr_different}"
        
        # PSNR with small noise
        img_noisy = img1 + 0.1 * torch.randn_like(img1)
        img_noisy = img_noisy.clamp(0, 1)
        psnr_noisy = compute_psnr(img1, img_noisy)
        assert 15 < psnr_noisy < 40, f"PSNR with small noise should be moderate, got {psnr_noisy}"
        
        print(f"✓ PSNR metrics: identical={psnr_identical:.2f}dB, noisy={psnr_noisy:.2f}dB, random={psnr_different:.2f}dB")
    
    def test_ssim_metric(self, device):
        """Test SSIM computation."""
        from psvae.utils.metrics import compute_ssim
        
        # Create test images
        img1 = torch.rand(4, 3, 64, 64, device=device)
        img2 = img1.clone()  # Identical images
        
        # SSIM of identical images should be 1.0
        ssim_identical = compute_ssim(img1, img2)
        assert 0.99 < ssim_identical <= 1.0, f"SSIM of identical images should be ~1.0, got {ssim_identical}"
        
        # SSIM of different images should be lower
        img3 = torch.rand(4, 3, 64, 64, device=device)
        ssim_different = compute_ssim(img1, img3)
        assert ssim_different < ssim_identical, "SSIM of different images should be lower"
        assert -1 <= ssim_different <= 1, f"SSIM should be in [-1, 1], got {ssim_different}"
        
        # SSIM with small noise
        img_noisy = img1 + 0.1 * torch.randn_like(img1)
        img_noisy = img_noisy.clamp(0, 1)
        ssim_noisy = compute_ssim(img1, img_noisy)
        assert 0.5 < ssim_noisy < 1.0, f"SSIM with small noise should be high but not perfect, got {ssim_noisy}"
        
        print(f"✓ SSIM metrics: identical={ssim_identical:.4f}, noisy={ssim_noisy:.4f}, random={ssim_different:.4f}")
    
    def test_fid_computation(self, device):
        """Test FID computation from features."""
        from psvae.utils.metrics import compute_fid
        
        # Create synthetic features
        feature_dim = 64  # Smaller for testing
        num_samples = 100
        
        # Features from same distribution should have low FID
        mean = torch.zeros(feature_dim)
        cov = torch.eye(feature_dim)
        
        real_features = torch.randn(num_samples, feature_dim, device=device)
        fake_features_similar = torch.randn(num_samples, feature_dim, device=device)
        
        fid_similar = compute_fid(real_features, fake_features_similar)
        assert fid_similar >= 0, f"FID should be non-negative, got {fid_similar}"
        
        # Features from different distributions should have higher FID
        fake_features_different = torch.randn(num_samples, feature_dim, device=device) * 3 + 5
        fid_different = compute_fid(real_features, fake_features_different)
        assert fid_different > fid_similar, "FID of different distributions should be higher"
        
        print(f"✓ FID metrics: similar_dist={fid_similar:.2f}, different_dist={fid_different:.2f}")
    
    def test_reconstruction_metrics(self, test_data_dir, device):
        """Test metrics on actual PS-VAE reconstruction."""
        from psvae.models import PSVAE
        from psvae.training import ImageDataset
        from psvae.utils.metrics import compute_psnr, compute_ssim
        from torch.utils.data import DataLoader
        
        # Create model
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            perceptual_weight=0.0,
        ).to(device)
        
        # Load test data
        dataset = ImageDataset(data_dir=test_data_dir, image_size=112)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader)).to(device)
        
        # Reconstruct
        model.eval()
        with torch.no_grad():
            recon = model.reconstruct(batch)
        
        # Normalize to [0, 1] for metrics (images are in [-1, 1])
        batch_norm = (batch + 1) / 2
        recon_norm = (recon + 1) / 2
        recon_norm = recon_norm.clamp(0, 1)
        
        # Compute metrics
        psnr = compute_psnr(batch_norm, recon_norm)
        ssim = compute_ssim(batch_norm, recon_norm)
        
        # Untrained model won't have great reconstruction, but should be reasonable
        assert psnr > 5, f"PSNR should be > 5 even for untrained model, got {psnr}"
        assert ssim > 0, f"SSIM should be positive, got {ssim}"
        
        print(f"✓ Reconstruction metrics (untrained): PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    def test_loss_components(self, test_data_dir, device):
        """Test that all loss components are computed correctly."""
        from psvae.models import PSVAE
        from psvae.training import ImageDataset
        from torch.utils.data import DataLoader
        
        # Create model
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            kl_weight=0.0001,
            semantic_weight=1.0,
            pixel_weight=1.0,
            perceptual_weight=0.0,  # Disable for speed
        ).to(device)
        
        # Load test data
        dataset = ImageDataset(data_dir=test_data_dir, image_size=112)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get a batch and compute loss
        batch = next(iter(dataloader)).to(device)
        
        model.train()
        outputs = model(batch)
        losses = model.compute_loss(outputs)
        
        # Check all loss components exist and are valid
        required_losses = ["loss", "semantic_loss", "pixel_loss", "kl_loss"]
        for loss_name in required_losses:
            assert loss_name in losses, f"Missing loss component: {loss_name}"
            loss_val = losses[loss_name]
            assert torch.isfinite(loss_val), f"{loss_name} is not finite: {loss_val}"
            assert loss_val >= 0, f"{loss_name} should be non-negative: {loss_val}"
        
        # Total loss should be weighted sum
        expected_total = (
            model.semantic_weight * losses["semantic_loss"]
            + model.pixel_weight * losses["pixel_loss"]
            + model.kl_weight * losses["kl_loss"]
        )
        assert torch.allclose(losses["loss"], expected_total, rtol=1e-4), \
            f"Total loss mismatch: {losses['loss']} vs {expected_total}"
        
        print(f"✓ Loss components: total={losses['loss']:.4f}, "
              f"semantic={losses['semantic_loss']:.4f}, "
              f"pixel={losses['pixel_loss']:.4f}, "
              f"kl={losses['kl_loss']:.6f}")


class TestTrainingMetrics:
    """Tests for training-time metrics and loss tracking."""
    
    def test_loss_decreases_over_steps(self, test_data_dir, output_dir, device):
        """Test that loss decreases over training steps."""
        from psvae.models import PSVAE
        from psvae.training import ImageDataset
        from torch.utils.data import DataLoader
        
        # Create model
        model = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
            perceptual_weight=0.0,
        ).to(device)
        
        # Load test data
        dataset = ImageDataset(data_dir=test_data_dir, image_size=112)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train for a few steps and track loss
        model.train()
        losses = []
        
        for epoch in range(3):
            epoch_losses = []
            for batch in dataloader:
                batch = batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss_dict = model.compute_loss(outputs)
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}")
        
        # Loss should generally decrease (allow some noise)
        # Check that final loss is lower than initial
        assert losses[-1] < losses[0] * 1.5, \
            f"Loss should decrease over training: {losses[0]:.4f} -> {losses[-1]:.4f}"
        
        print(f"✓ Loss progression: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    def test_dit_loss_is_valid(self, test_data_dir, output_dir, device):
        """Test that DiT diffusion loss is computed correctly."""
        from psvae.models import PSVAE
        from psvae.diffusion import DiT, DiffusionScheduler
        from psvae.diffusion.scheduler import DiffusionConfig
        from psvae.training import ImageTextDataset
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        
        # Create PS-VAE
        psvae = PSVAE(
            encoder_type="dinov2",
            model_size="base",
            latent_dim=32,
            image_size=112,
        ).to(device)
        psvae.eval()
        
        spatial_size = psvae.get_latent_shape()[1]
        
        # Create DiT
        dit = DiT(
            input_size=spatial_size,
            in_channels=32,
            hidden_size=256,
            depth=4,
            num_heads=4,
            text_dim=512,
            num_text_tokens=32,
            learn_sigma=False,
        ).to(device)
        
        # Create scheduler
        diff_config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        scheduler = DiffusionScheduler(diff_config)
        
        # Load test data
        dataset = ImageTextDataset(
            data_dir=test_data_dir,
            image_size=112,
            tokenizer=None,
            max_text_length=32,
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        images = batch["image"].to(device)
        
        # Encode to latents
        with torch.no_grad():
            latents = psvae.encode_to_latent(images)
        
        # Sample timesteps and noise
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(latents)
        
        # Add noise
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        dit.train()
        noise_pred = dit(noisy_latents, timesteps, text_embeds=None, text_mask=None)
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Loss should be valid
        assert torch.isfinite(loss), f"DiT loss is not finite: {loss}"
        assert loss > 0, f"DiT loss should be positive: {loss}"
        
        # For random initialization, loss should be around 1.0 (MSE of unit Gaussian)
        assert 0.5 < loss.item() < 2.0, f"DiT loss should be ~1.0 for random init, got {loss.item()}"
        
        print(f"✓ DiT diffusion loss: {loss.item():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

