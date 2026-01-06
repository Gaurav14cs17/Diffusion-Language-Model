#!/usr/bin/env python3
"""
Simple example demonstrating PS-VAE usage.

This example shows how to:
1. Create a PS-VAE model
2. Encode images to latent space
3. Decode latents back to images
4. Visualize reconstructions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from torchvision import transforms

from psvae import PSVAE


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create PS-VAE model
    # Note: This will download DINOv2 weights on first run
    print("Creating PS-VAE model...")
    model = PSVAE(
        encoder_type="dinov2",
        model_size="large",  # Use "base" for faster inference
        latent_dim=96,
        image_size=256,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create a random test image (replace with real image loading)
    print("\nGenerating random test image...")
    test_image = torch.randn(1, 3, 256, 256, device=device)
    
    # Alternatively, load a real image:
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # image = Image.open("your_image.jpg").convert("RGB")
    # test_image = transform(image).unsqueeze(0).to(device)
    
    # Encode to latent space
    print("Encoding to latent space...")
    with torch.no_grad():
        latent = model.encode_to_latent(test_image)
    
    print(f"Latent shape: {latent.shape}")  # Expected: [1, 16, 16, 96]
    print(f"Latent stats: min={latent.min():.3f}, max={latent.max():.3f}, mean={latent.mean():.3f}")
    
    # Decode back to image
    print("Decoding to image...")
    with torch.no_grad():
        reconstructed = model.decode_to_image(latent)
    
    print(f"Reconstructed shape: {reconstructed.shape}")  # Expected: [1, 3, 256, 256]
    
    # Compute reconstruction error
    mse = torch.mean((test_image - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Full forward pass with losses
    print("\nRunning full forward pass...")
    with torch.no_grad():
        outputs = model(test_image)
        losses = model.compute_loss(outputs)
    
    print("Losses:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")
    
    print("\nExample complete!")
    
    # Latent space exploration
    print("\n--- Latent Space Exploration ---")
    
    # Generate two random latents
    z1 = torch.randn(1, 16, 16, 96, device=device) * 0.5
    z2 = torch.randn(1, 16, 16, 96, device=device) * 0.5
    
    # Interpolate
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"Interpolating between two random latents...")
    
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = model.decode_to_image(z_interp)
        print(f"  alpha={alpha:.2f}: image range [{img.min():.3f}, {img.max():.3f}]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

