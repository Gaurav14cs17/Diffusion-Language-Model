"""
Visualization utilities for PS-VAE.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def tensor_to_pil(tensor: torch.Tensor) -> "Image.Image":
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor [C, H, W] or [H, W, C] in range [-1, 1] or [0, 1]
    
    Returns:
        PIL Image
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization")
    
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Ensure [C, H, W] format
    if tensor.shape[-1] in [1, 3, 4]:
        tensor = tensor.permute(2, 0, 1)
    
    # Normalize to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    array = (tensor * 255).byte().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    return Image.fromarray(array)


def save_image_grid(
    images: Union[torch.Tensor, List[torch.Tensor]],
    path: Union[str, Path],
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
) -> None:
    """
    Save a grid of images.
    
    Args:
        images: Tensor [B, C, H, W] or list of tensors
        path: Output path
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize images to [0, 1]
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization")
    
    if isinstance(images, list):
        images = torch.stack(images)
    
    B, C, H, W = images.shape
    
    # Normalize
    if normalize and images.min() < 0:
        images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Calculate grid size
    ncol = (B + nrow - 1) // nrow
    
    # Create grid
    grid_h = ncol * H + (ncol + 1) * padding
    grid_w = nrow * W + (nrow + 1) * padding
    grid = torch.ones(C, grid_h, grid_w)
    
    for idx in range(B):
        row = idx // nrow
        col = idx % nrow
        
        y = row * (H + padding) + padding
        x = col * (W + padding) + padding
        
        grid[:, y:y+H, x:x+W] = images[idx]
    
    # Save
    pil_image = tensor_to_pil(grid)
    pil_image.save(path)


def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    path: Optional[Union[str, Path]] = None,
) -> Optional["Image.Image"]:
    """
    Visualize original and reconstructed images side by side.
    
    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        path: Optional path to save the visualization
    
    Returns:
        PIL Image if path is None, otherwise None
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization")
    
    B = original.shape[0]
    
    # Interleave original and reconstructed
    combined = torch.stack([original, reconstructed], dim=1)
    combined = combined.view(B * 2, *original.shape[1:])
    
    # Create grid with pairs
    if path is not None:
        save_image_grid(combined, path, nrow=2)
        return None
    else:
        # Return single comparison for first image
        pair = torch.cat([original[0], reconstructed[0]], dim=2)
        return tensor_to_pil(pair)


def visualize_latents(
    latents: torch.Tensor,
    path: Optional[Union[str, Path]] = None,
    num_channels: int = 16,
) -> Optional["Image.Image"]:
    """
    Visualize latent space as channel heatmaps.
    
    Args:
        latents: Latent tensor [B, H, W, C] or [B, C, H, W]
        path: Optional path to save the visualization
        num_channels: Number of channels to visualize
    
    Returns:
        PIL Image if path is None, otherwise None
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization")
    
    # Ensure [B, C, H, W] format
    if latents.shape[-1] < latents.shape[1]:
        latents = latents.permute(0, 3, 1, 2)
    
    # Take first sample
    latent = latents[0]
    C, H, W = latent.shape
    
    # Select channels to visualize
    num_channels = min(num_channels, C)
    indices = torch.linspace(0, C - 1, num_channels).long()
    selected = latent[indices]
    
    # Normalize each channel
    selected = selected - selected.view(num_channels, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    selected = selected / (selected.view(num_channels, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1) + 1e-8)
    
    # Upsample for visibility
    selected = F.interpolate(
        selected.unsqueeze(0), 
        scale_factor=8, 
        mode="nearest"
    )[0]
    
    # Convert to RGB (use colormap)
    rgb_channels = []
    for i in range(num_channels):
        channel = selected[i]
        # Simple colormap: blue -> red
        r = channel
        g = torch.zeros_like(channel)
        b = 1 - channel
        rgb = torch.stack([r, g, b], dim=0)
        rgb_channels.append(rgb)
    
    rgb_tensor = torch.stack(rgb_channels)
    
    if path is not None:
        save_image_grid(rgb_tensor, path, nrow=4, normalize=False)
        return None
    else:
        return tensor_to_pil(rgb_tensor[0])


def create_interpolation_video(
    model,
    start_latent: torch.Tensor,
    end_latent: torch.Tensor,
    num_frames: int = 30,
    output_path: Union[str, Path] = "interpolation.gif",
) -> None:
    """
    Create an interpolation video between two latents.
    
    Args:
        model: PS-VAE model
        start_latent: Starting latent [1, H, W, C]
        end_latent: Ending latent [1, H, W, C]
        num_frames: Number of interpolation frames
        output_path: Output path for GIF
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization")
    
    frames = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Linear interpolation
        latent = (1 - alpha) * start_latent + alpha * end_latent
        
        # Decode
        with torch.no_grad():
            image = model.decode_to_image(latent)
        
        # Convert to PIL
        frame = tensor_to_pil(image[0])
        frames.append(frame)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

