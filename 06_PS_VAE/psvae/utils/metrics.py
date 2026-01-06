"""
Evaluation metrics for PS-VAE.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


def compute_lpips(
    images1: torch.Tensor,
    images2: torch.Tensor,
    net: str = "vgg",
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two sets of images.
    
    Args:
        images1: First set of images [B, C, H, W] in range [-1, 1] or [0, 1]
        images2: Second set of images [B, C, H, W]
        net: Network to use ('vgg', 'alex', 'squeeze')
        device: Device to use
    
    Returns:
        Mean LPIPS score
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips is required. Install with: pip install lpips")
    
    if device is None:
        device = images1.device
    
    # Initialize LPIPS
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    
    # Normalize to [-1, 1] if needed
    if images1.min() >= 0:
        images1 = images1 * 2 - 1
        images2 = images2 * 2 - 1
    
    with torch.no_grad():
        distances = loss_fn(images1, images2)
    
    return distances.mean().item()


def compute_psnr(
    images1: torch.Tensor,
    images2: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two sets of images.
    
    Args:
        images1: First set of images [B, C, H, W]
        images2: Second set of images [B, C, H, W]
        max_val: Maximum pixel value
    
    Returns:
        Mean PSNR in dB
    """
    mse = torch.mean((images1 - images2) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.mean().item()


def compute_ssim(
    images1: torch.Tensor,
    images2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> float:
    """
    Compute Structural Similarity Index between two sets of images.
    
    Args:
        images1: First set of images [B, C, H, W]
        images2: Second set of images [B, C, H, W]
        window_size: Size of the Gaussian window
        size_average: Whether to average over batch
    
    Returns:
        Mean SSIM score
    """
    C = images1.shape[1]
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=images1.device)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size)
    
    # Constants
    K1, K2 = 0.01, 0.03
    L = 1.0  # Dynamic range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Compute means
    mu1 = torch.nn.functional.conv2d(images1, window, padding=window_size//2, groups=C)
    mu2 = torch.nn.functional.conv2d(images2, window, padding=window_size//2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = torch.nn.functional.conv2d(images1 ** 2, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(images2 ** 2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(images1 * images2, window, padding=window_size//2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def compute_fid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> float:
    """
    Compute Fréchet Inception Distance between two sets of features.
    
    Args:
        real_features: Features from real images [N, D]
        fake_features: Features from generated images [M, D]
    
    Returns:
        FID score
    """
    # Compute statistics
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)
    
    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)
    
    # Compute FID
    diff = mu1 - mu2
    
    # Matrix square root using eigendecomposition
    covmean = _matrix_sqrt(sigma1 @ sigma2)
    
    if torch.isnan(covmean).any():
        # Fallback: add small epsilon to diagonal
        eps = 1e-6
        sigma1 = sigma1 + eps * torch.eye(sigma1.shape[0], device=sigma1.device)
        sigma2 = sigma2 + eps * torch.eye(sigma2.shape[0], device=sigma2.device)
        covmean = _matrix_sqrt(sigma1 @ sigma2)
    
    fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid.item()


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Compute matrix square root using eigendecomposition."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    return eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T


class InceptionV3Features(nn.Module):
    """
    Extract features from InceptionV3 for FID computation.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
        except ImportError:
            raise ImportError("torchvision is required for FID computation")
        
        # Load pretrained InceptionV3
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Identity()  # Remove final FC layer
        
        if device is not None:
            self.inception = self.inception.to(device)
        
        self.inception.eval()
        
        # Freeze
        for param in self.inception.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Images [B, 3, H, W] in range [0, 1]
        
        Returns:
            Features [B, 2048]
        """
        # Resize to 299x299
        if x.shape[-1] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        return self.inception(x)


def compute_fid_from_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute FID directly from images.
    
    Args:
        real_images: Real images [N, 3, H, W] in range [0, 1]
        fake_images: Generated images [M, 3, H, W]
        batch_size: Batch size for feature extraction
        device: Device to use
    
    Returns:
        FID score
    """
    if device is None:
        device = real_images.device
    
    # Initialize feature extractor
    feature_extractor = InceptionV3Features(device)
    
    # Extract features
    def extract_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            feat = feature_extractor(batch)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)
    
    real_features = extract_features(real_images)
    fake_features = extract_features(fake_images)
    
    return compute_fid(real_features, fake_features)

