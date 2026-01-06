"""
Decoder components for PS-VAE.

Includes both semantic decoder (for S-VAE) and pixel decoder (for PS-VAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from einops import rearrange


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvResidualBlock(nn.Module):
    """Convolutional residual block for spatial processing."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class SemanticDecoder(nn.Module):
    """
    Decoder for reconstructing semantic features from latent space.
    
    Used in S-VAE to reconstruct the original representation encoder features.
    
    Args:
        latent_dim: Latent space dimension
        output_dim: Output feature dimension (representation encoder dim)
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        latent_dim: int = 96,
        output_dim: int = 1024,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        # Build decoder layers
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Residual refinement blocks
        self.refinement = nn.Sequential(
            ResidualBlock(output_dim),
            ResidualBlock(output_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to semantic features.
        
        Args:
            z: Latent vectors [B, H, W, latent_dim] or [B, N, latent_dim]
        
        Returns:
            features: Reconstructed semantic features
        """
        features = self.decoder(z)
        features = self.refinement(features)
        return features


class PixelDecoder(nn.Module):
    """
    Decoder for reconstructing pixel-level images from latent space.
    
    Uses a convolutional architecture with upsampling to generate
    high-resolution images from compact latent representations.
    
    Args:
        latent_dim: Latent space dimension (channels)
        output_channels: Output image channels (3 for RGB)
        base_channels: Base channel count for decoder
        channel_multipliers: Channel multipliers for each resolution level
        num_res_blocks: Number of residual blocks per level
        spatial_size: Spatial size of latent features
        output_size: Target output image size
    """
    
    def __init__(
        self,
        latent_dim: int = 96,
        output_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (4, 4, 2, 1),
        num_res_blocks: int = 2,
        spatial_size: int = 16,
        output_size: int = 256,
    ):
        super().__init__()
        
        self.spatial_size = spatial_size
        self.output_size = output_size
        
        # Calculate number of upsampling stages needed
        num_upsample = 0
        size = spatial_size
        while size < output_size:
            size *= 2
            num_upsample += 1
        
        # Ensure we have enough channel multipliers
        if len(channel_multipliers) < num_upsample:
            channel_multipliers = channel_multipliers + (1,) * (num_upsample - len(channel_multipliers))
        
        # Initial projection from latent space
        init_channels = base_channels * channel_multipliers[0]
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_dim, init_channels, 3, padding=1),
            nn.GroupNorm(32, init_channels),
            nn.SiLU(),
        )
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        in_channels = init_channels
        
        for i in range(num_upsample):
            out_channels = base_channels * channel_multipliers[min(i + 1, len(channel_multipliers) - 1)]
            
            # Residual blocks
            for _ in range(num_res_blocks):
                self.blocks.append(ConvResidualBlock(in_channels, in_channels))
            
            # Upsample
            self.blocks.append(Upsample(in_channels))
            
            # Channel reduction
            if in_channels != out_channels:
                self.blocks.append(ConvResidualBlock(in_channels, out_channels))
                in_channels = out_channels
        
        # Final output layers
        self.output_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, output_channels, 3, padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to pixel images.
        
        Args:
            z: Latent vectors [B, H, W, C] or [B, C, H, W]
        
        Returns:
            images: Reconstructed images [B, 3, output_size, output_size]
        """
        # Ensure channel-first format [B, C, H, W]
        # If last dim equals spatial_size, it's channel-first; otherwise channel-last
        if z.shape[-1] != z.shape[-2]:  # [B, H, W, C] format (H==W but C != H)
            z = rearrange(z, "b h w c -> b c h w")
        
        # Initial projection
        h = self.input_proj(z)
        
        # Decoder blocks
        for block in self.blocks:
            h = block(h)
        
        # Final output
        out = self.output_layers(h)
        
        # Ensure correct output size
        if out.shape[-1] != self.output_size:
            out = F.interpolate(out, size=self.output_size, mode="bilinear", align_corners=False)
        
        return out


class Decoder(nn.Module):
    """
    Combined decoder for PS-VAE.
    
    Includes both semantic and pixel decoders for joint reconstruction.
    
    Args:
        latent_dim: Latent space dimension
        semantic_dim: Semantic feature dimension
        output_size: Target output image size
        spatial_size: Spatial size of latent features
    """
    
    def __init__(
        self,
        latent_dim: int = 96,
        semantic_dim: int = 1024,
        output_size: int = 256,
        spatial_size: int = 16,
    ):
        super().__init__()
        
        self.semantic_decoder = SemanticDecoder(
            latent_dim=latent_dim,
            output_dim=semantic_dim,
        )
        
        self.pixel_decoder = PixelDecoder(
            latent_dim=latent_dim,
            output_size=output_size,
            spatial_size=spatial_size,
        )
    
    def forward(
        self, 
        z: torch.Tensor,
        return_semantic: bool = True,
        return_pixel: bool = True,
    ) -> dict:
        """
        Decode latent vectors.
        
        Args:
            z: Latent vectors
            return_semantic: Whether to return semantic features
            return_pixel: Whether to return pixel images
        
        Returns:
            Dictionary with 'semantic' and/or 'pixel' keys
        """
        outputs = {}
        
        if return_semantic:
            outputs["semantic"] = self.semantic_decoder(z)
        
        if return_pixel:
            outputs["pixel"] = self.pixel_decoder(z)
        
        return outputs

