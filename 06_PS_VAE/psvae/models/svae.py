"""
S-VAE: Semantic Variational Autoencoder

Maps representation encoder features to a compact, KL-regularized latent space.
This addresses the off-manifold generation issue identified in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from einops import rearrange

from .encoder import RepresentationEncoder, SemanticProjector
from .decoder import SemanticDecoder


class SVAE(nn.Module):
    """
    Semantic VAE (S-VAE).
    
    Projects frozen representation features into a compact, KL-regularized
    latent space via a semantic autoencoder. This constraint eliminates
    off-manifold outliers during generation.
    
    Architecture:
        Input Image -> Frozen Rep. Encoder -> Semantic Projector -> z (96-dim)
                                                                     |
                                                                     v
        Reconstructed Features <- Semantic Decoder <-----------------+
    
    Args:
        encoder_type: Type of representation encoder
        model_size: Size of representation encoder
        latent_dim: Dimension of compact latent space (96 as per paper)
        kl_weight: Weight for KL divergence loss
    """
    
    def __init__(
        self,
        encoder_type: str = "dinov2",
        model_size: str = "large",
        latent_dim: int = 96,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Frozen representation encoder
        self.rep_encoder = RepresentationEncoder(
            encoder_type=encoder_type,
            model_size=model_size,
            freeze=True,  # Frozen for S-VAE
        )
        
        # Semantic projector (encoder part of VAE)
        self.projector = SemanticProjector(
            input_dim=self.rep_encoder.output_dim,
            latent_dim=latent_dim,
        )
        
        # Semantic decoder
        self.decoder = SemanticDecoder(
            latent_dim=latent_dim,
            output_dim=self.rep_encoder.output_dim,
        )
    
    def encode(
        self, 
        x: torch.Tensor,
        return_distribution: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.
        
        Args:
            x: Input images [B, 3, H, W]
            return_distribution: If True, return mu, log_var along with z
        
        Returns:
            z: Latent vectors [B, h, w, latent_dim]
            (optional) mu, log_var: Distribution parameters
        """
        # Extract representation features
        with torch.no_grad():
            rep_features = self.rep_encoder(x, return_spatial=True)
        
        # Project to latent distribution
        mu, log_var = self.projector(rep_features)
        
        # Sample latent
        z = self.projector.reparameterize(mu, log_var)
        
        if return_distribution:
            return z, mu, log_var
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to semantic features.
        
        Args:
            z: Latent vectors [B, h, w, latent_dim]
        
        Returns:
            features: Reconstructed semantic features [B, h, w, C]
        """
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction and KL loss computation.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Dictionary containing:
                - z: Latent vectors
                - recon_features: Reconstructed features
                - target_features: Original representation features
                - mu: Mean of latent distribution
                - log_var: Log variance of latent distribution
        """
        # Get target representation features
        with torch.no_grad():
            target_features = self.rep_encoder(x, return_spatial=True)
        
        # Encode to latent
        z, mu, log_var = self.encode(x, return_distribution=True)
        
        # Decode to semantic features
        recon_features = self.decode(z)
        
        return {
            "z": z,
            "recon_features": recon_features,
            "target_features": target_features,
            "mu": mu,
            "log_var": log_var,
        }
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute S-VAE losses.
        
        Args:
            outputs: Dictionary from forward pass
        
        Returns:
            Dictionary containing:
                - loss: Total loss
                - recon_loss: Semantic reconstruction loss
                - kl_loss: KL divergence loss
        """
        # Semantic reconstruction loss (MSE)
        recon_loss = F.mse_loss(
            outputs["recon_features"], 
            outputs["target_features"]
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(
            1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp()
        )
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
    
    def get_latent_spatial_size(self, input_size: int = 256) -> int:
        """Get spatial size of latent features given input size."""
        return self.rep_encoder.get_spatial_size(input_size)
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space (inference mode).
        
        Uses mean of distribution instead of sampling.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            z: Latent vectors (using mean, no sampling)
        """
        rep_features = self.rep_encoder(x, return_spatial=True)
        mu, _ = self.projector(rep_features)
        return mu

