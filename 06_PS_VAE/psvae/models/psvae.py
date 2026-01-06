"""
PS-VAE: Pixel-Semantic Variational Autoencoder

The full PS-VAE model that combines semantic and pixel-level reconstruction
objectives for optimal generation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal
from einops import rearrange

from .encoder import RepresentationEncoder, SemanticProjector
from .decoder import Decoder, SemanticDecoder, PixelDecoder


class PSVAE(nn.Module):
    """
    Pixel-Semantic VAE (PS-VAE).
    
    Extends S-VAE by adding pixel-level reconstruction loss alongside
    semantic reconstruction. The encoder is unfrozen and jointly optimized
    with both objectives.
    
    Key innovations from the paper:
    1. Compact 96-channel latent space with 16x16 spatial downsampling
    2. KL regularization to prevent off-manifold generation
    3. Joint semantic + pixel reconstruction for fine-grained details
    4. Unfrozen encoder for end-to-end optimization
    
    Architecture:
        Input Image -> Trainable Rep. Encoder -> Semantic Projector -> z
                              |                                        |
                              v                                        v
        Frozen Rep. Encoder --+-> Semantic Loss    Pixel Decoder -> Pixel Loss
                                        ^                    |
                                        |                    v
                              Semantic Decoder <-------------+
    
    Args:
        encoder_type: Type of representation encoder
        model_size: Size of representation encoder
        latent_dim: Dimension of compact latent space
        image_size: Input/output image size
        kl_weight: Weight for KL divergence loss
        semantic_weight: Weight for semantic reconstruction loss
        pixel_weight: Weight for pixel reconstruction loss
        perceptual_weight: Weight for perceptual (LPIPS) loss
    """
    
    def __init__(
        self,
        encoder_type: str = "dinov2",
        model_size: str = "large",
        latent_dim: int = 96,
        image_size: int = 256,
        kl_weight: float = 1e-4,
        semantic_weight: float = 1.0,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.kl_weight = kl_weight
        self.semantic_weight = semantic_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        
        # Trainable representation encoder (unfrozen for PS-VAE)
        self.rep_encoder = RepresentationEncoder(
            encoder_type=encoder_type,
            model_size=model_size,
            freeze=False,  # Unfrozen for joint training
        )
        
        # Frozen copy of encoder for semantic target
        self.frozen_encoder = RepresentationEncoder(
            encoder_type=encoder_type,
            model_size=model_size,
            freeze=True,
        )
        
        # Get spatial size
        self.spatial_size = self.rep_encoder.get_spatial_size(image_size)
        
        # Semantic projector (VAE encoder)
        self.projector = SemanticProjector(
            input_dim=self.rep_encoder.output_dim,
            latent_dim=latent_dim,
        )
        
        # Combined decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            semantic_dim=self.rep_encoder.output_dim,
            output_size=image_size,
            spatial_size=self.spatial_size,
        )
        
        # Perceptual loss (LPIPS)
        self.lpips = None  # Lazy initialization
    
    def _init_lpips(self):
        """Initialize LPIPS loss network."""
        if self.lpips is None:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net="vgg").to(
                    next(self.parameters()).device
                )
                self.lpips.eval()
                for param in self.lpips.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed. Perceptual loss disabled.")
                self.lpips = None
    
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
        # Extract representation features (trainable)
        rep_features = self.rep_encoder(x, return_spatial=True)
        
        # Project to latent distribution
        mu, log_var = self.projector(rep_features)
        
        # Sample latent
        if self.training:
            z = self.projector.reparameterize(mu, log_var)
        else:
            z = mu  # Use mean during inference
        
        if return_distribution:
            return z, mu, log_var
        return z
    
    def decode(
        self,
        z: torch.Tensor,
        return_semantic: bool = True,
        return_pixel: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent vectors.
        
        Args:
            z: Latent vectors [B, h, w, latent_dim]
            return_semantic: Whether to return semantic features
            return_pixel: Whether to return pixel images
        
        Returns:
            Dictionary with decoded outputs
        """
        return self.decoder(z, return_semantic, return_pixel)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all reconstructions.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Dictionary containing all outputs and targets
        """
        # Get target semantic features from frozen encoder
        with torch.no_grad():
            target_semantic = self.frozen_encoder(x, return_spatial=True)
        
        # Encode to latent
        z, mu, log_var = self.encode(x, return_distribution=True)
        
        # Decode
        decoded = self.decode(z)
        
        return {
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "recon_semantic": decoded["semantic"],
            "recon_pixel": decoded["pixel"],
            "target_semantic": target_semantic,
            "target_pixel": x,
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PS-VAE losses.
        
        Loss = semantic_weight * L_semantic + pixel_weight * L_pixel 
               + perceptual_weight * L_perceptual + kl_weight * L_kl
        
        Args:
            outputs: Dictionary from forward pass
        
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        
        # Semantic reconstruction loss
        losses["semantic_loss"] = F.mse_loss(
            outputs["recon_semantic"],
            outputs["target_semantic"],
        )
        
        # Pixel reconstruction loss (L1 + L2)
        l1_loss = F.l1_loss(outputs["recon_pixel"], outputs["target_pixel"])
        l2_loss = F.mse_loss(outputs["recon_pixel"], outputs["target_pixel"])
        losses["pixel_loss"] = 0.5 * l1_loss + 0.5 * l2_loss
        
        # Perceptual loss (LPIPS)
        if self.perceptual_weight > 0:
            self._init_lpips()
            if self.lpips is not None:
                # LPIPS expects images in [-1, 1]
                # Input images are already normalized to [-1, 1] by dataset transform
                # But pixel decoder outputs may be in different range, so we clamp
                recon_clamped = outputs["recon_pixel"].clamp(-1, 1)
                target_clamped = outputs["target_pixel"].clamp(-1, 1)
                losses["perceptual_loss"] = self.lpips(recon_clamped, target_clamped).mean()
            else:
                losses["perceptual_loss"] = torch.tensor(0.0, device=outputs["z"].device)
        else:
            losses["perceptual_loss"] = torch.tensor(0.0, device=outputs["z"].device)
        
        # KL divergence loss
        losses["kl_loss"] = -0.5 * torch.mean(
            1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp()
        )
        
        # Total loss
        losses["loss"] = (
            self.semantic_weight * losses["semantic_loss"]
            + self.pixel_weight * losses["pixel_loss"]
            + self.perceptual_weight * losses["perceptual_loss"]
            + self.kl_weight * losses["kl_loss"]
        )
        
        return losses
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space (inference mode).
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            z: Latent vectors (using mean, no sampling)
        """
        rep_features = self.rep_encoder(x, return_spatial=True)
        mu, _ = self.projector(rep_features)
        return mu
    
    @torch.no_grad()
    def decode_to_image(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images (inference mode).
        
        Args:
            z: Latent vectors [B, h, w, latent_dim]
        
        Returns:
            images: Reconstructed images [B, 3, H, W]
        """
        return self.decoder.pixel_decoder(z)
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images through the autoencoder.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            recon: Reconstructed images [B, 3, H, W]
        """
        z = self.encode_to_latent(x)
        return self.decode_to_image(z)
    
    def get_latent_shape(self, batch_size: int = 1) -> Tuple[int, ...]:
        """Get shape of latent vectors."""
        return (batch_size, self.spatial_size, self.spatial_size, self.latent_dim)


class PSVAELightning(PSVAE):
    """
    PyTorch Lightning wrapper for PS-VAE.
    
    Adds training/validation step methods for easy training.
    """
    
    def training_step(
        self, 
        batch: torch.Tensor, 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Assume first element is images
        else:
            x = batch
        
        outputs = self.forward(x)
        losses = self.compute_loss(outputs)
        
        return losses
    
    def validation_step(
        self, 
        batch: torch.Tensor, 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        with torch.no_grad():
            outputs = self.forward(x)
            losses = self.compute_loss(outputs)
        
        return losses

