"""
Representation Encoder for PS-VAE.

Wraps pretrained vision encoders (DINOv2, SigLIP, CLIP) to extract
semantic features for the generative latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from einops import rearrange


class RepresentationEncoder(nn.Module):
    """
    Wrapper for pretrained representation encoders.
    
    Supports DINOv2, SigLIP, and CLIP encoders that produce
    semantically rich features for generation.
    
    Args:
        encoder_type: Type of encoder ('dinov2', 'siglip', 'clip')
        model_size: Model size variant ('base', 'large', 'giant')
        freeze: Whether to freeze encoder weights
        output_dim: Output feature dimension (None keeps original)
    """
    
    def __init__(
        self,
        encoder_type: Literal["dinov2", "siglip", "clip"] = "dinov2",
        model_size: Literal["base", "large", "giant"] = "large",
        freeze: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.model_size = model_size
        self.freeze = freeze
        
        # Load pretrained encoder
        self.encoder, self.feature_dim = self._load_encoder()
        
        # Optional projection layer
        if output_dim is not None and output_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection = None
            self.output_dim = self.feature_dim
            
        if freeze:
            self._freeze_encoder()
    
    def _load_encoder(self) -> Tuple[nn.Module, int]:
        """Load pretrained encoder based on type and size."""
        
        if self.encoder_type == "dinov2":
            return self._load_dinov2()
        elif self.encoder_type == "siglip":
            return self._load_siglip()
        elif self.encoder_type == "clip":
            return self._load_clip()
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
    
    def _load_dinov2(self) -> Tuple[nn.Module, int]:
        """Load DINOv2 encoder."""
        model_names = {
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "giant": "dinov2_vitg14",
        }
        feature_dims = {
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        
        model_name = model_names[self.model_size]
        encoder = torch.hub.load("facebookresearch/dinov2", model_name)
        
        return encoder, feature_dims[self.model_size]
    
    def _load_siglip(self) -> Tuple[nn.Module, int]:
        """Load SigLIP encoder."""
        try:
            from transformers import SiglipVisionModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        model_names = {
            "base": "google/siglip-base-patch16-256",
            "large": "google/siglip-large-patch16-256",
        }
        feature_dims = {
            "base": 768,
            "large": 1024,
        }
        
        if self.model_size not in model_names:
            raise ValueError(f"SigLIP does not support size: {self.model_size}")
        
        encoder = SiglipVisionModel.from_pretrained(model_names[self.model_size])
        
        return encoder, feature_dims[self.model_size]
    
    def _load_clip(self) -> Tuple[nn.Module, int]:
        """Load CLIP vision encoder."""
        try:
            from transformers import CLIPVisionModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        model_names = {
            "base": "openai/clip-vit-base-patch16",
            "large": "openai/clip-vit-large-patch14",
        }
        feature_dims = {
            "base": 768,
            "large": 1024,
        }
        
        if self.model_size not in model_names:
            raise ValueError(f"CLIP does not support size: {self.model_size}")
        
        encoder = CLIPVisionModel.from_pretrained(model_names[self.model_size])
        
        return encoder, feature_dims[self.model_size]
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for joint training (PS-VAE)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze = False
    
    def forward(
        self, 
        x: torch.Tensor,
        return_spatial: bool = True,
    ) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input images [B, 3, H, W]
            return_spatial: If True, return spatial features [B, h, w, C]
                          If False, return global features [B, C]
        
        Returns:
            features: Extracted features
        """
        if self.freeze:
            self.encoder.eval()
        
        if self.encoder_type == "dinov2":
            features = self._forward_dinov2(x, return_spatial)
        elif self.encoder_type == "siglip":
            features = self._forward_siglip(x, return_spatial)
        elif self.encoder_type == "clip":
            features = self._forward_clip(x, return_spatial)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def _forward_dinov2(
        self, 
        x: torch.Tensor, 
        return_spatial: bool
    ) -> torch.Tensor:
        """Forward pass for DINOv2."""
        if return_spatial:
            # Get patch tokens (exclude CLS token)
            features = self.encoder.forward_features(x)
            
            # Handle different DINOv2 versions - may return dict or tensor
            if isinstance(features, dict):
                # Newer versions return a dict
                if "x_norm_patchtokens" in features:
                    patch_tokens = features["x_norm_patchtokens"]
                elif "x_prenorm" in features:
                    # Fallback to prenorm tokens
                    patch_tokens = features["x_prenorm"][:, 1:]  # Exclude CLS
                else:
                    raise ValueError(f"Unexpected DINOv2 output keys: {features.keys()}")
            else:
                # Older versions return tensor directly (with CLS token)
                patch_tokens = features[:, 1:]  # Exclude CLS token
            
            # Reshape to spatial format
            B, N, C = patch_tokens.shape
            h = w = int(N ** 0.5)
            features = rearrange(patch_tokens, "b (h w) c -> b h w c", h=h, w=w)
        else:
            features = self.encoder(x)
        
        return features
    
    def _forward_siglip(
        self, 
        x: torch.Tensor, 
        return_spatial: bool
    ) -> torch.Tensor:
        """Forward pass for SigLIP."""
        outputs = self.encoder(x)
        
        if return_spatial:
            # Get all patch embeddings
            features = outputs.last_hidden_state
            B, N, C = features.shape
            h = w = int(N ** 0.5)
            features = rearrange(features, "b (h w) c -> b h w c", h=h, w=w)
        else:
            features = outputs.pooler_output
        
        return features
    
    def _forward_clip(
        self, 
        x: torch.Tensor, 
        return_spatial: bool
    ) -> torch.Tensor:
        """Forward pass for CLIP."""
        outputs = self.encoder(x)
        
        if return_spatial:
            # Get all patch embeddings (exclude CLS token)
            features = outputs.last_hidden_state[:, 1:, :]
            B, N, C = features.shape
            h = w = int(N ** 0.5)
            features = rearrange(features, "b (h w) c -> b h w c", h=h, w=w)
        else:
            features = outputs.pooler_output
        
        return features
    
    def get_spatial_size(self, input_size: int = 256) -> int:
        """Get spatial size of output features given input size."""
        patch_sizes = {
            "dinov2": 14,
            "siglip": 16,
            "clip": 14 if self.model_size == "large" else 16,
        }
        patch_size = patch_sizes[self.encoder_type]
        return input_size // patch_size


class SemanticProjector(nn.Module):
    """
    Projects high-dimensional representation features to compact latent space.
    
    This is the encoder part of S-VAE that maps representation features
    to a KL-regularized latent space.
    
    Args:
        input_dim: Input feature dimension from representation encoder
        latent_dim: Latent space dimension (default: 96 as per paper)
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 96,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Mean and variance projections for VAE
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features to latent distribution parameters.
        
        Args:
            x: Input features [B, H, W, C] or [B, N, C]
        
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Flatten spatial dimensions if needed
        original_shape = x.shape
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = rearrange(x, "b h w c -> b (h w) c")
        
        # Encode
        h = self.encoder(x)
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        # Reshape back to spatial if needed
        if len(original_shape) == 4:
            mu = rearrange(mu, "b (h w) c -> b h w c", h=H, w=W)
            log_var = rearrange(log_var, "b (h w) c -> b h w c", h=H, w=W)
        
        return mu, log_var
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

