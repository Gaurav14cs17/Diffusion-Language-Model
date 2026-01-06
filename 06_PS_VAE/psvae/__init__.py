"""
PS-VAE: Pixel-Semantic Variational Autoencoder

A PyTorch implementation of "Both Semantics and Reconstruction Matter: 
Making Representation Encoders Ready for Text-to-Image Generation and Editing"

Paper: https://arxiv.org/pdf/2512.17909
"""

from .models import PSVAE, SVAE, RepresentationEncoder
from .diffusion import DiT, DiffusionScheduler

__version__ = "0.1.0"
__all__ = ["PSVAE", "SVAE", "RepresentationEncoder", "DiT", "DiffusionScheduler"]

