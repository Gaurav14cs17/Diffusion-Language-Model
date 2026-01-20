"""
Diffusion scheduler for PS-VAE.

Implements noise scheduling and diffusion process utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: Literal["linear", "cosine", "scaled_linear"] = "scaled_linear"
    prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0


class DiffusionScheduler(nn.Module):
    """
    Diffusion noise scheduler.
    
    Handles the forward diffusion process (adding noise) and provides
    utilities for the reverse process (denoising).
    
    Args:
        config: Diffusion configuration
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = DiffusionConfig()
        
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.prediction_type = config.prediction_type
        
        # Compute beta schedule
        betas = self._get_betas(
            config.beta_schedule,
            config.beta_start,
            config.beta_end,
            config.num_timesteps,
        )
        
        # Compute alpha schedule
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # Register buffers
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(alphas_cumprod_prev, dtype=torch.float32))
        
        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.tensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32)
        )
    
    def _get_betas(
        self,
        schedule: str,
        beta_start: float,
        beta_end: float,
        num_timesteps: int,
    ) -> np.ndarray:
        """Get beta schedule."""
        if schedule == "linear":
            return np.linspace(beta_start, beta_end, num_timesteps)
        
        elif schedule == "scaled_linear":
            # Scaled linear schedule (used in Stable Diffusion)
            return np.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        
        elif schedule == "cosine":
            # Cosine schedule
            steps = num_timesteps + 1
            s = 0.008
            x = np.linspace(0, num_timesteps, steps)
            alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples (forward diffusion process).
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        
        Args:
            x_start: Original samples [B, ...]
            noise: Noise to add [B, ...]
            timesteps: Timesteps [B]
        
        Returns:
            x_t: Noisy samples [B, ...]
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def get_velocity(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get velocity for v-prediction.
        
        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0
        
        Args:
            x_start: Original samples
            noise: Noise
            timesteps: Timesteps
        
        Returns:
            velocity: Velocity targets
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        
        return sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_start
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - sqrt(1 - alpha_cumprod) * noise) / sqrt(alpha_cumprod)
        
        Args:
            x_t: Noisy samples
            t: Timesteps
            noise: Predicted noise
        
        Returns:
            x_0: Predicted original samples
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (x_t - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod
    
    def predict_start_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted velocity.
        
        Args:
            x_t: Noisy samples
            t: Timesteps
            v: Predicted velocity
        
        Returns:
            x_0: Predicted original samples
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return sqrt_alpha_cumprod * x_t - sqrt_one_minus_alpha_cumprod * v
    
    def predict_noise_from_start(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from x_t and x_0.
        
        Args:
            x_t: Noisy samples
            t: Timesteps
            x_start: Original samples
        
        Returns:
            noise: Predicted noise
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (x_t - sqrt_alpha_cumprod * x_start) / sqrt_one_minus_alpha_cumprod
    
    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Original samples
            x_t: Noisy samples at timestep t
            t: Timesteps
        
        Returns:
            posterior_mean: Mean of posterior
            posterior_variance: Variance of posterior
            posterior_log_variance: Log variance of posterior
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from a at timesteps t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        
        # Reshape for broadcasting
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get signal-to-noise ratio for timesteps.
        
        SNR = alpha_cumprod / (1 - alpha_cumprod)
        
        Args:
            timesteps: Timesteps
        
        Returns:
            snr: Signal-to-noise ratio
        """
        alpha_cumprod = self.alphas_cumprod[timesteps]
        return alpha_cumprod / (1 - alpha_cumprod)

