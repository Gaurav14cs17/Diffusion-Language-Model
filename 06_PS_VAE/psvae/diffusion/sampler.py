"""
Samplers for diffusion models.

Implements DDPM and DDIM sampling for PS-VAE generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from tqdm import tqdm

from .scheduler import DiffusionScheduler


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler.
    
    Implements the standard DDPM sampling algorithm.
    
    Args:
        scheduler: Diffusion scheduler
        model: Denoising model (DiT)
    """
    
    def __init__(
        self,
        scheduler: DiffusionScheduler,
        model: nn.Module,
    ):
        self.scheduler = scheduler
        self.model = model
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        device: Optional[torch.device] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDPM.
        
        Args:
            shape: Shape of samples to generate [B, H, W, C]
            text_embeds: Text embeddings for conditioning
            text_mask: Text attention mask
            cfg_scale: Classifier-free guidance scale
            device: Device to use (defaults to model device)
            progress: Whether to show progress bar
        
        Returns:
            samples: Generated samples
        """
        # Default device
        if device is None:
            device = next(self.model.parameters()).device
        
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Sampling loop
        timesteps = list(range(self.scheduler.num_timesteps))[::-1]
        if progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            if cfg_scale > 1.0 and text_embeds is not None:
                noise_pred = self.model.forward_with_cfg(
                    x, t_batch, text_embeds, cfg_scale, text_mask
                )
            else:
                noise_pred = self.model(x, t_batch, text_embeds, text_mask)
            
            # Handle learned variance
            if noise_pred.shape[-1] == shape[-1] * 2:
                noise_pred, _ = noise_pred.chunk(2, dim=-1)
            
            # Get x_0 prediction
            x_start = self.scheduler.predict_start_from_noise(x, t_batch, noise_pred)
            
            # Clip x_0
            if self.scheduler.config.clip_sample:
                x_start = x_start.clamp(
                    -self.scheduler.config.clip_sample_range,
                    self.scheduler.config.clip_sample_range,
                )
            
            # Get posterior
            model_mean, _, model_log_variance = self.scheduler.q_posterior(
                x_start=x_start, x_t=x, t=t_batch
            )
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            else:
                x = model_mean
        
        return x


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    Implements accelerated sampling with fewer steps.
    
    Args:
        scheduler: Diffusion scheduler
        model: Denoising model (DiT)
    """
    
    def __init__(
        self,
        scheduler: DiffusionScheduler,
        model: nn.Module,
    ):
        self.scheduler = scheduler
        self.model = model
    
    def _get_timesteps(
        self,
        num_inference_steps: int,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Get timesteps for DDIM sampling."""
        # Uniform spacing
        step_ratio = self.scheduler.num_timesteps // num_inference_steps
        timesteps = (
            (torch.arange(0, num_inference_steps) * step_ratio)
            .round()
            .long()
            .flip(0)
        )
        
        # Apply strength for img2img
        if strength < 1.0:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            timesteps = timesteps[-init_timestep:]
        
        return timesteps
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        progress: bool = True,
        init_latents: Optional[torch.Tensor] = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.
        
        Args:
            shape: Shape of samples to generate [B, H, W, C]
            text_embeds: Text embeddings for conditioning
            text_mask: Text attention mask
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            device: Device to use
            progress: Whether to show progress bar
            init_latents: Initial latents for img2img
            strength: Denoising strength for img2img
        
        Returns:
            samples: Generated samples
        """
        # Default device
        if device is None:
            device = next(self.model.parameters()).device
        
        batch_size = shape[0]
        
        # Get timesteps
        timesteps = self._get_timesteps(num_inference_steps, strength).to(device)
        
        # Initialize latents
        if init_latents is not None and strength < 1.0:
            # Add noise to init_latents for img2img
            noise = torch.randn(shape, device=device)
            x = self.scheduler.add_noise(init_latents, noise, timesteps[:1].expand(batch_size))
        else:
            x = torch.randn(shape, device=device)
        
        # Sampling loop
        if progress:
            timesteps_iter = tqdm(timesteps, desc="DDIM Sampling")
        else:
            timesteps_iter = timesteps
        
        for i, t in enumerate(timesteps_iter):
            t_batch = t.expand(batch_size)
            
            # Predict noise
            if cfg_scale > 1.0 and text_embeds is not None:
                noise_pred = self.model.forward_with_cfg(
                    x, t_batch, text_embeds, cfg_scale, text_mask
                )
            else:
                noise_pred = self.model(x, t_batch, text_embeds, text_mask)
            
            # Handle learned variance
            if noise_pred.shape[-1] == shape[-1] * 2:
                noise_pred, _ = noise_pred.chunk(2, dim=-1)
            
            # Get x_0 prediction
            x_start = self.scheduler.predict_start_from_noise(x, t_batch, noise_pred)
            
            # Clip x_0
            if self.scheduler.config.clip_sample:
                x_start = x_start.clamp(
                    -self.scheduler.config.clip_sample_range,
                    self.scheduler.config.clip_sample_range,
                )
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                t_next = torch.tensor(0, device=device)
            
            x = self._ddim_step(x, x_start, noise_pred, t, t_next, eta)
        
        return x
    
    def _ddim_step(
        self,
        x_t: torch.Tensor,
        x_start: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """
        Perform one DDIM step.
        
        Args:
            x_t: Current noisy sample
            x_start: Predicted x_0
            noise_pred: Predicted noise
            t: Current timestep
            t_next: Next timestep
            eta: DDIM eta parameter
        
        Returns:
            x_next: Sample at next timestep
        """
        device = x_t.device
        alpha_cumprod_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_cumprod_t_next = self.scheduler.alphas_cumprod[t_next].to(device) if t_next >= 0 else torch.tensor(1.0, device=device)
        
        # Compute sigma
        sigma = eta * torch.sqrt(
            (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)
            * (1 - alpha_cumprod_t / alpha_cumprod_t_next)
        )
        
        # Compute direction pointing to x_t
        pred_direction = torch.sqrt(1 - alpha_cumprod_t_next - sigma**2) * noise_pred
        
        # Compute x_{t-1}
        x_next = torch.sqrt(alpha_cumprod_t_next) * x_start + pred_direction
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_next = x_next + sigma * noise
        
        return x_next
    
    @torch.no_grad()
    def sample_edit(
        self,
        source_latents: torch.Tensor,
        source_text_embeds: torch.Tensor,
        target_text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        edit_strength: float = 0.8,
        device: Optional[torch.device] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Edit images using DDIM inversion and denoising.
        
        Args:
            source_latents: Source image latents
            source_text_embeds: Source text embeddings
            target_text_embeds: Target text embeddings
            text_mask: Text attention mask
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            edit_strength: How much to change the image
            device: Device to use
            progress: Whether to show progress bar
        
        Returns:
            edited: Edited image latents
        """
        # Default device
        if device is None:
            device = next(self.model.parameters()).device
        
        # DDIM inversion to get noise
        inverted = self._ddim_inversion(
            source_latents,
            source_text_embeds,
            text_mask,
            num_inference_steps,
            device,
            progress,
        )
        
        # Denoise with target text
        edited = self.sample(
            source_latents.shape,
            target_text_embeds,
            text_mask,
            cfg_scale,
            num_inference_steps,
            eta=0.0,
            device=device,
            progress=progress,
            init_latents=inverted,
            strength=edit_strength,
        )
        
        return edited
    
    @torch.no_grad()
    def _ddim_inversion(
        self,
        latents: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        num_inference_steps: int,
        device: torch.device,
        progress: bool,
    ) -> torch.Tensor:
        """
        DDIM inversion to find noise that reconstructs the image.
        
        Args:
            latents: Image latents
            text_embeds: Text embeddings
            text_mask: Text attention mask
            num_inference_steps: Number of inversion steps
            device: Device to use
            progress: Whether to show progress bar
        
        Returns:
            inverted: Inverted noise latents
        """
        batch_size = latents.shape[0]
        
        # Get timesteps (forward direction for inversion)
        timesteps = self._get_timesteps(num_inference_steps).flip(0).to(device)
        
        x = latents
        
        if progress:
            timesteps_iter = tqdm(timesteps, desc="DDIM Inversion")
        else:
            timesteps_iter = timesteps
        
        for i, t in enumerate(timesteps_iter):
            t_batch = t.expand(batch_size)
            
            # Predict noise
            noise_pred = self.model(x, t_batch, text_embeds, text_mask)
            
            # Handle learned variance
            if noise_pred.shape[-1] == latents.shape[-1] * 2:
                noise_pred, _ = noise_pred.chunk(2, dim=-1)
            
            # Get next timestep
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                t_next = torch.tensor(self.scheduler.num_timesteps - 1, device=device)
            
            # Inverse DDIM step
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t].to(device)
            alpha_cumprod_t_next = self.scheduler.alphas_cumprod[t_next].to(device)
            
            x_start = self.scheduler.predict_start_from_noise(x, t_batch, noise_pred)
            
            # Compute x_{t+1}
            x = (
                torch.sqrt(alpha_cumprod_t_next) * x_start
                + torch.sqrt(1 - alpha_cumprod_t_next) * noise_pred
            )
        
        return x

