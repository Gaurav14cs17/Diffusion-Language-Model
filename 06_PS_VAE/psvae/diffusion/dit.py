"""
Diffusion Transformer (DiT) for PS-VAE.

Implements a DiT architecture adapted for the PS-VAE latent space,
enabling text-to-image generation and editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: Timesteps [B]
        
        Returns:
            embeddings: Timestep embeddings [B, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


class TextConditionEmbedding(nn.Module):
    """
    Text condition embedding module.
    
    Projects text embeddings from a text encoder (e.g., T5, CLIP)
    to the DiT hidden dimension.
    """
    
    def __init__(
        self,
        text_dim: int = 4096,  # T5-XXL dimension
        hidden_dim: int = 1152,
        num_tokens: int = 77,
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable position embeddings for text tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
    
    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings.
        
        Args:
            text_embeds: Text embeddings [B, num_tokens, text_dim]
        
        Returns:
            projected: Projected embeddings [B, num_tokens, hidden_dim]
        """
        B, N, _ = text_embeds.shape
        pos = self.pos_embed[:, :N, :]
        return self.proj(text_embeds) + pos


class Attention(nn.Module):
    """Multi-head self-attention with optional cross-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-attention forward pass.
        
        Args:
            x: Input tensor [B, N, C]
        
        Returns:
            output: Attention output [B, N, C]
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            x: Query tensor [B, N, C]
            context: Key/value tensor [B, M, context_dim]
            context_mask: Optional attention mask [B, M]
        
        Returns:
            output: Cross-attention output [B, N, C]
        """
        B, N, C = x.shape
        M = context.shape[1]
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if context_mask is not None:
            mask = context_mask[:, None, None, :]  # [B, 1, 1, M]
            attn = attn.masked_fill(~mask, float("-inf"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaptive layer norm.
    
    Includes self-attention, cross-attention (for text conditioning),
    and feed-forward network with modulation.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        
        # Cross-attention for text conditioning
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(
                dim, context_dim, num_heads=num_heads, 
                attn_drop=dropout, proj_drop=dropout
            )
        
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, dropout=dropout)
        
        # AdaLN modulation
        num_modulations = 9 if self.has_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, num_modulations * dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, C]
            c: Conditioning tensor (timestep + optional class) [B, C]
            context: Text context for cross-attention [B, M, context_dim]
            context_mask: Attention mask for context [B, M]
        
        Returns:
            output: Block output [B, N, C]
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        
        if self.has_cross_attn:
            (shift_msa, scale_msa, gate_msa,
             shift_mca, scale_mca, gate_mca,
             shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(9, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=1)
        
        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # Cross-attention
        if self.has_cross_attn and context is not None:
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(
                modulate(self.norm2(x), shift_mca, scale_mca),
                context,
                context_mask,
            )
        
        # Feed-forward
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT with adaptive layer norm."""
    
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """
    Diffusion Transformer for PS-VAE latent space.
    
    Generates in the compact 96-channel latent space of PS-VAE,
    with text conditioning for T2I generation.
    
    Args:
        input_size: Spatial size of latent features
        in_channels: Number of latent channels (96 for PS-VAE)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        text_dim: Text embedding dimension
        num_text_tokens: Number of text tokens
        class_dropout_prob: Probability of dropping class conditioning
        learn_sigma: Whether to predict variance
    """
    
    def __init__(
        self,
        input_size: int = 16,
        in_channels: int = 96,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        text_dim: int = 4096,
        num_text_tokens: int = 77,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma
        
        # Patch embedding (treating each spatial location as a patch)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, input_size * input_size, hidden_size)
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedding(hidden_size)
        
        # Text conditioning
        self.text_embedder = TextConditionEmbedding(
            text_dim=text_dim,
            hidden_dim=hidden_size,
            num_tokens=num_text_tokens,
        )
        
        # Null text embedding for classifier-free guidance
        self.null_text_embed = nn.Parameter(torch.randn(1, num_text_tokens, hidden_size) * 0.02)
        self.class_dropout_prob = class_dropout_prob
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                context_dim=hidden_size,
            )
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy latent [B, H, W, C] or [B, C, H, W]
            t: Timesteps [B]
            text_embeds: Text embeddings [B, num_tokens, text_dim]
            text_mask: Text attention mask [B, num_tokens]
        
        Returns:
            output: Predicted noise (and variance if learn_sigma)
        """
        # Ensure [B, H, W, C] format
        if x.shape[1] == self.in_channels:  # [B, C, H, W]
            x = rearrange(x, "b c h w -> b h w c")
        
        B, H, W, C = x.shape
        
        # Flatten spatial dimensions
        x = rearrange(x, "b h w c -> b (h w) c")
        
        # Embed patches
        x = self.x_embedder(x) + self.pos_embed
        
        # Timestep conditioning
        c = self.t_embedder(t)
        
        # Text conditioning
        if text_embeds is not None:
            context = self.text_embedder(text_embeds)
            
            # Apply dropout for classifier-free guidance during training
            if self.training and self.class_dropout_prob > 0:
                mask = torch.rand(B, device=x.device) < self.class_dropout_prob
                null_context = self.null_text_embed.expand(B, -1, -1)
                context = torch.where(mask[:, None, None], null_context, context)
        else:
            context = self.null_text_embed.expand(B, -1, -1)
            text_mask = None
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, context, text_mask)
        
        # Final layer
        x = self.final_layer(x, c)
        
        # Reshape to spatial
        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
        
        return x
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeds: torch.Tensor,
        cfg_scale: float = 7.5,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        
        Args:
            x: Noisy latent
            t: Timesteps
            text_embeds: Text embeddings
            cfg_scale: Guidance scale
            text_mask: Text attention mask
        
        Returns:
            output: Guided prediction
        """
        # Conditional prediction
        cond_output = self.forward(x, t, text_embeds, text_mask)
        
        # Unconditional prediction
        uncond_output = self.forward(x, t, None, None)
        
        # Classifier-free guidance
        output = uncond_output + cfg_scale * (cond_output - uncond_output)
        
        return output

