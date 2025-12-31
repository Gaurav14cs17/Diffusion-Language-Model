# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from wedlm.utils.context import get_context

# Optional imports with fallbacks
TRITON_AVAILABLE = False
FLASH_ATTN_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None


# Triton kernel for KV cache storage (only available when triton is installed)
if TRITON_AVAILABLE:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_fallback(key, value, k_cache, v_cache, slot_mapping):
    """Pure PyTorch fallback for KV cache storage."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    key_flat = key.view(N, D)
    value_flat = value.view(N, D)
    
    for idx in range(N):
        slot = slot_mapping[idx].item()
        if slot == -1:
            continue
        k_cache.view(-1, D)[slot] = key_flat[idx]
        v_cache.view(-1, D)[slot] = value_flat[idx]


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Store key-value pairs in KV cache using triton kernel or fallback."""
    if TRITON_AVAILABLE and key.is_cuda:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )
    else:
        store_kvcache_fallback(key, value, k_cache, v_cache, slot_mapping)


def _fallback_attention(q, k, v, scale, causal=True):
    """Pure PyTorch attention implementation as fallback.
    
    Args:
        q: Query tensor [seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_kv_heads, head_dim]
        v: Value tensor [seq_len, num_kv_heads, head_dim]
        scale: Softmax scale factor
        causal: Whether to apply causal masking
        
    Returns:
        Output tensor [seq_len, num_heads, head_dim]
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = k.shape[0]
    num_kv_heads = k.shape[1]
    
    # Handle GQA (grouped query attention) by repeating KV heads
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    
    # Transpose for batch matrix multiplication: [num_heads, seq_len, head_dim]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    
    # Compute attention scores: [num_heads, seq_len_q, seq_len_k]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if causal and seq_len_q > 1:
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=seq_len_k - seq_len_q + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
    
    # Softmax and apply to values
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    # Transpose back: [seq_len, num_heads, head_dim]
    return output.transpose(0, 1)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        wedlm_window_size: Optional[int] = None,
        max_context_len: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.wedlm_window_size = wedlm_window_size
        self.max_context_len = max_context_len if max_context_len is not None else 4096

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        device = q.device

        # 1) Store KV Cache
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2) Prefill
        if context.is_prefill:
            k_src, v_src = (
                (k_cache, v_cache) if context.block_tables is not None else (k, v)
            )
            
            # Use flash attention if available and on CUDA
            if FLASH_ATTN_AVAILABLE and q.is_cuda:
                return flash_attn_varlen_func(
                    q,
                    k_src,
                    v_src,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                # Fallback to standard attention
                return _fallback_attention(q, k_src, v_src, self.scale, causal=True)

        if context.per_seq_wedlm_sizes is None:
            raise RuntimeError(
                "context.per_seq_wedlm_sizes is None inside Attention.forward (Decode mode)."
            )

        per_seq_wedlm_sizes = context.per_seq_wedlm_sizes

        cu_seqlens_q = F.pad(per_seq_wedlm_sizes.cumsum(0), (1, 0)).to(dtype=torch.int32)

        if context.max_seqlen_q > 0:
            max_seqlen_q = context.max_seqlen_q
        else:
            max_seqlen_q = torch.max(per_seq_wedlm_sizes).item()

        prefix_lens = context.context_lens
        k_lens = (prefix_lens + per_seq_wedlm_sizes).to(torch.int32)
        cu_seqlens_k = F.pad(k_lens.cumsum(dim=0), (1, 0)).to(torch.int32)

        # Use flash attention if available and on CUDA
        if FLASH_ATTN_AVAILABLE and q.is_cuda:
            return flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=self.max_context_len,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            # Fallback to standard attention
            # For decode phase with KV cache, we need to handle this differently
            return _fallback_attention(q, k_cache, v_cache, self.scale, causal=True)
