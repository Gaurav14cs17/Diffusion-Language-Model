"""
WeDLM Core Implementation
=========================
Simplified WeDLM for educational purposes.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    window_size: int = 16
    entropy_threshold: float = 0.4
    pos_penalty: float = 0.02
    temperature: float = 0.0

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(P) = -Σ p_i log(p_i)"""
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

def select_positions(entropy: torch.Tensor, indices: List[int], 
                     threshold: float, penalty: float) -> List[int]:
    """Select positions with adjusted entropy < threshold."""
    pos = torch.tensor(indices, device=entropy.device, dtype=torch.float)
    adjusted = entropy + penalty * (pos - pos[0])
    selected = (adjusted < threshold).nonzero(as_tuple=True)[0]
    if len(selected) == 0:
        selected = adjusted.argmin().unsqueeze(0)
    return selected.tolist()

@torch.no_grad()
def generate(model, tokenizer, prompt: str, config: Config, 
             max_tokens: int = 100, mask_token: str = "<|mask|>") -> Tuple[str, dict]:
    """WeDLM generation with streaming parallel decoding."""
    device = next(model.parameters()).device
    
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        model.resize_token_embeddings(len(tokenizer))
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    gen = tokenizer.encode(prompt, return_tensors="pt").to(device)[0].tolist()
    window = [mask_id] * config.window_size
    flags = [True] * config.window_size
    steps, tokens = 0, 0
    
    while tokens < max_tokens:
        steps += 1
        logits = model(torch.tensor([gen + window], device=device)).logits[0]
        
        mask_idx = [i for i, f in enumerate(flags) if f]
        if not mask_idx:
            break
        
        mask_logits = torch.stack([logits[len(gen) + i - 1] for i in mask_idx])
        entropy = compute_entropy(mask_logits)
        fill_idx = select_positions(entropy, mask_idx, config.entropy_threshold, config.pos_penalty)
        
        for k in fill_idx:
            pos = mask_idx[k]
            if config.temperature > 0:
                probs = F.softmax(mask_logits[k] / config.temperature, dim=-1)
                window[pos] = torch.multinomial(probs, 1).item()
            else:
                window[pos] = mask_logits[k].argmax().item()
            flags[pos] = False
        
        commit = next((i for i, f in enumerate(flags) if f), len(window))
        if commit == 0:
            commit = 1
        
        gen.extend(window[:commit])
        tokens += commit
        
        if tokenizer.eos_token_id in window[:commit]:
            break
        
        window = window[commit:] + [mask_id] * commit
        flags = flags[commit:] + [True] * commit
    
    return tokenizer.decode(gen, skip_special_tokens=True), {
        "steps": steps, "tokens": tokens, "tokens_per_step": tokens/steps if steps > 0 else 0
    }
