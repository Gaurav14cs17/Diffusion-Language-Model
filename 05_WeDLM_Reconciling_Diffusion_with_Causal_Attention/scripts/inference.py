#!/usr/bin/env python
"""
WeDLM Inference Script
======================

Run WeDLM inference with streaming parallel decoding.

Usage:
    python scripts/inference.py --model ./wedlm_model --prompt "Hello world"
    python scripts/inference.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Solve: 5 + 3 ="
"""

import argparse
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_entropy(logits):
    """Compute entropy: H(P) = -Σ p_i log(p_i)"""
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


@torch.no_grad()
def wedlm_generate(
    model, tokenizer, prompt,
    max_new_tokens=100,
    window_size=16,
    entropy_threshold=0.4,
    pos_penalty=0.02,
    temperature=0.0,
    mask_token="<|mask|>",
    verbose=False
):
    """WeDLM Streaming Parallel Decoding."""
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure mask token exists
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        model.resize_token_embeddings(len(tokenizer))
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    eos_id = tokenizer.eos_token_id
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0].tolist()
    generated = prompt_ids.copy()
    
    # Initialize window
    window = [mask_id] * window_size
    window_flags = [True] * window_size
    
    tokens_generated = 0
    steps = 0
    
    while tokens_generated < max_new_tokens:
        steps += 1
        
        # Forward pass
        input_ids = torch.tensor([generated + window], device=device)
        logits = model(input_ids).logits[0]
        
        prefix_len = len(generated)
        mask_indices = [i for i, f in enumerate(window_flags) if f]
        
        if not mask_indices:
            break
        
        # Get logits for mask positions
        mask_logits = torch.stack([logits[prefix_len + i - 1] for i in mask_indices])
        
        # Compute adjusted entropy
        entropy = compute_entropy(mask_logits)
        pos = torch.tensor(mask_indices, device=device, dtype=torch.float)
        adjusted = entropy + pos_penalty * (pos - pos[0])
        
        # Select positions to fill
        fill_indices = (adjusted < entropy_threshold).nonzero(as_tuple=True)[0]
        if len(fill_indices) == 0:
            fill_indices = adjusted.argmin().unsqueeze(0)
        
        # Sample tokens
        for idx in fill_indices.tolist():
            pos = mask_indices[idx]
            if temperature > 0:
                probs = F.softmax(mask_logits[idx] / temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
            else:
                token = mask_logits[idx].argmax().item()
            window[pos] = token
            window_flags[pos] = False
        
        # Commit prefix
        commit = 0
        for i in range(len(window)):
            if not window_flags[i]:
                commit += 1
            else:
                break
        if commit == 0:
            commit = 1
        
        committed = window[:commit]
        generated.extend(committed)
        tokens_generated += commit
        
        if verbose:
            print(f"Step {steps}: filled {len(fill_indices)}, committed {commit}")
        
        if eos_id in committed:
            break
        
        # Slide window
        window = window[commit:] + [mask_id] * commit
        window_flags = window_flags[commit:] + [True] * commit
    
    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    stats = {
        "steps": steps,
        "tokens": tokens_generated,
        "tokens_per_step": tokens_generated / steps if steps > 0 else 0
    }
    
    return output_text, stats


def main():
    parser = argparse.ArgumentParser(description="WeDLM Inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Max tokens to generate")
    parser.add_argument("--window_size", type=int, default=16,
                        help="WeDLM window size")
    parser.add_argument("--entropy_threshold", type=float, default=0.4,
                        help="Entropy threshold (τ)")
    parser.add_argument("--pos_penalty", type=float, default=0.02,
                        help="Position penalty (λ)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step progress")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    print("Model loaded!")
    
    # Generate
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    
    start = time.time()
    output, stats = wedlm_generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        window_size=args.window_size,
        entropy_threshold=args.entropy_threshold,
        pos_penalty=args.pos_penalty,
        temperature=args.temperature,
        verbose=args.verbose
    )
    elapsed = time.time() - start
    
    print(f"\nOutput: {output}")
    print("\n" + "-" * 60)
    print(f"Stats: {stats}")
    print(f"Time: {elapsed:.2f}s ({stats['tokens']/elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()

