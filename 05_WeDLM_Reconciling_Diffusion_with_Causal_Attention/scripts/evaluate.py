#!/usr/bin/env python
"""
WeDLM Evaluation Script
=======================

Evaluate WeDLM on standard benchmarks.

Usage:
    python scripts/evaluate.py --model ./wedlm_model --benchmark gsm8k
    python scripts/evaluate.py --model ./wedlm_model --benchmark mmlu --max_samples 100
    python scripts/evaluate.py --list  # List available benchmarks
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import get_benchmark, BENCHMARKS


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


@torch.no_grad()
def wedlm_generate(model, tokenizer, prompt, mask_id, device,
                   max_tokens=256, window_size=16, 
                   entropy_threshold=0.4, pos_penalty=0.02):
    """WeDLM generation."""
    model.eval()
    
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0].tolist()
    generated = prompt_ids.copy()
    window = [mask_id] * window_size
    window_flags = [True] * window_size
    tokens = 0
    
    while tokens < max_tokens:
        input_ids = torch.tensor([generated + window], device=device)
        logits = model(input_ids).logits[0]
        prefix_len = len(generated)
        
        mask_idx = [i for i, f in enumerate(window_flags) if f]
        if not mask_idx:
            break
        
        mask_logits = torch.stack([logits[prefix_len + i - 1] for i in mask_idx])
        entropy = compute_entropy(mask_logits)
        pos = torch.tensor(mask_idx, device=device, dtype=torch.float)
        adjusted = entropy + pos_penalty * (pos - pos[0])
        
        fill_idx = (adjusted < entropy_threshold).nonzero(as_tuple=True)[0]
        if len(fill_idx) == 0:
            fill_idx = adjusted.argmin().unsqueeze(0)
        
        for k in fill_idx.tolist():
            p = mask_idx[k]
            window[p] = mask_logits[k].argmax().item()
            window_flags[p] = False
        
        commit = next((i for i, f in enumerate(window_flags) if f), len(window))
        if commit == 0:
            commit = 1
        
        generated.extend(window[:commit])
        tokens += commit
        
        if tokenizer.eos_token_id in window[:commit]:
            break
        
        window = window[commit:] + [mask_id] * commit
        window_flags = window_flags[commit:] + [True] * commit
    
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_evaluation(model, tokenizer, mask_id, device, benchmark, max_samples=-1, 
                   window_size=16, entropy_threshold=0.4):
    """Run evaluation on a benchmark."""
    
    # Load samples
    samples = benchmark.load()
    if max_samples > 0:
        samples = samples[:max_samples]
    
    # Get config
    config = benchmark.get_config()
    max_tokens = config.get("max_new_tokens", 256)
    
    # Run inference
    predictions = []
    for sample in tqdm(samples, desc=benchmark.name.upper()):
        prompt = benchmark.format_prompt(sample)
        output = wedlm_generate(
            model, tokenizer, prompt, mask_id, device,
            max_tokens=max_tokens, window_size=window_size,
            entropy_threshold=entropy_threshold
        )
        
        pred = benchmark.extract_answer(output)
        predictions.append({
            "id": sample.get("id", ""),
            "prediction": pred,
            "ground_truth": sample.get("ground_truth", ""),
            "output": output
        })
    
    # Evaluate
    results = benchmark.evaluate(predictions)
    results["predictions"] = predictions
    
    return results


def main():
    parser = argparse.ArgumentParser(description="WeDLM Evaluation")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--benchmark", "-b", type=str, 
                        choices=list(BENCHMARKS.keys()) + ["all"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--window_size", type=int, default=16,
                        help="WeDLM window size")
    parser.add_argument("--entropy_threshold", type=float, default=0.4,
                        help="Entropy threshold")
    parser.add_argument("--output", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available benchmarks")
    args = parser.parse_args()
    
    # List benchmarks
    if args.list:
        print("\n📊 Available Benchmarks:")
        print("-" * 40)
        for name, cls in BENCHMARKS.items():
            print(f"  • {name}: {cls.name}")
        print("-" * 40)
        return
    
    # Validate arguments
    if not args.model or not args.benchmark:
        parser.print_help()
        print("\n💡 Use --list to see available benchmarks")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Add mask token
    MASK_TOKEN = "<|mask|>"
    if MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    print("Model loaded!")
    
    # Determine benchmarks to run
    if args.benchmark == "all":
        benchmark_names = list(BENCHMARKS.keys())
    else:
        benchmark_names = [args.benchmark]
    
    # Run evaluations
    all_results = {}
    for name in benchmark_names:
        print(f"\n{'='*60}")
        print(f"Evaluating on {name.upper()}")
        print('='*60)
        
        try:
            benchmark = get_benchmark(name)
            results = run_evaluation(
                model, tokenizer, mask_id, device, benchmark,
                max_samples=args.max_samples,
                window_size=args.window_size,
                entropy_threshold=args.entropy_threshold
            )
            
            all_results[name] = {
                "accuracy": results["accuracy"],
                "correct": results["correct"],
                "total": results["total"]
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"  Accuracy: {results['accuracy']*100:.2f}%")
            print(f"  Correct: {results['correct']}/{results['total']}")
            
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
