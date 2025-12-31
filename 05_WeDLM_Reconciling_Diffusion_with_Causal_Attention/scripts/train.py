#!/usr/bin/env python
"""
WeDLM Training Script
=====================

Fine-tune a pretrained AR model into WeDLM using Causal Masked Language Modeling.

Usage:
    python scripts/train.py --model Qwen/Qwen2.5-0.5B --output ./wedlm_model
    python scripts/train.py --model Qwen/Qwen2.5-0.5B --epochs 3 --batch_size 8
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm


class CausalMLMDataset(Dataset):
    """Dataset for Causal Masked Language Modeling training."""
    
    def __init__(self, tokenizer, texts, max_length, mask_token_id, 
                 mask_ratio_min=0.1, mask_ratio_max=0.5):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        mask_ratio = np.random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        masked_ids, labels, mask_flags = self._random_span_masking(input_ids, mask_ratio)
        
        return {
            "input_ids": masked_ids, 
            "attention_mask": attention_mask,
            "labels": labels, 
            "mask_flags": mask_flags
        }
    
    def _random_span_masking(self, input_ids, mask_ratio, span_mean=3):
        """Apply random span masking to input sequence."""
        seq_len = len(input_ids)
        num_to_mask = int(seq_len * mask_ratio)
        
        masked_ids = input_ids.clone()
        mask_flags = torch.zeros(seq_len, dtype=torch.bool)
        
        positions_masked = 0
        attempts = 0
        
        while positions_masked < num_to_mask and attempts < seq_len * 10:
            attempts += 1
            span_len = min(np.random.geometric(p=1/span_mean), seq_len - 1)
            start = np.random.randint(0, max(1, seq_len - span_len + 1))
            
            for i in range(start, min(start + span_len, seq_len)):
                if not mask_flags[i]:
                    masked_ids[i] = self.mask_token_id
                    mask_flags[i] = True
                    positions_masked += 1
                    if positions_masked >= num_to_mask:
                        break
        
        return masked_ids, input_ids, mask_flags


def compute_cmlm_loss(model, batch, device):
    """Compute loss only on masked positions with causal attention."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    mask_flags = batch["mask_flags"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    shift_mask = mask_flags[:, 1:].reshape(-1)
    
    # Loss only on masked positions
    if shift_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    return F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask])


def main():
    parser = argparse.ArgumentParser(description="WeDLM Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Base model name or path")
    parser.add_argument("--output", type=str, default="./wedlm_model",
                        help="Output directory for trained model")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Training dataset name")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max training samples (-1 for all)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Add mask token
    MASK_TOKEN = "<|mask|>"
    if MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    print(f"Model loaded. Mask token ID: {mask_token_id}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
    else:
        dataset = load_dataset(args.dataset, split="train")
        texts = dataset["text"]
    
    if args.max_samples > 0:
        texts = texts[:args.max_samples]
    
    print(f"Using {len(texts)} training samples")
    
    train_dataset = CausalMLMDataset(tokenizer, texts, args.max_length, mask_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss = compute_cmlm_loss(model, batch, device)
                loss = loss / args.gradient_accumulation
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation
            progress_bar.set_postfix({"loss": f"{total_loss/(step+1):.4f}"})
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    print(f"\nSaving model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✅ Training complete!")


if __name__ == "__main__":
    main()

