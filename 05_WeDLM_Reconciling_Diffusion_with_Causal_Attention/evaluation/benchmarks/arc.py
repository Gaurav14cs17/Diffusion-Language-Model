"""
ARC Benchmark
=============

AI2 Reasoning Challenge - Science reasoning evaluation.
"""

import json
import re
from typing import List, Dict, Any, Optional

from .base import BaseBenchmark


class ARC_Challenge(BaseBenchmark):
    """ARC-Challenge science reasoning benchmark."""
    
    name = "arc_c"
    data_file = "arc-c.json"
    
    def load(self) -> List[Dict[str, Any]]:
        """Load ARC-Challenge samples."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                f"Run: python scripts/download_data.py --dataset arc"
            )
        
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        samples = []
        for idx, item in enumerate(data):
            samples.append({
                "id": f"arc_c_{idx}",
                "question": item["question"],
                "choices": item["choices"],
                "ground_truth": item["answer"]
            })
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format ARC prompt with multiple choice."""
        choices = sample["choices"]
        if isinstance(choices, dict):
            # Handle {"label": [...], "text": [...]} format
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choices_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
        else:
            # Handle list format
            choices_str = "\n".join([
                f"{chr(65+i)}. {choice}" 
                for i, choice in enumerate(choices)
            ])
        
        return f"Question: {sample['question']}\n\n{choices_str}\n\nAnswer:"
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract letter answer from response."""
        match = re.search(r'\b([ABCDE])\b', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_correct(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        if not prediction or not ground_truth:
            return False
        return prediction.upper().strip() == ground_truth.upper().strip()
    
    def download(self):
        """Download ARC-Challenge dataset."""
        from datasets import load_dataset
        
        print("Downloading ARC-Challenge...")
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        
        data = []
        for item in dataset:
            data.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answerKey"]
            })
        
        self.data_dir.mkdir(exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved {len(data)} samples to {self.data_path}")
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 10,
            "temperature": 0.0,
        }


class ARC_Easy(ARC_Challenge):
    """ARC-Easy science reasoning benchmark."""
    
    name = "arc_e"
    data_file = "arc-e.json"
    
    def download(self):
        """Download ARC-Easy dataset."""
        from datasets import load_dataset
        
        print("Downloading ARC-Easy...")
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        
        data = []
        for item in dataset:
            data.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answerKey"]
            })
        
        self.data_dir.mkdir(exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved {len(data)} samples to {self.data_path}")

