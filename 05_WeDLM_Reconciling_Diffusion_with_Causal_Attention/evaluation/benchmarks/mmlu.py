"""
MMLU Benchmark
==============

Massive Multitask Language Understanding - General knowledge evaluation.
"""

import json
import re
from typing import List, Dict, Any, Optional

from .base import BaseBenchmark


class MMLU(BaseBenchmark):
    """MMLU general knowledge benchmark."""
    
    name = "mmlu"
    data_file = "mmlu.json"
    
    def load(self) -> List[Dict[str, Any]]:
        """Load MMLU samples."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                f"Run: python scripts/download_data.py --dataset mmlu"
            )
        
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        samples = []
        for idx, item in enumerate(data):
            answer_idx = item["answer"]
            if isinstance(answer_idx, int):
                ground_truth = chr(65 + answer_idx)  # 0->A, 1->B, etc.
            else:
                ground_truth = str(answer_idx)
            
            samples.append({
                "id": f"mmlu_{idx}",
                "question": item["question"],
                "choices": item["choices"],
                "answer": answer_idx,
                "subject": item.get("subject", ""),
                "ground_truth": ground_truth
            })
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format MMLU prompt with multiple choice."""
        choices = "\n".join([
            f"{chr(65+i)}. {choice}" 
            for i, choice in enumerate(sample["choices"])
        ])
        return f"Question: {sample['question']}\n\n{choices}\n\nAnswer:"
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract letter answer from response."""
        # Look for A, B, C, or D
        match = re.search(r'\b([ABCD])\b', text.upper())
        if match:
            return match.group(1)
        return None
    
    def check_correct(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        if not prediction or not ground_truth:
            return False
        return prediction.upper().strip() == ground_truth.upper().strip()
    
    def download(self):
        """Download MMLU dataset."""
        from datasets import load_dataset
        
        print("Downloading MMLU...")
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        data = []
        for item in dataset:
            data.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "subject": item.get("subject", "")
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

