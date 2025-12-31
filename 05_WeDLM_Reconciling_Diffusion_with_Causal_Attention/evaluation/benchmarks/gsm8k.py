"""
GSM8K Benchmark
===============

Grade School Math 8K - Math reasoning evaluation.
"""

import json
import re
from typing import List, Dict, Any, Optional

from .base import BaseBenchmark


class GSM8K(BaseBenchmark):
    """GSM8K math reasoning benchmark."""
    
    name = "gsm8k"
    data_file = "gsm8k.jsonl"
    
    def load(self) -> List[Dict[str, Any]]:
        """Load GSM8K samples."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                f"Run: python scripts/download_data.py --dataset gsm8k"
            )
        
        samples = []
        with open(self.data_path, "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                samples.append({
                    "id": f"gsm8k_{idx}",
                    "question": data["question"],
                    "answer": data["answer"],
                    "ground_truth": self.extract_answer(data["answer"])
                })
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format GSM8K prompt."""
        return f"Problem: {sample['question']}\n\nSolve step by step:\n"
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from GSM8K response."""
        # Look for #### pattern (GSM8K format)
        match = re.search(r'####\s*(\-?\d[\d,]*\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Look for boxed answer (LaTeX format)
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fall back to last number
        numbers = re.findall(r'\-?\d[\d,]*\.?\d*', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return None
    
    def check_correct(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        if not prediction or not ground_truth:
            return False
        
        try:
            pred_num = float(prediction.replace(',', ''))
            true_num = float(ground_truth.replace(',', ''))
            return abs(pred_num - true_num) < 1e-6
        except ValueError:
            return prediction.strip() == ground_truth.strip()
    
    def download(self):
        """Download GSM8K dataset."""
        from datasets import load_dataset
        
        print("Downloading GSM8K...")
        dataset = load_dataset("openai/gsm8k", "main")
        
        self.data_dir.mkdir(exist_ok=True)
        with open(self.data_path, "w") as f:
            for item in dataset["test"]:
                f.write(json.dumps({
                    "question": item["question"],
                    "answer": item["answer"]
                }) + "\n")
        
        print(f"✅ Saved {len(dataset['test'])} samples to {self.data_path}")
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 512,
            "temperature": 0.0,
        }

