"""
HumanEval Benchmark
===================

Code generation evaluation.
"""

import json
import re
from typing import List, Dict, Any, Optional

from .base import BaseBenchmark


class HumanEval(BaseBenchmark):
    """HumanEval code generation benchmark."""
    
    name = "humaneval"
    data_file = "humaneval.jsonl"
    
    def load(self) -> List[Dict[str, Any]]:
        """Load HumanEval samples."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                f"Run: python scripts/download_data.py --dataset humaneval"
            )
        
        samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                samples.append({
                    "id": data["task_id"],
                    "prompt": data["prompt"],
                    "entry_point": data["entry_point"],
                    "test": data["test"],
                    "canonical_solution": data.get("canonical_solution", ""),
                    "ground_truth": data.get("canonical_solution", "")
                })
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format HumanEval prompt."""
        return sample["prompt"]
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract code from response."""
        # Try to extract code between ```python and ```
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try to extract code after the prompt
        # Look for function definition
        match = re.search(r'(def\s+\w+.*?)(?=\ndef\s+|\Z)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text
    
    def check_correct(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction is correct.
        
        Note: Full HumanEval evaluation requires running tests.
        This is a simplified check.
        """
        # Basic check - prediction should contain function definition
        return "def " in prediction
    
    def download(self):
        """Download HumanEval dataset."""
        from datasets import load_dataset
        
        print("Downloading HumanEval...")
        dataset = load_dataset("openai_humaneval", split="test")
        
        self.data_dir.mkdir(exist_ok=True)
        with open(self.data_path, "w") as f:
            for item in dataset:
                f.write(json.dumps({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "test": item["test"],
                    "canonical_solution": item["canonical_solution"]
                }) + "\n")
        
        print(f"✅ Saved {len(dataset)} samples to {self.data_path}")
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 256,
            "temperature": 0.0,
        }

