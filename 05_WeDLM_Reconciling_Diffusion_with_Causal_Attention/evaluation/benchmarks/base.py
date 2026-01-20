"""
Base Benchmark Class
====================

All benchmarks inherit from this base class.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    name: str = "base"
    data_file: str = ""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_file
    
    def is_downloaded(self) -> bool:
        """Check if dataset is downloaded."""
        return self.data_path.exists()
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load dataset samples."""
        pass
    
    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format a sample into a prompt for the model."""
        pass
    
    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from model output."""
        pass
    
    @abstractmethod
    def check_correct(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        pass
    
    def evaluate(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions.
        
        Args:
            predictions: List of dicts with 'prediction' and 'ground_truth' keys
            
        Returns:
            Dict with evaluation metrics
        """
        correct = 0
        total = len(predictions)
        
        for pred in predictions:
            if self.check_correct(pred.get("prediction", ""), pred.get("ground_truth", "")):
                correct += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total
        }
    
    def download(self):
        """Download the dataset. Override in subclass."""
        raise NotImplementedError(f"Download not implemented for {self.name}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get recommended generation config for this benchmark."""
        return {
            "max_new_tokens": 256,
            "temperature": 0.0,
        }

