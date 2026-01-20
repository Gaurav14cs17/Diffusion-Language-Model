"""
WeDLM Evaluation Benchmarks
===========================

Each benchmark module contains both dataset loading and evaluation logic.

Usage:
    from evaluation.benchmarks import GSM8K, MMLU, HumanEval
    
    # Load and evaluate
    benchmark = GSM8K()
    samples = benchmark.load()
    results = benchmark.evaluate(predictions)
"""

from .base import BaseBenchmark
from .gsm8k import GSM8K
from .mmlu import MMLU
from .humaneval import HumanEval
from .arc import ARC_Challenge, ARC_Easy

BENCHMARKS = {
    "gsm8k": GSM8K,
    "mmlu": MMLU,
    "humaneval": HumanEval,
    "arc_c": ARC_Challenge,
    "arc_e": ARC_Easy,
}


def get_benchmark(name: str) -> BaseBenchmark:
    """Get benchmark by name."""
    name = name.lower()
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]()

