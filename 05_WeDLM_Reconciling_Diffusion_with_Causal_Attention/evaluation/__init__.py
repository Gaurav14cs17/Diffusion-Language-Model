"""
WeDLM Evaluation
================

Benchmark evaluation for WeDLM.

Usage:
    python scripts/evaluate.py --benchmark gsm8k
    python scripts/download_data.py --status
"""

from .benchmarks import get_benchmark, BENCHMARKS

__all__ = ["get_benchmark", "BENCHMARKS"]

