#!/usr/bin/env python3
"""
Integration tests for the full training pipeline.

These tests verify that the training scripts work end-to-end
with a tiny open-source dataset (CIFAR-10).

Usage:
    python tests/test_training.py
"""

# Re-export the validation test for backwards compatibility
from tests.test_training_validation import main

if __name__ == "__main__":
    main()

