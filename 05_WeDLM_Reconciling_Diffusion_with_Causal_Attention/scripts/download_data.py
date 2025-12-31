#!/usr/bin/env python
"""
WeDLM Dataset Downloader
========================

Download evaluation datasets.

Usage:
    python scripts/download_data.py --dataset gsm8k
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --status  # Check download status
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import BENCHMARKS


def check_status():
    """Check which datasets are downloaded."""
    print("\n📊 Dataset Status:")
    print("-" * 50)
    
    all_ready = True
    for name, cls in BENCHMARKS.items():
        benchmark = cls()
        status = "✅ Ready" if benchmark.is_downloaded() else "❌ Missing"
        print(f"  {name:15} {status}")
        if not benchmark.is_downloaded():
            all_ready = False
    
    print("-" * 50)
    return all_ready


def download_dataset(name):
    """Download a specific dataset."""
    if name not in BENCHMARKS:
        print(f"❌ Unknown dataset: {name}")
        print(f"   Available: {', '.join(BENCHMARKS.keys())}")
        return False
    
    benchmark = BENCHMARKS[name]()
    
    if benchmark.is_downloaded():
        print(f"✅ {name} already downloaded")
        return True
    
    try:
        benchmark.download()
        return True
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="WeDLM Dataset Downloader")
    parser.add_argument("--dataset", "-d", type=str, 
                        choices=list(BENCHMARKS.keys()) + ["all"],
                        help="Dataset to download")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Check download status")
    args = parser.parse_args()
    
    if args.status:
        check_status()
        return
    
    if not args.dataset:
        parser.print_help()
        print("\n📊 Available datasets:")
        for name in BENCHMARKS.keys():
            print(f"  • {name}")
        print("\n💡 Use --status to check which are downloaded")
        return
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download
    if args.dataset == "all":
        print("\n📥 Downloading all datasets...")
        for name in BENCHMARKS.keys():
            print(f"\n[{name}]")
            download_dataset(name)
    else:
        download_dataset(args.dataset)
    
    print("\n")
    check_status()


if __name__ == "__main__":
    main()
