#!/usr/bin/env python3
"""Setup script for PS-VAE package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="psvae",
    version="0.1.0",
    author="Gaurav14cs17",
    author_email="",
    description="Pixel-Semantic VAE for Text-to-Image Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gaurav14cs17/PS-VAE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "psvae-train=scripts.train_psvae:main",
            "psvae-train-dit=scripts.train_dit:main",
            "psvae-generate=scripts.generate:main",
        ],
    },
)

