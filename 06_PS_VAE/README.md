# 🎨 PS-VAE: Pixel-Semantic Variational Autoencoder

> **"Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing"**

<p align="center">
  <a href="https://arxiv.org/abs/2512.17909"><img src="https://img.shields.io/badge/arXiv-2512.17909-b31b1b.svg" alt="arXiv"></a>
  <a href="https://jshilong.github.io/PS-VAE-PAGE/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://github.com/Gaurav14cs17"><img src="https://img.shields.io/badge/GitHub-Gaurav14cs17-181717.svg?logo=github" alt="GitHub"></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick-Start-green.svg" alt="Quick Start"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

## 📂 Repository Map

```
📦 PS-VAE
|
+-- 🎓 docs/
|   +-- 00_Theory/                → Mathematical foundations
|   +-- 01_Paper_Breakdown/       → Detailed paper analysis
|   +-- architecture.md           → Complete architecture docs
|   +-- images/                   → Architecture diagrams (SVG)
|
+-- 🧠 psvae/
|   +-- models/                   → S-VAE, PS-VAE implementations
|   +-- diffusion/                → DiT, schedulers, samplers
|   +-- training/                 → Trainers, datasets
|   +-- utils/                    → Metrics, visualization
|
+-- 📜 scripts/                   → Training & generation scripts
+-- ⚙️ configs/                   → YAML configurations
+-- 📝 examples/                  → Getting started examples
+-- 🧪 tests/                     → Unit tests
```

---

## 🌟 The Big Picture

<p align="center">
  <img src="docs/images/readme_big_picture.svg" alt="PS-VAE Big Picture" width="100%">
</p>

---

## 🧮 The Core Mathematics

### The Two Fundamental Problems

<p align="center">
  <img src="docs/images/readme_off_manifold.svg" alt="Off-Manifold Problem" width="100%">
</p>

<p align="center">
  <img src="docs/images/readme_weak_reconstruction.svg" alt="Weak Reconstruction Problem" width="100%">
</p>

### PS-VAE Loss Function

<p align="center">
  <img src="docs/images/readme_loss_function.svg" alt="PS-VAE Loss Function" width="100%">
</p>

---

## 🏗️ Architecture Overview

### Three-Stage Training Pipeline

<p align="center">
  <img src="docs/images/readme_training_pipeline.svg" alt="Training Pipeline" width="100%">
</p>

### Latent Space Comparison

<p align="center">
  <img src="docs/images/readme_latent_comparison.svg" alt="Latent Space Comparison" width="100%">
</p>

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Gaurav14cs17/PS-VAE.git
cd PS-VAE

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from psvae import PSVAE
import torch

# Create model
model = PSVAE(
    encoder_type="dinov2",
    model_size="large",
    latent_dim=96,
    image_size=256,
)

# Encode image to latent
image = torch.randn(1, 3, 256, 256)
latent = model.encode_to_latent(image)  # [1, 16, 16, 96]

# Decode latent to image
reconstructed = model.decode_to_image(latent)  # [1, 3, 256, 256]
```

### Training

```bash
# Stage 1: Train S-VAE
python scripts/train_psvae.py --config configs/default.yaml --stage svae

# Stage 2: Train PS-VAE
python scripts/train_psvae.py --config configs/default.yaml --stage psvae

# Stage 3: Train DiT
python scripts/train_dit.py --config configs/default.yaml
```

### Generation

```bash
# Generate images from text
python scripts/generate.py \
    --prompt "A beautiful sunset over mountains" \
    --num_samples 4 \
    --cfg_scale 7.5
```

---

## 📊 Key Contributions

| Innovation | Description |
|------------|-------------|
| 🗜️ **Compact Latent** | 96 channels (vs 1024+ in RAE) with 16×16 spatial |
| 🎯 **KL Regularization** | Prevents off-manifold generation |
| 🔀 **Dual Reconstruction** | Semantic + Pixel objectives |
| 🖼️ **Unified Framework** | Single model for T2I generation AND editing |

---

## 💡 Key Insights

<p align="center">
  <img src="docs/images/readme_key_insights.svg" alt="Key Insights" width="100%">
</p>

---

## 📐 Detailed Architecture

For comprehensive mathematical details, see [docs/architecture.md](docs/architecture.md).

### S-VAE Architecture

<p align="center">
  <img src="docs/images/readme_svae_architecture.svg" alt="S-VAE Architecture" width="100%">
</p>

### PS-VAE Architecture

<p align="center">
  <img src="docs/images/readme_psvae_architecture.svg" alt="PS-VAE Architecture" width="100%">
</p>

### DiT Architecture

<p align="center">
  <img src="docs/images/readme_dit_architecture.svg" alt="DiT Architecture" width="100%">
</p>

---

## 🔗 Resources & Links

| Resource | Link |
|----------|------|
| 📄 Paper | [arXiv:2512.17909](https://arxiv.org/abs/2512.17909) |
| 🌐 Project Page | [jshilong.github.io/PS-VAE-PAGE](https://jshilong.github.io/PS-VAE-PAGE/) |
| 📚 Architecture Docs | [docs/architecture.md](docs/architecture.md) |
| 🎓 Theory Guide | [docs/00_Theory](docs/00_Theory/README.md) |
| 📄 Paper Breakdown | [docs/01_Paper_Breakdown](docs/01_Paper_Breakdown/README.md) |
| 💻 Examples | [examples/](examples/) |

---

## 📜 Citation

```bibtex
@article{zhang2024psvae,
  title={Both Semantics and Reconstruction Matter: Making Representation 
         Encoders Ready for Text-to-Image Generation and Editing},
  author={Zhang, Shilong and Zhang, He and Zhang, Zhifei and 
          Ge, Chongjian and others},
  journal={arXiv preprint arXiv:2512.17909},
  year={2024}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Made with 💜 by <a href="https://github.com/Gaurav14cs17">Gaurav14cs17</a></b>
  <br>
  <i>"The best latent space is one that preserves both meaning and detail."</i>
</p>
