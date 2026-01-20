<div align="center">

# 🌀 Diffusion Language Models

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=9D4EDD&center=true&vCenter=true&multiline=true&repeat=false&width=800&height=100&lines=From+Noise+to+Language;The+Future+of+Text+Generation" alt="Typing SVG" />

<br>

[![Diffusion](https://img.shields.io/badge/🔄_Diffusion-Language_Models-9D4EDD?style=for-the-badge&labelColor=1a1a2e)](.)
[![Research](https://img.shields.io/badge/📚_Research-Papers-16a085?style=for-the-badge&labelColor=1a1a2e)](.)
[![Math](https://img.shields.io/badge/🧮_Complete-Proofs-e74c3c?style=for-the-badge&labelColor=1a1a2e)](.)

<br>

*A deep dive into the revolutionary paradigm of iterative text generation*

---

### 🎬 The Diffusion Process

```
.--------------------------------------------------------------------------.
|                                                                          |
|   ⬇️  FORWARD (Corruption)                                               |
|   ====================================================================   |
|                                                                          |
|   "The cat sat on the mat"                                               |
|            ↓                                                             |
|   "The [M] sat on the mat"                                               |
|            ↓                                                             |
|   "[M] [M] sat [M] the [M]"                                              |
|            ↓                                                             |
|   "[M] [M] [M] [M] [M] [M]"  ← Pure Noise                                |
|                                                                          |
|   ⬆️  REVERSE (Generation)                                               |
|   ====================================================================   |
|                                                                          |
|   "[M] [M] [M] [M] [M] [M]"  ← Start with noise                          |
|            ↓                                                             |
|   "[M] cat [M] [M] [M] mat"                                              |
|            ↓                                                             |
|   "The cat sat [M] the mat"                                              |
|            ↓                                                             |
|   "The cat sat on the mat"  ← Clean text!                                |
|                                                                          |
'--------------------------------------------------------------------------'

```

</div>

---

## 🌟 Why Diffusion for Language?

<table>
<tr>
<td width="50%">

### ❌ Autoregressive Limitations

```
Token by token, left to right...
Never looking back...

[The] → [The][cat] → [The][cat][sat]

• Can't revise past decisions
• Only sees left context  
• Sequential = slow for long text

```

</td>
<td width="50%">

### ✅ Diffusion Advantages

```
All at once, refine iteratively...
Full context, full power...

[M][M][M] → [The][M][sat] → [The][cat][sat]

• Revise any position anytime
• Bidirectional context
• Parallel = fast generation

```

</td>
</tr>
</table>

---

## 📚 The Learning Journey

<div align="center">

```
                    +-------------------------------------+
                    |         START YOUR JOURNEY          |
                    |              HERE ↓                 |
                    +-------------------------------------+
                                     |
                    +================╧================+
                    |   📐 00_Diffusion_Theory        |
                    |   -------------------------     |
                    |   Master the fundamentals:      |
                    |   • Data Space & Probability    |
                    |   • Forward & Reverse Process   |
                    |   • ELBO & Training Objective   |
                    |   • Sampling Algorithms         |
                    +================╤================+
                                     |
                    +================╧================+
                    |   🎭 01_MDLM                    |
                    |   -------------------------     |
                    |   Simple & Effective Masked     |
                    |   Diffusion Language Models     |
                    |   • Absorbing state diffusion   |
                    |   • Simplified ELBO             |
                    +================╤================+
                                     |
                    +================╧================+
                    |   🔀 02_Block_Diffusion         |
                    |   -------------------------     |
                    |   Best of Both Worlds:          |
                    |   AR + Diffusion Hybrid         |
                    |   • Block-causal attention      |
                    |   • KV cache compatibility      |
                    +================╤================+
                                     |
                    +================╧================+
                    |   🚀 03_Dream_7B                |
                    |   -------------------------     |
                    |   Scaling Up Diffusion LLMs     |
                    |   • Large-scale training        |
                    |   • Competitive with AR         |
                    +================╤================+
                                     |
                    +================╧================+
                    |   ⚡ 04_LLaDA_MoE               |
                    |   -------------------------     |
                    |   Efficiency through Sparsity   |
                    |   • Mixture of Experts          |
                    |   • Compute-efficient inference |
                    +================================+
                                     |
                    +----------------+----------------+
                    |     🎓 YOU'RE NOW AN EXPERT!    |
                    +---------------------------------+

```

</div>

---

## 📂 Repository Map

```
📦 Diffusion Language Model
|
+-- 🎓 00_Diffusion_Theory/
|   |
|   +-- 01 Data Space/              → What is our playground?
|   +-- 02 Probability Assumptions/ → The math foundations
|   +-- 03 Gaussian Transition/     → How noise is added
|   +-- 04 Forward Process/         → Corruption step-by-step
|   +-- 05 Noise Schedule/          → Controlling the chaos
|   +-- 06 Marginal Distributions/  → Any-time noise levels
|   +-- 07 Reverse Process/         → Learning to denoise
|   +-- 08 Training Objective/      → What we optimize
|   +-- 09 Parameterization/        → Network design choices
|   +-- 10 Sampling/                → Generation algorithms
|
+-- 📄 01_Simple_and_Effective_MDLM/
|   +-- 📖 README.md                → Detailed blog post
|   +-- 🎨 svg_diagrams/            → Visual explanations
|   +-- 📑 paper.pdf                → Original paper
|
+-- 📄 02_Block_Diffusion/
|   +-- 📖 README.md                → Detailed blog post
|   +-- 🎨 images/                  → Visual explanations
|   +-- 📑 paper.pdf                → Original paper
|
+-- 📄 03_Dream_7B/
|   +-- 📖 README.md                → Detailed blog post
|   +-- 🎨 svg_diagrams/            → Visual explanations
|   +-- 📑 paper.pdf                → Original paper
|
+-- 📄 04_LLaDA_MoE/
    +-- 📖 README.md                → Detailed blog post
    +-- 🎨 images/                  → Visual explanations
    +-- 📑 paper.pdf                → Original paper

```

---

## 🧮 The Core Mathematics

<div align="center">

### Forward Process: Adding Noise

```
.--------------------------------------------------------------------.
|                                                                    |
|   At noise level t ∈ [0, 1], each token independently:            |
|                                                                    |
|   +---------------------------------------------------------+     |
|   |                                                         |     |
|   |   q(yₜⁱ | yⁱ) = + (1-t)  →  keep original token        |     |
|   |                 |                                       |     |
|   |                 +  t     →  replace with [MASK]        |     |
|   |                                                         |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   t = 0.0  →  "The quick brown fox"        (clean)                |
|   t = 0.3  →  "The [M] brown [M]"          (some masked)          |
|   t = 0.7  →  "[M] [M] [M] fox"            (mostly masked)        |
|   t = 1.0  →  "[M] [M] [M] [M]"            (fully masked)         |
|                                                                    |
'--------------------------------------------------------------------'

```

### Training Objective: ELBO

```
.--------------------------------------------------------------------.
|                                                                    |
|   Minimize the negative log-likelihood bound:                      |
|                                                                    |
|   +---------------------------------------------------------+     |
|   |                                                         |     |
|   |   𝓛(θ) = −𝔼   𝔼      𝔼     [ 1/t · Σᵢ 𝟙[yₜⁱ=M]        |     |
|   |           y   t~U    yₜ                                 |     |
|   |                                                         |     |
|   |                            · log pθ(yⁱ | yₜ) ]         |     |
|   |                                                         |     |
|   +---------------------------------------------------------+     |
|                                                                    |
|   Key insight: 1/t weighting makes gradients unbiased!            |
|                                                                    |
'--------------------------------------------------------------------'

```

</div>

---

## 📊 Paper Overview

<div align="center">

| | Paper | Innovation | Key Idea |
|:---:|:---|:---|:---|
| 🎭 | **MDLM** | Masked Diffusion | Absorbing state with simplified training |
| 🔀 | **Block Diffusion** | Hybrid AR+Diffusion | Blocks for AR, diffusion within blocks |
| 🚀 | **Dream 7B** | Scaling Laws | First competitive large diffusion LLM |
| ⚡ | **LLaDA-MoE** | Sparse Efficiency | MoE architecture for fast inference |

</div>

---

## 🔗 Resources & Links

<div align="center">

| Paper | arXiv | Code/Models |
|:------|:-----:|:-----------:|
| Simple & Effective MDLM | [📄 2406.07524](https://arxiv.org/abs/2406.07524) | — |
| Block Diffusion | [📄 2503.09573](https://arxiv.org/abs/2503.09573) | — |
| Dream 7B | [📄 2508.15487](https://arxiv.org/abs/2508.15487) | [🤗 HuggingFace](https://huggingface.co/Dream-org) |
| LLaDA-MoE | [📄 2509.24389](https://arxiv.org/abs/2509.24389) | [🤗 HuggingFace](https://huggingface.co/collections/inclusionAI/llada) |

</div>

---

## 🚀 Quick Start

```bash
# Start with the theory
cd "00_Diffusion_Theory"
cat README.md

# Then explore each paper in order
cd "../01_Simple_and_Effective_Masked_Diffusion_Language_Models"
cd "../02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion"
cd "../03_Dream_7B_Diffusion_Large_Language_Models"
cd "../04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model"

```

---

## 💡 Key Insights

<div align="center">

```
.--------------------------------------------------------------------------.
|                                                                          |
|                     🌟 THE BIG PICTURE 🌟                                |
|                                                                          |
|  +----------------+  +----------------+  +----------------+             |
|  |                |  |                |  |                |             |
|  |   PARALLEL     |  |   REVISABLE    |  |   EFFICIENT    |             |
|  |   GENERATION   |  |   OUTPUTS      |  |   WITH MoE     |             |
|  |                |  |                |  |                |             |
|  |  All tokens    |  |  Fix mistakes  |  |  Sparse        |             |
|  |  at once       |  |  iteratively   |  |  computation   |             |
|  |                |  |                |  |                |             |
|  +-------+--------+  +-------+--------+  +-------+--------+             |
|          |                   |                   |                      |
|          +-------------------+-------------------+                      |
|                              |                                          |
|                              ▼                                          |
|                    +-----------------+                                  |
|                    |                 |                                  |
|                    |  THE FUTURE OF  |                                  |
|                    |  LANGUAGE AI    |                                  |
|                    |                 |                                  |
|                    +-----------------+                                  |
|                                                                          |
'--------------------------------------------------------------------------'

```

</div>

---

<div align="center">

### 📜 License

*Educational purposes only. All papers belong to their respective authors.*

---

**Made with 💜 for the Machine Learning Community**

<br>

*"The best way to predict the future is to invent it."*

</div>

