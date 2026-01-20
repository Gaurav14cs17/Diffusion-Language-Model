# Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models

> **Paper**: [arXiv:2503.09573](https://arxiv.org/abs/2503.09573) | **Published at ICLR 2025**  
> **Authors**: Marianne Arriola, Zhixuan Qi, Aaron Kerem Gokaslan, Subham Sekhar Sahoo, Jiaqi Han, Justin T. Chiu, Zhihan Yang, Volodymyr Kuleshov  
> **Institution**: Cornell Tech  
> **Code**: [github.com/kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/01_Overview_Block_Diffusion.svg" alt="Overview" width="100%">
</p>

## Introduction

Language models have revolutionized AI, but the dominant paradigm—**autoregressive (AR) generation**—comes with a fundamental limitation: tokens must be generated one at a time, sequentially. This sequential dependency bottleneck has led researchers to explore **diffusion models** as an alternative, which can generate multiple tokens in parallel through iterative denoising.

However, discrete diffusion models for text have struggled to match the quality of autoregressive models and face their own limitations. This paper introduces **Block Discrete Denoising Diffusion Language Models (BD3-LMs)**, a novel approach that elegantly interpolates between these two paradigms, capturing the best of both worlds.

## The Problem: Three Limitations of Diffusion Language Models

Before diving into the solution, let's understand why we need Block Diffusion in the first place. Current discrete diffusion models face **three critical limitations**:

### 1. Fixed-Length Generation
Most diffusion architectures can only generate fixed-length sequences. In real applications like chatbots, we need to generate responses of varying lengths—sometimes a short "Yes" and sometimes a detailed multi-paragraph explanation.

### 2. No KV Caching
Diffusion models use bidirectional context during generation, which means they can't reuse previous computations with KV (Key-Value) caching. Every generation step requires recomputing attention over the entire sequence, making inference significantly slower.

### 3. Lower Quality
Despite recent advances, diffusion models still lag behind autoregressive models in perplexity—a standard measure of language modeling quality. This gap has limited their practical applicability.

---

## The Solution: Block Diffusion

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/02_Mathematical_Foundations.svg" alt="Mathematical Foundations" width="100%">
</p>

### Core Idea

Block Diffusion introduces a beautifully simple idea: **divide the sequence into blocks and apply autoregressive generation over blocks, while using diffusion within each block**.

Mathematically, given a sequence of L tokens divided into B blocks of size L' each:

```
log p_θ(x) = Σ_{b=1}^{B} log p_θ(x^b | x^{1:b-1})
```

Each conditional `p_θ(x^b | x^{1:b-1})` is modeled using discrete diffusion over a block of L' tokens, conditioned on all previous blocks.

### The Block Size Spectrum

The block size L' acts as a continuous dial between two extremes:

| Block Size | Equivalent To | Properties |
|------------|---------------|------------|
| **L' = 1** | Autoregressive | Best quality, no parallelism |
| **L' = L** | Full Diffusion | Maximum parallelism, fixed length |
| **1 < L' < L** | Block Diffusion | **Best of both worlds!** |

This elegant parameterization allows us to trade off between quality and parallelism based on application needs.

---

## How It Works: Training and Sampling

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/03_Training_Sampling_Algorithms.svg" alt="Training and Sampling Algorithms" width="100%">
</p>

### Training Algorithm

The training algorithm is surprisingly efficient, requiring only **two forward passes** through the model:

1. **Sample noise levels** t₁, ..., t_B for each block
2. **Apply noise** to each block according to its noise level
3. **Compute KV cache** for the clean sequence (first forward pass)
4. **Compute predictions** for all blocks using cached KVs (second forward pass)
5. **Take gradient step** on the block diffusion loss

The key insight is using a **vectorized implementation** that processes all blocks in parallel, computing the loss for the entire sequence in one batched forward pass.

### Sampling Algorithm

Sampling proceeds block by block:

```
for b = 1 to B:
    x^b ← SAMPLE(x^b_θ, K_{1:b-1}, V_{1:b-1})  # Diffusion sampling
    K_b, V_b ← x^b_θ(x^b)                       # Update KV cache
    x ← x ⊕ x^b                                 # Append to output
return x
```

**Key advantages:**
- ✅ **Variable-length generation**: Can generate any number of blocks
- ✅ **KV caching**: Reuse computation from previous blocks
- ✅ **Parallel sampling**: Tokens within a block are sampled in parallel

### Block-Causal Attention

The architecture uses a **block-causal attention mask** where tokens in block b can attend to all tokens in blocks 1 through b. This is more permissive than token-level causal masking (AR) but still enables KV caching.

---

## The Gradient Variance Problem

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/04_Gradient_Variance_Noise_Schedules.svg" alt="Gradient Variance and Noise Schedules" width="100%">
</p>

### A Surprising Discovery

Here's something surprising: with block size L' = 1, the Block Diffusion objective is **mathematically equivalent** to the autoregressive negative log-likelihood in expectation. Yet, there's a ~2 point perplexity gap between them!

**Why?** The answer lies in **gradient variance**.

### The Root Cause

- **Autoregressive training**: Computes cross-entropy loss for ALL L tokens in every batch
- **Diffusion training**: Only computes loss for MASKED tokens (~50% on average)

This means diffusion effectively trains with half the data per batch, leading to **2x higher gradient variance**.

### The Masking Rate Problem

Not all mask rates are equally useful for learning:

| Mask Rate | Problem |
|-----------|---------|
| ≈ 0 (few masked) | Reconstruction is trivially easy, no learning signal |
| ≈ 1 (all masked) | Optimal prediction is just marginal distribution, also easy |
| **0.3 - 0.7** | **Sweet spot**: Challenging but learnable |

Standard linear schedules sample uniformly over [0, 1], including these "bad" regions!

---

## Low-Variance Noise Schedules

### Clipped Schedules

The solution is elegantly simple: **restrict the mask rate to a useful range**:

```
1 - α_t ~ U[β, ω]    where 0 ≤ β < ω ≤ 1
```

This "clips" the schedule to avoid extreme mask rates that produce poor gradients.

### Data-Driven Optimization

Different block sizes need different optimal schedules. The paper proposes:

```
min_{β,ω} Var_{X,t}[L(X; θ, β, ω)]
```

A grid search finds the optimal (β, ω) for each block size during training.

### Results

| Block Size | Schedule | Variance | PPL |
|------------|----------|----------|-----|
| L' = 1 | Linear (standard) | 1.52 | ≤25.56 |
| L' = 1 | **Full masking** | **0.11** | **22.88** |

With the tuned schedule, L' = 1 Block Diffusion **exactly matches** autoregressive perplexity!

---

## Architecture Deep Dive

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/05_Architecture_Flow_Diagram.svg" alt="Architecture Flow Diagram" width="100%">
</p>

### Model Signature

The model has a clean functional signature:

```
x^b_logits, K_b, V_b ← x^b_θ(x^b_t, K_{1:b-1}, V_{1:b-1})
```

Where:
- `x^b_logits`: Predictions for clean tokens in block b
- `K_b, V_b`: Key-value cache for block b
- `K_{1:b-1}, V_{1:b-1}`: Cached KVs from previous blocks

### Within-Block Diffusion

The diffusion process within each block proceeds through multiple denoising steps:

```
t=1.0: [MASK][MASK][MASK][MASK]     (all masked)
t=0.75: [~][MASK][~][MASK]          (~25% revealed)
t=0.50: [the][~][is][~]             (~50% revealed)
t=0.25: [the][cat][is][~]           (~75% revealed)
t=0.0:  [the][cat][is][happy]       (block complete!)
```

**All tokens within a block are denoised IN PARALLEL** at each step—this is the source of speedup over autoregressive generation.

---

## Experimental Results

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/02_Block_Diffusion_Interpolating_Between_AR_and_Diffusion/images/06_Results_Experiments.svg" alt="Results and Experiments" width="100%">
</p>

### Main Results

| Model | Type | LM1B PPL ↓ | Variable Length | KV Cache |
|-------|------|------------|-----------------|----------|
| GPT-2 (AR baseline) | Autoregressive | 22.88 | ✓ | ✓ |
| MDLM | Diffusion | 31.73 | ✗ | ✗ |
| SEDD | Diffusion | 32.10 | ✗ | ✗ |
| **BD3-LM (L'=4)** | Block Diffusion | **27.52** | ✓ | ✓ |
| **BD3-LM (L'=1, tuned)** | Block Diffusion | **22.88** | ✓ | ✓ |

### Key Findings

1. **State-of-the-Art**: BD3-LM achieves the best perplexity among all discrete diffusion models

2. **Gap Closed**: With L'=1 and tuned schedule, Block Diffusion matches AR perplexity exactly

3. **Variable Length**: First diffusion model to support generation beyond training context length

4. **Efficient Inference**: 10x fewer generation steps than Gaussian diffusion baselines

5. **Strong Correlation**: NELBO variance is strongly correlated with test perplexity—lower variance → better model

---

## Practical Implications

### When to Use Block Diffusion

| Use Case | Recommended L' | Reason |
|----------|----------------|--------|
| Maximum quality | L' = 1 | Matches AR perplexity |
| Balanced | L' = 4-16 | Good quality + parallelism |
| Maximum speed | L' = 64-128 | Most parallel, slightly lower quality |

### Advantages Over Alternatives

**vs. Autoregressive:**
- Parallel generation within blocks
- Potentially better long-term planning (bidirectional context within blocks)

**vs. Full Diffusion:**
- Variable-length generation
- KV caching for efficiency
- Better perplexity

**vs. Continuous Diffusion (SSD-LM, etc.):**
- Tractable likelihood estimation
- 10x fewer generation steps
- Better generative perplexity

---

## Conclusion

Block Diffusion represents a significant step forward for language modeling. By elegantly interpolating between autoregressive and diffusion paradigms, it achieves:

✅ **State-of-the-art perplexity** among diffusion models  
✅ **Variable-length generation** (first for diffusion!)  
✅ **KV caching support** for efficient inference  
✅ **Parallel sampling** within blocks  
✅ **Theoretical insight** into the variance-quality connection  

The key insight—that **gradient variance explains the perplexity gap**—opens new research directions for improving diffusion models. The clipped schedule solution is simple yet powerful, achieving AR-level quality while maintaining diffusion's unique benefits.

As LLMs continue to scale and applications demand both quality and efficiency, Block Diffusion offers a compelling middle ground that may define the next generation of language models.

---

## Citation

```bibtex
@inproceedings{arriola2025block,
  title={Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models},
  author={Arriola, Marianne and Qi, Zhixuan and Gokaslan, Aaron Kerem and Sahoo, Subham Sekhar and Han, Jiaqi and Chiu, Justin T and Yang, Zhihan and Kuleshov, Volodymyr},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## Resources

- 📄 **Paper**: [arXiv:2503.09573](https://arxiv.org/abs/2503.09573)
- 💻 **Code**: [github.com/kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)
- 🌐 **Project Page**: [m-arriola.com/bd3lms](https://m-arriola.com/bd3lms)
- 🎥 **Video Tutorial**: Available on project page

---

*Last updated: December 2024*

