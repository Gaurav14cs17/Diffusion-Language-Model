<div align="center">

# 📊 Data Space

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=Where+Real+Data+Lives;The+Data+Manifold" alt="Typing SVG" />

[← Home](../README.md) · **Page 1 of 10** · [Next: Probability Assumptions →](../02%20Probability%20Assumptions/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

The **data space** $\mathcal{X} \subseteq \mathbb{R}^d$ is the domain where real-world data exists. Understanding this space is foundational to understanding diffusion models.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/01%20Data%20Space/data_space.svg" alt="Data Space Diagram" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/01%20Data%20Space/data_space_theory.svg" alt="Data Space Theory" width="100%">
</p>

---

## 📐 Step 1: Formal Definition

### Statement

Real-world data comes from an unknown **data distribution** $p\_{\text{data}}(x)$.

### Derivation

1. **Define the space**: Let $\mathcal{X} \subseteq \mathbb{R}^d$ be our data space
2. **Define the distribution**: $p\_{\text{data}}: \mathcal{X} \to \mathbb{R}\_{\geq 0}$ is a probability density
3. **Normalization constraint**:

```math
\int_{\mathcal{X}} p_{\text{data}}(x) \, dx = 1

```

4. **Sampling notation**: We write $x \sim p\_{\text{data}}$ or $x\_0 \sim p\_{\text{data}}$

### Key Insight

> 💡 We never know $p\_{\text{data}}$ exactly — we only have **samples** from it!

---

## 📐 Step 2: Types of Data Spaces

### Continuous vs Discrete

| Type | Space | Example |
|:----:|-------|---------|
| **Continuous** | $\mathcal{X} = \mathbb{R}^d$ | Images, audio |
| **Discrete** | $\mathcal{X} = \{1, ..., K\}^d$ | Text tokens |
| **Mixed** | Combination | Structured data |

### For Images

```math
x \in [0, 1]^{H \times W \times C}

```

where:
- $H$ = height
- $W$ = width  
- $C$ = channels (3 for RGB)

### For Latent Diffusion

```math
z = \text{Encoder}(x), \quad z \in \mathbb{R}^{h \times w \times c}

```

Latent space is typically **8× compressed** per dimension.

---

## 📐 Step 3: The Data Manifold Hypothesis

### Statement

Real data lies on a **low-dimensional manifold** $\mathcal{M} \subset \mathbb{R}^d$.

### Intuition

1. **High ambient dimension**: Images are $256 \times 256 \times 3 = 196,608$ dimensions
2. **Low intrinsic dimension**: Most random vectors don't look like real images
3. **Manifold structure**: Real images form a thin "sheet" in high-dimensional space

### Formal Statement

```math
\dim(\mathcal{M}) \ll d
p_{\text{data}}(x) \approx 0 \text{ for } x \notin \mathcal{M}

```

### Why This Matters

| Implication | Benefit for Diffusion |
|-------------|----------------------|
| Data is structured | Model can learn patterns |
| Many dimensions unused | Compression possible |
| Local linearity | Gaussian noise works well |

---

## 📐 Step 4: Boundary Conditions

### At $t = 0$ (Start)

```math
x_0 \sim p_{\text{data}}(x)

```

This is our **clean data**.

### At $t = T$ (End)

```math
x_T \sim \mathcal{N}(0, I)

```

This is **pure noise** — data has been completely destroyed.

### Verification

The diffusion process must satisfy:

```math
\lim_{t \to 0} q(x_t) = p_{\text{data}}(x)
\lim_{t \to T} q(x_t) \approx \mathcal{N}(0, I)

```

### Visual

```
t=0          t=T/4         t=T/2         t=3T/4        t=T
📊 ----------- 📉 ----------- 📈 ----------- 🌫️ ----------- ⚪
Data         Some noise    Half noise    Mostly noise  Pure noise

```

---

## 📐 Step 5: Data Preprocessing

### Standard Practice

1. **Normalize to [-1, 1]**:

```math
x_{\text{norm}} = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1

```

2. **Why this range?**
   - Matches standard Gaussian noise range
   - Symmetric around 0
   - Prevents numerical issues

### For Images

```python
# Common preprocessing
x = (x / 255.0) * 2 - 1  # [0, 255] → [-1, 1]

```

### For Other Modalities

| Data Type | Preprocessing |
|-----------|---------------|
| Audio | Normalize waveform or spectrogram |
| Text | Embed to continuous space |
| 3D | Normalize point cloud |

---

## 🔑 Summary

<div align="center">

| Concept | Key Point |
|---------|-----------|
| **Data Space** | $\mathcal{X} \subseteq \mathbb{R}^d$ |
| **Distribution** | Unknown $p\_{\text{data}}(x)$ |
| **Manifold** | Data lives on low-dim surface |
| **Preprocessing** | Normalize to $[-1, 1]$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Home](../README.md) · **Page 1 of 10** · [Next: Probability Assumptions →](../02%20Probability%20Assumptions/)

</div>
