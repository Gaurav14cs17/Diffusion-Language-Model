<div align="center">

# 📈 Noise Schedule

<img src="https://github.com/Gaurav14cs17/Diffusion-Language-Model/blob/main/00_Diffusion_Theory/05%20Noise%20Schedule/noise_schedule.svg" alt="Typing SVG" />

[← Forward Process](../04%20Forward%20Process/) · **Page 5 of 10** · [Next: Marginal Distributions →](../06%20Marginal%20Distributions/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

The **noise schedule** $\{\beta\_t\}\_{t=1}^{T}$ controls how quickly noise is added. Different schedules lead to different training dynamics and sample quality.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/05%20Noise%20Schedule/noise_schedule.svg" alt="Noise Schedule Diagram" width="100%">
</p>

---

## 📐 Step 1: Linear Schedule (DDPM)

### Definition

```math
\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})

```

### Typical Values (DDPM)

| Parameter | Value |
|:---------:|:-----:|
| $\beta\_{\min}$ | $10^{-4}$ |
| $\beta\_{\max}$ | $0.02$ |
| $T$ | 1000 |

### Derivation of $\bar{\alpha}\_t$

```math
\alpha_t = 1 - \beta_t
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)

```

### Properties

✅ Simple to implement  
✅ Works well for many tasks  
❌ Information destroyed too quickly in early steps  
❌ Too slow at end

### Visualization

```
β_t
0.02 +                              ●●●●●●
     |                         ●●●●●
     |                    ●●●●●
     |               ●●●●●
     |          ●●●●●
0.0001+●●●●●●●●●
      +-----------------------------------▶ t
      0                                   T

```

---

## 📐 Step 2: Cosine Schedule

### Motivation

Design $\bar{\alpha}\_t$ directly for **smoother** information destruction.

### Definition

```math
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)

```

where $s = 0.008$ is a small offset to prevent $\beta\_t$ from being too small at $t=0$.

### Derived $\beta\_t$

```math
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = 1 - \frac{f(t)}{f(t-1)}

```

### Proof of Smoothness

The cosine function provides:

1. $\bar{\alpha}\_0 \approx 1$ (starts at pure signal)

2. $\bar{\alpha}\_T \approx 0$ (ends at pure noise)

3. Smooth S-curve transition (no sudden jumps)

### Comparison with Linear

| Property | Linear | Cosine |
|----------|:------:|:------:|
| Early steps | Fast destruction | Slow, preserves detail |
| Middle steps | Linear decay | Smooth transition |
| Late steps | Slow | Fast, efficient |

### Visual Comparison

```
ᾱ_t (signal remaining)
1.0 +● ●                              Linear: ----
    | ● ●●                            Cosine: ····
0.8 |  ●  ●●
    |   ·   ●●●
0.6 |    ·     ●●●
    |     ·       ●●●
0.4 |      ··         ●●●
    |        ··           ●●●●
0.2 |          ···            ●●●●●
    |             ····             ●●●●
0.0 |                 ·····················
    +-------------------------------------▶ t

```

---

## 📐 Step 3: Schedule Design Principles

### Key Requirements

1. **Boundary conditions**:
   - $\bar{\alpha}\_0 \approx 1$ (start with data)
   - $\bar{\alpha}\_T \approx 0$ (end with noise)

2. **Smoothness**:
   - No sudden jumps in $\beta\_t$
   - Gradual information destruction

3. **Efficiency**:
   - Not too slow (waste compute)
   - Not too fast (hard to learn)

### Mathematical Constraints

```math
0 < \beta_t < 1 \quad \forall t
\prod_{t=1}^{T} (1-\beta_t) \approx 0

```

### Proof of Final Noise

For the process to destroy information:

```math
\bar{\alpha}_T = \prod_{t=1}^{T} (1-\beta_t) \leq \exp\left(-\sum_{t=1}^{T} \beta_t\right)

```

Using $\ln(1-x) \leq -x$. So if $\sum\_t \beta\_t \gg 1$, then $\bar{\alpha}\_T \approx 0$. ✓

---

## 📐 Step 4: Other Schedules

### Quadratic Schedule

```math
\beta_t = \beta_{\min} + \left(\frac{t-1}{T-1}\right)^2 (\beta_{\max} - \beta_{\min})

```

### Sigmoid Schedule

```math
\beta_t = \sigma\left(\frac{t - T/2}{\tau}\right) \cdot (\beta_{\max} - \beta_{\min}) + \beta_{\min}

```

### Learned Schedule

Train $\beta\_t$ as neural network parameters! (Kingma et al., VDM)

### Comparison Table

| Schedule | Pros | Cons |
|----------|------|------|
| **Linear** | Simple, baseline | Suboptimal |
| **Cosine** | Smooth, better quality | Slightly complex |
| **Quadratic** | Slow start | Fast end |
| **Learned** | Optimal for task | Expensive to train |

---

## 📐 Step 5: SNR Schedule Perspective

### Signal-to-Noise Ratio

```math
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}

```

### Log-SNR (more useful)

```math
\log\text{SNR}(t) = \log\bar{\alpha}_t - \log(1-\bar{\alpha}_t)

```

### Derivation from Schedule

For linear schedule:

```math
\log\text{SNR}(t) \approx \log(1-\frac{t}{T}) - \log(\frac{t}{T})

```

For cosine schedule:

```math
\log\text{SNR}(t) = 2\log\cos\left(\frac{\pi t}{2T}\right) - 2\log\sin\left(\frac{\pi t}{2T}\right)

```

### Visual

```
log SNR
  ∞  +●
     |●
  10 + ●
     |  ●●
   0 +    ●●●●
     |        ●●●●
 -10 +            ●●●●●●
     |                  ●●●
 -∞  +                    ●
     +---------------------▶ t
     0        T/2         T

```

---

## 📐 Step 6: Computing $\beta\_t$ from $\bar{\alpha}\_t$

### Problem

Given a desired $\bar{\alpha}\_t$ schedule, find $\beta\_t$.

### Derivation

```math
\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s) = \bar{\alpha}_{t-1} \cdot (1-\beta_t)

```

Solving for $\beta\_t$:

```math
1-\beta_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
\boxed{\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}

```

### Clipping

In practice, clip $\beta\_t$ to avoid numerical issues:

```math
\beta_t = \text{clip}\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0, 0.999\right)

```

---

## 🔑 Summary

<div align="center">

| Schedule | Formula for $\bar{\alpha}\_t$ | Best For |
|----------|------------------------------|----------|
| **Linear** | $\prod(1-\beta\_{\text{lin}})$ | Baseline |
| **Cosine** | $\cos^2(\frac{\pi t}{2T})$ | High quality images |
| **Learned** | Neural network | Task-specific |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Forward Process](../04%20Forward%20Process/) · **Page 5 of 10** · [Next: Marginal Distributions →](../06%20Marginal%20Distributions/)

</div>
