<div align="center">

# ⬅️ Reverse Process

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=Learning+to+Denoise;From+Noise+to+Data" alt="Typing SVG" />

[← Marginal Distributions](../06%20Marginal%20Distributions/) · **Page 7 of 10** · [Next: Training Objective →](../08%20Training%20Objective/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

The **reverse process** learns to undo the forward diffusion — gradually denoising pure noise back into data samples!

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/07%20Reverse%20Process/reverse_process.svg" alt="Reverse Process Diagram" width="100%">
</p>

---

## 📐 Step 1: The Challenge

### What We Want

Given $x\_t$ (noisy), generate $x\_{t-1}$ (less noisy).

### The Problem

```math
p(x_{t-1} \mid x_t) = \frac{p(x_t \mid x_{t-1}) p(x_{t-1})}{p(x_t)}
```

We need $p(x\_{t-1})$ — the marginal distribution of data at step $t-1$!

This is **intractable** because:
1. $p(x\_0) = p\_{\text{data}}$ is unknown
2. $p(x\_t) = \int p(x\_t \mid x\_0) p(x\_0) dx\_0$ is intractable

### The Solution

**Learn** $p\_\theta(x\_{t-1} \mid x\_t)$ with a neural network!

```
+-------------------------------------------------------------+
|  Forward:  Known, fixed (Gaussian transitions)              |
|  Reverse:  Unknown, learned (neural network)                |
+-------------------------------------------------------------+
```

---

## 📐 Step 2: Parameterizing the Reverse Process

### Key Insight

For small $\beta\_t$, the reverse process is **also Gaussian**! (See Feller, 1949)

### Parameterization

```math
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
```

### Variance Choices

| Choice | Formula | Notes |
|--------|---------|-------|
| **Fixed (DDPM)** | $\sigma\_t^2 = \beta\_t$ | Simple, works well |
| **Fixed (optimal)** | $\sigma\_t^2 = \tilde{\beta}\_t$ | Lower bound |
| **Learned** | $\sigma\_t^2 = \exp(v \log\beta\_t + (1-v)\log\tilde{\beta}\_t)$ | Interpolate |

### Focus: Learning the Mean

The mean $\mu\_\theta(x\_t, t)$ is what the network must predict!

---

## 📐 Step 3: What Should the Mean Be?

### Target: The True Posterior

If we knew $x\_0$, the optimal reverse step would be:

```math
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)
```

where:

```math
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
```

### The Problem

We don't know $x\_0$ during generation!

### The Solution

**Predict** $x\_0$ from $x\_t$:

```math
\hat{x}_0 = f_\theta(x_t, t)
```

Then substitute into $\tilde{\mu}\_t$!

---

## 📐 Step 4: Three Parameterizations

### Option A: Predict $x\_0$ Directly

```math
\mu_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
```

### Option B: Predict $\epsilon$ (Most Common!)

Since $x\_0 = \frac{x\_t - \sqrt{1-\bar{\alpha}\_t}\epsilon}{\sqrt{\bar{\alpha}\_t}}$:

```math
\boxed{\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)}
```

### Proof of Equivalence

Start with $\mu\_\theta$ in terms of $x\_0$:

```math
\mu_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
```

Substitute $x\_0 = \frac{x\_t - \sqrt{1-\bar{\alpha}\_t}\epsilon}{\sqrt{\bar{\alpha}\_t}}$:

After algebraic simplification (tedious but straightforward):

```math
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right) \quad \checkmark
```

### Option C: Predict $v$ (Velocity)

Define: $v\_t = \sqrt{\bar{\alpha}\_t}\epsilon - \sqrt{1-\bar{\alpha}\_t}x\_0$

Then: $\epsilon = \sqrt{\bar{\alpha}\_t}v\_t + \sqrt{1-\bar{\alpha}\_t}x\_t/\sqrt{\bar{\alpha}\_t}$

(See Parameterization chapter for details)

---

## 📐 Step 5: The Sampling Algorithm

### DDPM Sampling (Algorithm)

```python
# Input: Trained noise predictor ε_θ
# Output: Generated sample x_0

x_T ~ N(0, I)  # Start from pure noise

for t in [T, T-1, ..., 1]:
    # Predict noise
    ε = ε_θ(x_t, t)
    
    # Compute mean
    μ = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε)
    
    # Sample (add noise except at t=1)
    if t > 1:
        z ~ N(0, I)
        x_{t-1} = μ + σ_t * z
    else:
        x_0 = μ
        
return x_0
```

### Why Add Noise?

Even though we're denoising, we add small noise $\sigma\_t z$ because:
1. Matches the true reverse posterior
2. Helps exploration / diversity
3. Only skip at $t=1$ to get clean final sample

### Visual

```
t=T        t=T-1      t=T-2       ...      t=1       t=0
⚪ ----▶ 🌫️ ----▶ 🌫️ ----▶ ... ----▶ 📉 ----▶ 📊
noise    denoise   denoise            denoise   clean!
```

---

## 📐 Step 6: Derivation of Reverse Variance

### True Posterior Variance

```math
\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}
```

### Proof

From completing the square in the posterior derivation:

Precision (inverse variance):

```math
\tilde{\beta}_t^{-1} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}
= \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}
```

Using $\alpha\_t + \beta\_t = 1$:

```math
= \frac{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}{\beta_t(1-\bar{\alpha}_{t-1})}
= \frac{1 - \bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}
```

Therefore:

```math
\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \quad \checkmark
```

### Two Common Choices

| $\sigma\_t^2$ | Formula | When to Use |
|:------------:|---------|-------------|
| $\beta\_t$ | Fixed, simple | Default (DDPM) |
| $\tilde{\beta}\_t$ | $\frac{(1-\bar{\alpha}\_{t-1})\beta\_t}{1-\bar{\alpha}\_t}$ | Theoretically optimal |

Both give similar results for well-trained models!

---

## 🔑 Summary

<div align="center">

| Component | Formula |
|-----------|---------|
| **Reverse transition** | $p\_\theta(x\_{t-1} \mid x\_t) = \mathcal{N}(\mu\_\theta, \sigma\_t^2 I)$ |
| **Mean (ε-pred)** | $\mu\_\theta = \frac{1}{\sqrt{\alpha\_t}}(x\_t - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon\_\theta)$ |
| **Variance** | $\sigma\_t^2 = \beta\_t$ or $\tilde{\beta}\_t$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Marginal Distributions](../06%20Marginal%20Distributions/) · **Page 7 of 10** · [Next: Training Objective →](../08%20Training%20Objective/)

</div>
