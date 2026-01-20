<div align="center">

# 🎲 Sampling

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=DDPM%2C+DDIM%2C+and+Beyond;From+Noise+to+Beautiful+Samples" alt="Typing SVG" />

[← Parameterization](../09%20Parameterization/) · **Page 10 of 10** · [🏠 Home](../README.md)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

Once trained, we need to **sample** from the diffusion model. This chapter covers DDPM, DDIM, and advanced solvers.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/10%20Sampling/sampling_algorithms.svg" alt="Sampling Algorithms Diagram" width="100%">
</p>

---

## 📐 Step 1: DDPM Sampling

### Algorithm

```python
x_T ~ N(0, I)  # Start from noise

for t in [T, T-1, ..., 1]:
    ε = ε_θ(x_t, t)  # Predict noise
    μ = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε)
    
    if t > 1:
        x_{t-1} = μ + σ_t * z,  z ~ N(0, I)
    else:
        x_0 = μ

```

### Mean Formula

```math
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right)

```

| ✅ Pros | ❌ Cons |
|---------|---------|
| Simple, high quality | Slow (1000 steps) |
| Stochastic diversity | Fixed step count |

---

## 📐 Step 2: DDIM Sampling

### Key Insight

DDPM's stochasticity is optional! We can make sampling **deterministic**.

### Update Rule

```math
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta + \sigma_t \epsilon_t

```

where $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$

### The η Parameter

| η | Behavior |
|:-:|----------|
| 0 | Deterministic (DDIM) |
| 1 | Stochastic (DDPM) |

### Accelerated Sampling

DDIM allows using fewer steps: 1000 → 50 → 20 → 10!

---

## 📐 Step 3: Comparing DDPM vs DDIM

| Property | DDPM | DDIM |
|----------|:----:|:----:|
| Stochastic | ✅ | ❌ (η=0) |
| Acceleratable | ❌ | ✅ |
| Quality@1000 | High | High |
| Quality@50 | N/A | Good |

---

## 📐 Step 4: Advanced Samplers

### DPM-Solver

Uses higher-order ODE solvers:

| Order | Steps Needed |
|:-----:|:------------:|
| 1 | ~100 |
| 2 | ~20-50 |
| 3 | ~10-20 |

### Comparison

| Sampler | Steps for Quality |
|---------|:-----------------:|
| DDPM | 1000 |
| DDIM | 50-100 |
| DPM-Solver++ | 10-20 |

---

## 📐 Step 5: Classifier-Free Guidance

### Formula

```math
\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, \varnothing) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))

```

### Effect of Guidance Scale

| w | Effect |
|:-:|--------|
| 1 | Standard conditional |
| 3-7 | Typical for images |
| 7-15 | Strong guidance |
| >15 | Oversaturated |

---

## 📐 Step 6: Complete Algorithm

```python
def sample_ddim(model, timesteps, guidance_scale):
    x = torch.randn(shape)
    
    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # CFG
        eps_uncond = model(x, t, null)
        eps_cond = model(x, t, condition)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # Predict x_0
        x0 = (x - sqrt(1-alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
        
        # DDIM update
        x = sqrt(alpha_bar[t_prev]) * x0 + sqrt(1-alpha_bar[t_prev]) * eps
    
    return x

```

---

## 🔑 Summary

<div align="center">

| Sampler | Steps | Speed |
|---------|:-----:|:-----:|
| **DDPM** | 1000 | 🐢 |
| **DDIM** | 50-100 | 🐇 |
| **DPM-Solver++** | 10-20 | 🚀 |

| Algorithm | Key Formula |
|-----------|-------------|
| **DDPM** | $x_{t-1} = \mu_\theta + \sigma_t z$ |
| **DDIM** | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta$ |
| **CFG** | $\tilde{\epsilon} = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

## 🎉 Congratulations!

You've completed the Diffusion Theory course!

[← Parameterization](../09%20Parameterization/) · **Page 10 of 10** · [🏠 Back to Home](../README.md)

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=10B981&center=true&vCenter=true&width=500&lines=You're+now+a+Diffusion+Expert!+🎓;Go+build+something+amazing!+🚀" alt="Congrats" />

</div>
