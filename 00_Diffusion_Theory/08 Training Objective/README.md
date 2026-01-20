<div align="center">

# 🎯 Training Objective

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=From+ELBO+to+Simple+MSE;The+Elegant+Simplification" alt="Typing SVG" />

[← Reverse Process](../07%20Reverse%20Process/) · **Page 8 of 10** · [Next: Parameterization →](../09%20Parameterization/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

We derive the training loss, starting from the **Evidence Lower Bound (ELBO)** and simplifying to the surprisingly simple **MSE noise prediction** loss!

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/08%20Training%20Objective/training_objective.svg" alt="Training Objective Diagram" width="100%">
</p>

---

## 📐 Step 1: Starting Point — The ELBO

### Goal

Maximize $\log p\_\theta(x\_0)$ — the likelihood of real data.

### Problem

```math
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) dx_{1:T}

```

This integral is intractable!

### Solution: Variational Bound

```math
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]

```

### Proof

By Jensen's inequality:

```math
\log p_\theta(x_0) = \log \int q(x_{1:T}|x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} dx_{1:T}
\geq \int q(x_{1:T}|x_0) \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} dx_{1:T}
= \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -\mathcal{L}_{\text{VLB}}

```

---

## 📐 Step 2: Decomposing the ELBO

### Expand the Ratio

```math
\mathcal{L}_{\text{VLB}} = \mathbb{E}_q\left[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]

```

### Factor Both Distributions

**Forward:** $q(x\_{1:T}|x\_0) = \prod\_{t=1}^T q(x\_t|x\_{t-1})$

**Reverse:** $p\_\theta(x\_{0:T}) = p(x\_T) \prod\_{t=1}^T p\_\theta(x\_{t-1}|x\_t)$

### Substitute and Expand

```math
\mathcal{L}_{\text{VLB}} = \mathbb{E}_q\left[\log \frac{\prod_{t=1}^T q(x_t|x_{t-1})}{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)}\right]
= \mathbb{E}_q\left[-\log p(x_T) + \sum_{t=1}^T \log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}\right]

```

### Rewrite Using Posterior

For $t > 1$:

```math
\log \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} = \log \frac{q(x_{t-1}|x_t, x_0)}{p_\theta(x_{t-1}|x_t)} + \log \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}

```

### Final Decomposition

```math
\boxed{\mathcal{L}_{\text{VLB}} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}}

```

---

## 📐 Step 3: Understanding Each Term

### The Three Components

| Term | Formula | Meaning |
|:----:|---------|---------|
| $L\_T$ | $D\_{KL}(q(x\_T\|x\_0) \| p(x\_T))$ | Prior matching |
| $L\_{t-1}$ | $D\_{KL}(q(x\_{t-1}\|x\_t,x\_0) \| p\_\theta(x\_{t-1}\|x\_t))$ | Denoising steps |
| $L\_0$ | $-\log p\_\theta(x\_0\|x\_1)$ | Reconstruction |

### Key Insight

- **$L\_T$**: Constant (no learnable params)! Just measures how close $x\_T$ is to $\mathcal{N}(0,I)$

- **$L\_{t-1}$**: The main learning signal! Match reverse to true posterior

- **$L\_0$**: Discretized decoder (often ignored or simplified)

---

## 📐 Step 4: KL Between Gaussians

### Setup

Both $q(x\_{t-1}|x\_t,x\_0)$ and $p\_\theta(x\_{t-1}|x\_t)$ are Gaussian!

- True: $\mathcal{N}(\tilde{\mu}\_t, \tilde{\beta}\_t I)$

- Model: $\mathcal{N}(\mu\_\theta, \sigma\_t^2 I)$

### KL for Gaussians

```math
D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2 I) \| \mathcal{N}(\mu_2, \sigma_2^2 I)) = \frac{1}{2\sigma_2^2}\|\mu_1 - \mu_2\|^2 + \text{const}

```

(when variances are fixed)

### Apply to Our Case

```math
L_{t-1} = \frac{1}{2\sigma_t^2}\|\tilde{\mu}_t - \mu_\theta\|^2 + C

```

### The Key Result

```math
\boxed{L_{t-1} \propto \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2}

```

We just need to **match the means**!

---

## 📐 Step 5: From Mean Matching to Noise Prediction

### Recall the Means

**True posterior mean:**

```math
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)

```

**Model mean (ε-prediction):**

```math
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)

```

### Compute the Difference

```math
\tilde{\mu}_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}(\epsilon_\theta - \epsilon)

```

### Substitute into Loss

```math
L_{t-1} \propto \left\|\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}(\epsilon_\theta - \epsilon)\right\|^2
= \frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2

```

### The Simplified Loss (DDPM)

Ho et al. found that **ignoring the coefficient** works better!

```math
\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}

```

where $x\_t = \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon$.

---

## 📐 Step 6: Why Does Simplified Work?

### The Weighting

| Loss | Weight on $L\_t$ |
|------|-----------------|
| VLB | $\frac{\beta\_t^2}{\alpha\_t(1-\bar{\alpha}\_t)}$ |
| Simple | $1$ (uniform) |

### Empirical Finding

- VLB weighting: Emphasizes early steps (high $t$)

- Uniform weighting: Better sample quality!

### Intuition

Early steps (high $t$) are "easy" — just predicting noise direction.
Late steps (low $t$) are "hard" — need fine details.

Uniform weighting gives more attention to hard steps!

### Visual

```
VLB weights:
█████████░░░░░░░░░░░░░░░░░░░░  Early t (high weight)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░█  Late t (low weight)

Simple (uniform):
████████████████████████████████  All t equal

```

---

## 📐 Step 7: Complete Training Algorithm

### Training Loop

```python
# Initialize model ε_θ with random weights
# Set noise schedule {β_t}, compute {ᾱ_t}

for each training iteration:
    # 1. Sample data
    x_0 ~ p_data
    
    # 2. Sample timestep uniformly
    t ~ Uniform({1, ..., T})
    
    # 3. Sample noise
    ε ~ N(0, I)
    
    # 4. Create noisy sample
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    
    # 5. Predict noise
    ε_pred = ε_θ(x_t, t)
    
    # 6. Compute loss
    loss = ||ε - ε_pred||²
    
    # 7. Gradient update
    θ = θ - lr · ∇_θ loss

```

### Key Points

1. **Random $t$**: Train on all timesteps equally

2. **Simple MSE**: Just match the noise!

3. **Efficient**: One forward pass per iteration

### Time Conditioning

The model $\epsilon\_\theta(x\_t, t)$ must know which timestep it's at!

Common approaches:

- Sinusoidal embeddings (like Transformers)

- Learned embeddings

- Fourier features

---

## 🔑 Summary

<div align="center">

| From | To | Key Step |
|------|-----|----------|
| Log-likelihood | ELBO | Jensen's inequality |
| ELBO | Sum of KLs | Decomposition |
| KL terms | Mean matching | Gaussians |
| Mean matching | Noise prediction | Reparameterization |
| Weighted loss | **Simple MSE** | Drop weights! |

### Final Training Loss

```math
\boxed{\mathcal{L} = \mathbb{E}_{t \sim U[1,T], x_0, \epsilon \sim \mathcal{N}(0,I)}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]}

```

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Reverse Process](../07%20Reverse%20Process/) · **Page 8 of 10** · [Next: Parameterization →](../09%20Parameterization/)

</div>
