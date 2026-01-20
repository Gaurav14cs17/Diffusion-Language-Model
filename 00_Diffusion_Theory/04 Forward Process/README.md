<div align="center">

# ➡️ Forward Process

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=How+Noise+Destroys+Data;From+x₀+to+Pure+Noise" alt="Typing SVG" />

[← Gaussian Transition](../03%20Gaussian%20Transition%20Derivation/) · **Page 4 of 10** · [Next: Noise Schedule →](../05%20Noise%20Schedule/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

The forward process **gradually adds noise** to data over $T$ timesteps, transforming any data distribution into pure Gaussian noise.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/04%20Forward%20Process/forward_process.svg" alt="Forward Process Diagram" width="100%">
</p>

---

## 📐 Step 1: Single-Step Transition

### Definition

Each step adds a small amount of Gaussian noise:

```math
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)

```

### Equivalent Sampling Form

```math
x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)

```

### Notation

| Symbol | Definition | Typical Values |
|:------:|------------|----------------|
| $\beta\_t$ | Noise variance at step $t$ | $10^{-4}$ to $0.02$ |
| $\alpha\_t$ | $1 - \beta\_t$ (signal retention) | $0.98$ to $0.9999$ |
| $\bar{\alpha}\_t$ | $\prod\_{s=1}^{t} \alpha\_s$ (cumulative) | $1 \to 0$ |

### Why Small Steps?

```
Large β (big steps):  x₀ ----------------------▶ x_T  
                      Fast but hard to reverse!

Small β (small steps): x₀ -·-·-·-·-·-·-·-·-·-·-▶ x_T
                      Slow but easy to reverse!

```

---

## 📐 Step 2: Joint Distribution

### Markov Chain Factorization

The complete forward trajectory:

```math
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})

```

### Proof

By Markov property and chain rule:

1. $q(x\_{1:T} \mid x\_0) = q(x\_1 \mid x\_0) q(x\_2 \mid x\_1, x\_0) q(x\_3 \mid x\_2, x\_1, x\_0) \cdots$

2. Apply Markov: $q(x\_t \mid x\_{t-1}, ..., x\_0) = q(x\_t \mid x\_{t-1})$

3. Result: $q(x\_{1:T} \mid x\_0) = \prod\_{t=1}^{T} q(x\_t \mid x\_{t-1})$

### Full Joint (Including $x\_0$)

```math
q(x_{0:T}) = q(x_0) \prod_{t=1}^{T} q(x_t \mid x_{t-1})

```

---

## 📐 Step 3: Closed-Form Marginal

### Goal

Find $q(x\_t \mid x\_0)$ directly — skip intermediate steps!

### Derivation

**Step 1**: Write two consecutive steps:

```math
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1
x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2

```

**Step 2**: Substitute $x\_1$ into $x\_2$:

```math
x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1) + \sqrt{1-\alpha_2} \epsilon_2
= \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{\alpha_2(1-\alpha_1)} \epsilon_1 + \sqrt{1-\alpha_2} \epsilon_2

```

**Step 3**: Combine noise terms (sum of Gaussians):

```math
\sqrt{\alpha_2(1-\alpha_1)} \epsilon_1 + \sqrt{1-\alpha_2} \epsilon_2 \sim \mathcal{N}(0, [\alpha_2(1-\alpha_1) + 1-\alpha_2]I)

```

**Step 4**: Simplify variance:

```math
\alpha_2(1-\alpha_1) + 1-\alpha_2 = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1\alpha_2

```

**Step 5**: Define $\bar{\alpha}\_2 = \alpha\_1 \alpha\_2$:

```math
x_2 = \sqrt{\bar{\alpha}_2} x_0 + \sqrt{1-\bar{\alpha}_2} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

```

**Step 6**: By induction, for any $t$:

```math
\boxed{x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon}

```

where $\bar{\alpha}\_t = \prod\_{s=1}^{t} \alpha\_s$.

---

## 📐 Step 4: Distribution Form

### Marginal Distribution

```math
\boxed{q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)}

```

### Interpretation

| Component | Formula | Meaning |
|-----------|---------|---------|
| **Mean** | $\sqrt{\bar{\alpha}\_t} x\_0$ | Scaled original |
| **Variance** | $(1-\bar{\alpha}\_t) I$ | Accumulated noise |

### Visual Timeline

```
t=0:     x₀ = pure data, no noise
         Mean = x₀, Var = 0
         
t=T/2:   x_{T/2} = mix of signal and noise  
         Mean = √ᾱ_{T/2}·x₀ ≈ 0.5x₀
         Var ≈ 0.75I
         
t=T:     x_T ≈ pure noise
         Mean ≈ 0, Var ≈ I

```

---

## 📐 Step 5: Boundary Behavior

### At $t = 0$

```math
\bar{\alpha}_0 = 1 \implies x_0 = 1 \cdot x_0 + 0 \cdot \epsilon = x_0 \quad \checkmark

```

### At $t = T$ (Large T)

```math
\bar{\alpha}_T = \prod_{t=1}^{T}(1-\beta_t) \approx 0 \quad \text{(for appropriate } \beta_t \text{)}
\implies x_T \approx 0 \cdot x_0 + 1 \cdot \epsilon = \epsilon \sim \mathcal{N}(0, I) \quad \checkmark

```

### Proof of $\bar{\alpha}\_T \to 0$

For $\beta\_t = \beta$ constant:

```math
\bar{\alpha}_T = (1-\beta)^T
\lim_{T \to \infty} (1-\beta)^T = 0 \quad \text{for } \beta \in (0,1)

```

### Visual

```
+------------------------------------------------------------+

|  √ᾱ_t (signal weight)                                     |
|  1.0 +■■■■■                                                |
|  0.8 +     ■■■■                                            |
|  0.6 +         ■■■                                         |
|  0.4 +            ■■■                                      |
|  0.2 +               ■■■■                                  |
|  0.0 +                   ■■■■■■■■■                         |
|      +--+---+---+---+---+---+---+---+---+--▶ t            |
|         0       T/4     T/2     3T/4    T                  |
+------------------------------------------------------------+

```

---

## 📐 Step 6: Signal-to-Noise Ratio

### Definition

```math
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}

```

### Interpretation

| $t$ | $\bar{\alpha}\_t$ | SNR | Description |
|:---:|:----------------:|:---:|-------------|
| 0 | 1.0 | ∞ | Pure signal |
| T/4 | ~0.8 | ~4 | Mostly signal |
| T/2 | ~0.5 | ~1 | Equal mix |
| 3T/4 | ~0.2 | ~0.25 | Mostly noise |
| T | ~0 | ~0 | Pure noise |

### Log-SNR

Often use log-SNR for numerical stability:

```math
\log \text{SNR}(t) = \log \bar{\alpha}_t - \log(1 - \bar{\alpha}_t)

```

---

## 🔑 Summary

<div align="center">

| Property | Formula |
|----------|---------|
| **Single step** | $x\_t = \sqrt{\alpha\_t} x\_{t-1} + \sqrt{\beta\_t} \epsilon\_t$ |
| **Marginal** | $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t} x\_0, (1-\bar{\alpha}\_t)I)$ |
| **Sampling** | $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$ |
| **Boundary** | $x\_0 = \text{data}$, $x\_T \approx \mathcal{N}(0, I)$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Gaussian Transition](../03%20Gaussian%20Transition%20Derivation/) · **Page 4 of 10** · [Next: Noise Schedule →](../05%20Noise%20Schedule/)

</div>
