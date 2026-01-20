<div align="center">

# 📉 Marginal Distributions

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=Skip+Steps%2C+Train+Efficiently;The+Key+to+Fast+Training" alt="Typing SVG" />

[← Noise Schedule](../05%20Noise%20Schedule/) · **Page 6 of 10** · [Next: Reverse Process →](../07%20Reverse%20Process/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

The **marginal distribution** $q(x\_t \mid x\_0)$ lets us sample noisy data at **any timestep directly** — no need to simulate the entire forward chain!

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/06%20Marginal%20Distributions/marginal_distributions.svg" alt="Marginal Distributions Diagram" width="100%">
</p>

---

## 📐 Step 1: Main Result

### The Marginal Distribution

```math
\boxed{q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)}
```

### Equivalent Sampling

```math
\boxed{x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon}, \quad \epsilon \sim \mathcal{N}(0, I)
```

### Why This Matters

| Without Marginal | With Marginal |
|:----------------:|:-------------:|
| $x\_0 \to x\_1 \to x\_2 \to \cdots \to x\_t$ | $x\_0 \to x\_t$ directly! |
| $O(t)$ operations | $O(1)$ operation |
| Sequential | Parallelizable |

---

## 📐 Step 2: Complete Derivation

### Goal

Derive $q(x\_t \mid x\_0)$ from the chain of Gaussian transitions.

### Step 1: Single Transition

```math
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1
```

### Step 2: Two Transitions

```math
x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2
= \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1) + \sqrt{1-\alpha_2} \epsilon_2
= \sqrt{\alpha_1\alpha_2} x_0 + \sqrt{\alpha_2(1-\alpha_1)} \epsilon_1 + \sqrt{1-\alpha_2} \epsilon_2
```

### Step 3: Combine Noise Terms
Let $\tilde{\epsilon} = \sqrt{\alpha\_2(1-\alpha\_1)} \epsilon\_1 + \sqrt{1-\alpha\_2} \epsilon\_2$

**Variance of $\tilde{\epsilon}$:**

```math
\text{Var}[\tilde{\epsilon}] = \alpha_2(1-\alpha_1) + (1-\alpha_2) = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1\alpha_2
```

### Step 4: Standardize

```math
\tilde{\epsilon} \sim \mathcal{N}(0, (1-\alpha_1\alpha_2)I)
```

So: $\tilde{\epsilon} = \sqrt{1-\alpha\_1\alpha\_2} \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$

### Step 5: General Pattern
Define $\bar{\alpha}\_t = \prod\_{s=1}^{t} \alpha\_s$

```math
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
```

### Step 6: Verify by Induction

**Base:** $t=1$: $x\_1 = \sqrt{\alpha\_1}x\_0 + \sqrt{1-\alpha\_1}\epsilon$ ✓

**Induction:** Assume true for $t-1$. Then:

```math
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t
= \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon') + \sqrt{1-\alpha_t}\epsilon_t
```

Combining noise (variance = $\alpha\_t(1-\bar{\alpha}\_{t-1}) + 1-\alpha\_t = 1-\bar{\alpha}\_t$):

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \quad \checkmark
```

---

## 📐 Step 3: Posterior Distribution

### Goal

Find $q(x\_{t-1} \mid x\_t, x\_0)$ — the "reverse" step given both endpoints.

### Why We Need This

This is the **target distribution** our neural network must match!

### Bayes' Theorem Application

```math
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
```

Since $q(x\_t \mid x\_{t-1}, x\_0) = q(x\_t \mid x\_{t-1})$ (Markov property):

```math
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
```

### All Three Terms are Gaussian!

1. $q(x\_t \mid x\_{t-1}) = \mathcal{N}(\sqrt{\alpha\_t}x\_{t-1}, \beta\_t I)$
2. $q(x\_{t-1} \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_{t-1}}x\_0, (1-\bar{\alpha}\_{t-1})I)$
3. $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}x\_0, (1-\bar{\alpha}\_t)I)$

### Result

Product of Gaussians is Gaussian:

```math
\boxed{q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t, \tilde{\beta}_t I)}
```

### Posterior Mean

```math
\boxed{\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t}
```

### Posterior Variance

```math
\boxed{\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}}
```

---

## 📐 Step 4: Deriving Posterior Parameters

### Method: Complete the Square

For Gaussians, the posterior is found by completing the square in the exponent.

### Log of Product

```math
\log q(x_{t-1} \mid x_t, x_0) \propto \log q(x_t \mid x_{t-1}) + \log q(x_{t-1} \mid x_0)
```

### Expand Quadratics

```math
\propto -\frac{1}{2\beta_t}\|x_t - \sqrt{\alpha_t}x_{t-1}\|^2 - \frac{1}{2(1-\bar{\alpha}_{t-1})}\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\|^2
```

### Collect Terms in $x\_{t-1}$

After expansion, group terms quadratic in $x\_{t-1}$:

```math
\propto -\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}x_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}\right)x_{t-1}\right]
```

### Read Off Parameters

**Precision (inverse variance):**

```math
\tilde{\beta}_t^{-1} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}
```

**Mean times precision:**

```math
\tilde{\beta}_t^{-1} \tilde{\mu}_t = \frac{\sqrt{\alpha_t}x_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}
```

### Final Simplification

After algebra (using $\beta\_t = 1-\alpha\_t$ and $1-\bar{\alpha}\_t = 1-\alpha\_t\bar{\alpha}\_{t-1}$):

```math
\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
```

---

## 📐 Step 5: Alternative Mean Formulation

### Express in Terms of $\epsilon$

Since $x\_t = \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon$, we can write:

```math
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
```

### Substitute into $\tilde{\mu}\_t$

```math
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
```

### After Simplification

```math
\boxed{\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)}
```

### Why This Matters

| Form | Useful When |
|------|-------------|
| $\tilde{\mu}\_t(x\_0, x\_t)$ | You know $x\_0$ |
| $\tilde{\mu}\_t(x\_t, \epsilon)$ | You predict $\epsilon$ (training!) |

---

## 📐 Step 6: Score Function Connection

### Definition

The **score function** is the gradient of log-density:

```math
s_t(x) = \nabla_x \log q(x_t)
```

### For Our Marginal

```math
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
\log q(x_t \mid x_0) = -\frac{\|x_t - \sqrt{\bar{\alpha}_t}x_0\|^2}{2(1-\bar{\alpha}_t)} + C
```

### Take Gradient

```math
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}
```

### Use $x\_t = \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon$

```math
= -\frac{\sqrt{1-\bar{\alpha}_t}\epsilon}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
```

### Key Identity

```math
\boxed{\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}}
```

### Implication

Predicting $\epsilon$ is equivalent to estimating the score! This is why **score matching** and **denoising** are connected.

---

## 🔑 Summary

<div align="center">

| Distribution | Formula |
|--------------|---------|
| **Marginal** | $q(x\_t \mid x\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t}x\_0, (1-\bar{\alpha}\_t)I)$ |
| **Posterior** | $q(x\_{t-1} \mid x\_t, x\_0) = \mathcal{N}(\tilde{\mu}\_t, \tilde{\beta}\_t I)$ |
| **Score** | $\nabla \log q = -\epsilon / \sqrt{1-\bar{\alpha}\_t}$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Noise Schedule](../05%20Noise%20Schedule/) · **Page 6 of 10** · [Next: Reverse Process →](../07%20Reverse%20Process/)

</div>
