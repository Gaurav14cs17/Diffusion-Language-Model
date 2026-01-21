<div align="center">

# 📐 Probability Assumptions

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=Markov+%2B+Gaussian+%3D+Tractability;The+Foundation+of+Diffusion" alt="Typing SVG" />

[← Data Space](../01%20Data%20Space/) · **Page 2 of 10** · [Next: Gaussian Transition →](../03%20Gaussian%20Transition%20Derivation/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

Diffusion models rely on **three key assumptions** that make them mathematically tractable while remaining expressive.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/02%20Probability%20Assumptions/probability_assumptions.svg" alt="Probability Assumptions Diagram" width="100%">
</p>

---

## 📐 Step 1: The Markov Assumption

### Statement

The forward process forms a **Markov chain**: each step depends only on the previous state.

### Formal Definition

$$q(x_t \mid x_0, x_1, ..., x_{t-1}) = q(x_t \mid x_{t-1})$$

### Why This Helps

**Joint distribution factorizes nicely:**

$$q(x_{0:T}) = q(x_0) \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$

### Proof of Factorization

1. Start with chain rule:

$$q(x_{0:T}) = q(x_0) q(x_1 \mid x_0) q(x_2 \mid x_0, x_1) \cdots$$

2. Apply Markov property:

$$q(x_2 \mid x_0, x_1) = q(x_2 \mid x_1)$$

3. Result:

$$\boxed{q(x_{0:T}) = q(x_0) \prod_{t=1}^{T} q(x_t \mid x_{t-1})}$$

---

## 📐 Step 2: The Gaussian Assumption

### Statement

Each transition is a **Gaussian perturbation**:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)$$

### Why Gaussian?

| Property | Benefit |
|----------|---------|
| **Closed under addition** | Sum of Gaussians is Gaussian |
| **Closed under marginalization** | Marginals are Gaussian |
| **KL divergence tractable** | Closed-form training loss |
| **Reparameterizable** | Enables backprop |

### Equivalent Forms

$$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

### Proof of Equivalence

Let $x_{t-1}$ be fixed, $\epsilon_t \sim \mathcal{N}(0, I)$:

1. **Mean**: $\mathbb{E}[x_t] = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \cdot 0 = \sqrt{\alpha_t} x_{t-1}$

2. **Variance**: $\text{Var}[x_t] = \beta_t \cdot I$

3. **Distribution**: $x_t \sim \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$ ✓

---

## 📐 Step 3: Variance Preservation ($\alpha_t + \beta_t = 1$)

### Statement

We require:

$$\alpha_t + \beta_t = 1 \quad \text{for all } t$$

### Why This Constraint?

**Goal**: Keep variance bounded throughout diffusion.

### Proof by Induction

**Base case**: Assume $\text{Var}[x_0] = I$

**Inductive step**: If $\text{Var}[x_{t-1}] = I$, then:

$$\text{Var}[x_t] = \alpha_t \cdot \text{Var}[x_{t-1}] + \beta_t \cdot I
= \alpha_t \cdot I + \beta_t \cdot I = (\alpha_t + \beta_t) I = I$$

**Conclusion**: Variance stays constant at $I$ for all $t$! ✓

### Visual Intuition

```
+---------------------------------------------+
|  α_t = "how much signal to keep"            |
|  β_t = "how much noise to add"              |
|                                             |
|  α_t + β_t = 1 ensures "total = 1"          |
|  → Variance doesn't explode or vanish!      |
+---------------------------------------------+

```

---

## 📐 Step 4: Marginal Distribution Derivation

### Goal

Find $q(x_t \mid x_0)$ without computing intermediate steps.

### Recursive Expansion

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t
x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}} \epsilon_{t-1}$$

### Substituting

$$x_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}} \epsilon_{t-1}) + \sqrt{\beta_t} \epsilon_t
= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t \beta_{t-1}} \epsilon_{t-1} + \sqrt{\beta_t} \epsilon_t$$

### After Full Recursion

Define $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:

$$\boxed{x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon}$$

where $\epsilon \sim \mathcal{N}(0, I)$.

---

## 📐 Step 5: Posterior Distribution

### Goal

Compute $q(x_{t-1} \mid x_t, x_0)$ — needed for training!

### Using Bayes' Rule

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

### All Terms are Gaussian!

Since Gaussian × Gaussian = Gaussian:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$$

### Closed-Form Parameters

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}$$

---

## 🔑 Summary

<div align="center">

| Assumption | Mathematical Form | Purpose |
|------------|------------------|---------|
| **Markov** | $q(x_t \mid x_{0:t-1}) = q(x_t \mid x_{t-1})$ | Factorization |
| **Gaussian** | $q(x_t \mid x_{t-1}) = \mathcal{N}(\cdot)$ | Tractable KL |
| **Variance Preserving** | $\alpha_t + \beta_t = 1$ | Stability |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Data Space](../01%20Data%20Space/) · **Page 2 of 10** · [Next: Gaussian Transition →](../03%20Gaussian%20Transition%20Derivation/)

</div>
