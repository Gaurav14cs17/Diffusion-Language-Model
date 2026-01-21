<div align="center">

# 🔢 Gaussian Transition Derivation

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=The+Heart+of+Diffusion;Step-by-Step+Mathematical+Proof" alt="Typing SVG" />

[← Probability Assumptions](../02%20Probability%20Assumptions/) · **Page 3 of 10** · [Next: Forward Process →](../04%20Forward%20Process/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Goal

Derive and prove the Gaussian transition formula:

$$\boxed{x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t}$$

where $\epsilon_t \sim \mathcal{N}(0, I)$ and $\alpha_t + \beta_t = 1$.

---

## 📐 Step 1: Fundamental Assumption

### Statement

We **assume** each forward step is a Gaussian transition:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \mu_t, \sigma_t^2 I)$$

### Design Choice

We choose:

- **Mean**: $\mu_t = \sqrt{\alpha_t} \cdot x_{t-1}$ (scaled previous state)

- **Variance**: $\sigma_t^2 = \beta_t$ (added noise)

### Why This Design?

| Choice | Reason |
|--------|--------|
| Scale by $\sqrt{\alpha_t}$ | Preserve signal power |
| Add variance $\beta_t$ | Gradual noise injection |
| $\alpha_t + \beta_t = 1$ | Variance preservation |

### Resulting Transition

$$\boxed{q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)}$$

---

## 📐 Step 2: Reparameterization Trick

### The Problem

Sampling $x_t \sim \mathcal{N}(\mu, \sigma^2 I)$ is not differentiable!

### The Solution

Any Gaussian sample can be written as:

$$x = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Proof

Let $\epsilon \sim \mathcal{N}(0, I)$. Define $x = \mu + \sigma \epsilon$.

**Mean:**

$$\mathbb{E}[x] = \mathbb{E}[\mu + \sigma \epsilon] = \mu + \sigma \cdot 0 = \mu \quad \checkmark$$

**Variance:**

$$\text{Var}[x] = \text{Var}[\sigma \epsilon] = \sigma^2 \text{Var}[\epsilon] = \sigma^2 I \quad \checkmark$$

**Distribution:**

$$x \sim \mathcal{N}(\mu, \sigma^2 I) \quad \checkmark$$

### Why This Matters

```
Before: x ~ N(μ, σ²)      → Cannot backprop through sampling
After:  x = μ + σε        → Gradients flow through μ and σ!

```

---

## 📐 Step 3: Apply to Diffusion

### Setup

From our transition density:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)$$

### Apply Reparameterization

- Mean: $\mu = \sqrt{\alpha_t} x_{t-1}$

- Std: $\sigma = \sqrt{\beta_t}$

- Noise: $\epsilon_t \sim \mathcal{N}(0, I)$

### Result

$$\boxed{x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t}$$

### Intuition

```
+--------------------------------------------------------------+
|                                                              |
|   x_t  =  √α_t · x_{t-1}  +  √β_t · ε_t                     |
|           ------------       ----------                      |
|           "keep signal"      "add noise"                     |
|                                                              |
+--------------------------------------------------------------+

```

---

## 📐 Step 4: Why Square Roots?

### The Question

Why $\sqrt{\alpha_t}$ and $\sqrt{\beta_t}$, not $\alpha_t$ and $\beta_t$?

### Answer: Variance Additivity

When adding independent random variables:

$$\text{Var}[aX + bY] = a^2 \text{Var}[X] + b^2 \text{Var}[Y]$$

### Applied to Our Formula

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t
\text{Var}[x_t] = (\sqrt{\alpha_t})^2 \text{Var}[x_{t-1}] + (\sqrt{\beta_t})^2 \text{Var}[\epsilon_t]
= \alpha_t \cdot \text{Var}[x_{t-1}] + \beta_t \cdot I$$

### If We Used α and β Directly

$$\text{Var}[x_t] = \alpha_t^2 \cdot \text{Var}[x_{t-1}] + \beta_t^2 \cdot I$$

This would **not** preserve variance! ❌

---

## 📐 Step 5: Variance Preservation Proof

### Goal

Prove that if $\text{Var}[x_0] = I$ and $\alpha_t + \beta_t = 1$, then $\text{Var}[x_t] = I$ for all $t$.

### Proof by Strong Induction

**Base Case ($t = 0$):**

$$\text{Var}[x_0] = I \quad \checkmark$$

**Inductive Hypothesis:**
Assume $\text{Var}[x_{t-1}] = I$.

**Inductive Step:**

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t
\text{Var}[x_t] = \alpha_t \cdot \text{Var}[x_{t-1}] + \beta_t \cdot \text{Var}[\epsilon_t]
= \alpha_t \cdot I + \beta_t \cdot I
= (\alpha_t + \beta_t) \cdot I
= 1 \cdot I = I \quad \checkmark$$

### Conclusion

$$\boxed{\text{Var}[x_t] = I \text{ for all } t \in \{0, 1, ..., T\}}$$

### Visual Summary

```
t:    0      1      2      3    ...    T
     -+-----+-----+-----+-------------+-
Var:  I      I      I      I           I
     -+-----+-----+-----+-------------+-
     
✅ Variance is CONSTANT throughout the diffusion process!

```

---

## 📐 Bonus: Alternative Derivation via Score

### Langevin Dynamics Connection

The diffusion process can also be derived from:

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW_t$$

### Discretizing

$$x_t - x_{t-1} = -\frac{1}{2}\beta_t x_{t-1} + \sqrt{\beta_t} \epsilon_t
x_t = (1 - \frac{1}{2}\beta_t) x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

### With $\alpha_t = 1 - \beta_t$

For small $\beta_t$: $\sqrt{\alpha_t} \approx 1 - \frac{1}{2}\beta_t$

This recovers our formula! ✓

---

## 🔑 Summary

<div align="center">

| Step | Result |
|------|--------|
| **1. Assumption** | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, \beta_t I)$ |
| **2. Reparameterize** | $x = \mu + \sigma \epsilon$ |
| **3. Combine** | $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t$ |
| **4. Square roots** | Needed for variance additivity |
| **5. Preservation** | $\alpha_t + \beta_t = 1 \Rightarrow \text{Var}[x_t] = I$ |

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Probability Assumptions](../02%20Probability%20Assumptions/) · **Page 3 of 10** · [Next: Forward Process →](../04%20Forward%20Process/)

</div>
