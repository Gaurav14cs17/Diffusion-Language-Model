<div align="center">

# ⚙️ Parameterization

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&duration=3000&pause=1000&color=818CF8&center=true&vCenter=true&width=500&lines=ε%2C+x₀%2C+or+v+Prediction;All+Roads+Lead+to+Rome" alt="Typing SVG" />

[← Training Objective](../08%20Training%20Objective/) · **Page 9 of 10** · [Next: Sampling →](../10%20Sampling/)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

</div>

## 🎯 Overview

There are **three equivalent ways** to parameterize what the diffusion model predicts. Each has different numerical properties!

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/00_Diffusion_Theory/09%20Parameterization/parameterization.svg" alt="Parameterization Diagram" width="100%">
</p>

---

## 📐 Step 1: The Three Predictions

| Prediction | Symbol | The Model Outputs |
|:----------:|:------:|-------------------|
| **Noise** | $\epsilon\_\theta$ | The noise that was added |
| **Data** | $x\_{0,\theta}$ | The clean data directly |
| **Velocity** | $v\_\theta$ | A combination of both |

### Relationship

Given $x\_t = \sqrt{\bar{\alpha}\_t}x\_0 + \sqrt{1-\bar{\alpha}\_t}\epsilon$:

$$
\boxed{v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0}
$$

---

## 📐 Step 2: ε-Prediction (DDPM)

### What We Predict

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

### Loss Function

$$
\mathcal{L}_\epsilon = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

### Recovery Formula

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
$$

| ✅ Pros | ❌ Cons |
|---------|---------|
| Most studied | Numerically unstable at $t \to 0$ |
| Works well for images | Division by small $\sqrt{\bar{\alpha}\_t}$ |

---

## 📐 Step 3: x₀-Prediction

### What We Predict

$$
x_{0,\theta}(x_t, t) \approx x_0
$$

### Loss Function

$$
\mathcal{L}_{x_0} = \mathbb{E}_{t,x_0,\epsilon}\left[\|x_0 - x_{0,\theta}(x_t, t)\|^2\right]
$$

### Recovery Formula

$$
\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t}x_{0,\theta}}{\sqrt{1-\bar{\alpha}_t}}
$$

| ✅ Pros | ❌ Cons |
|---------|---------|
| Direct interpretation | Unstable at $t \to T$ |
| Good for low noise | High variance at large $t$ |

---

## 📐 Step 4: v-Prediction

### Definition

$$
v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0
$$

### Loss Function

$$
\mathcal{L}_v = \mathbb{E}_{t,x_0,\epsilon}\left[\|v_t - v_\theta(x_t, t)\|^2\right]
$$

### Recovery Formulas

$$
x_0 = \sqrt{\bar{\alpha}_t}x_t - \sqrt{1-\bar{\alpha}_t}v_\theta
\epsilon = \sqrt{1-\bar{\alpha}_t}x_t + \sqrt{\bar{\alpha}_t}v_\theta
$$

| ✅ Pros | ❌ Cons |
|---------|---------|
| Stable for all $t$ | Less intuitive |
| Best for distillation | Newer, less studied |

---

## 📐 Step 5: Equivalence Proof

All three losses are equivalent up to scaling.

### Conversion Table

| Predict | Get $x\_0$ | Get $\epsilon$ |
|---------|-----------|----------------|
| $\epsilon\_\theta$ | $\frac{x\_t - \sqrt{1-\bar{\alpha}\_t}\epsilon\_\theta}{\sqrt{\bar{\alpha}\_t}}$ | $\epsilon\_\theta$ |
| $x\_{0,\theta}$ | $x\_{0,\theta}$ | $\frac{x\_t - \sqrt{\bar{\alpha}\_t}x\_{0,\theta}}{\sqrt{1-\bar{\alpha}\_t}}$ |
| $v\_\theta$ | $\sqrt{\bar{\alpha}\_t}x\_t - \sqrt{1-\bar{\alpha}\_t}v\_\theta$ | $\sqrt{1-\bar{\alpha}\_t}x\_t + \sqrt{\bar{\alpha}\_t}v\_\theta$ |

---

## 📐 Step 6: When to Use Each

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Standard image generation | $\epsilon$-prediction | Most tested |
| High resolution | $v$-prediction | Better numerics |
| Progressive distillation | $v$-prediction | Stable |
| Conditional generation | $x\_0$-prediction | Direct control |

---

## 🔑 Summary

<div align="center">

| Parameterization | Best For |
|:----------------:|----------|
| **ε-prediction** | Standard generation |
| **x₀-prediction** | Editing, low noise |
| **v-prediction** | Distillation, stability |

> 💡 All three are mathematically equivalent — the choice affects **numerical stability**!

</div>

---

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">

[← Training Objective](../08%20Training%20Objective/) · **Page 9 of 10** · [Next: Sampling →](../10%20Sampling/)

</div>
