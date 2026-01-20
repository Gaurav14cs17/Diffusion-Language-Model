# The Complete Mathematics of Diffusion Language Models: A Deep Dive into Dream 7B

*A comprehensive guide to understanding the mathematical foundations of discrete diffusion for text generation*

---

## Table of Contents

1. [Introduction: Why Diffusion for Language?](#1-introduction-why-diffusion-for-language)
2. [The Forward Diffusion Process](#2-the-forward-diffusion-process)
3. [The Reverse Denoising Process](#3-the-reverse-denoising-process)
4. [Training Objective: ELBO Derivation](#4-training-objective-elbo-derivation)
5. [Score Matching and Loss Functions](#5-score-matching-and-loss-functions)
6. [Sampling and Generation](#6-sampling-and-generation)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction: Why Diffusion for Language?

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/01_overview_architecture.svg" alt="Overview Architecture" width="100%">
</p>

The field of large language models (LLMs) has been dominated by **autoregressive models** like GPT, LLaMA, and Claude. These models generate text one token at a time, left-to-right, using the factorization:

```math
P(x) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, \ldots, x_{i-1})
```

While incredibly successful, this approach has fundamental limitations:

### Limitations of Autoregressive Models

1. **Sequential Generation**: Each token must wait for all previous tokens, making generation inherently slow.

2. **Unidirectional Context**: When generating token $x\_i$, the model can only see tokens $x\_1, \ldots, x\_{i-1}$, not future tokens.

3. **No Revision**: Once a token is generated, it cannot be revised based on later context.

4. **Linear Time Complexity**: Generating $n$ tokens requires $O(n)$ forward passes.

### Enter Diffusion Language Models

**Dream 7B** and similar diffusion-based language models offer a fundamentally different approach. Instead of generating sequentially, they:

1. Start with a fully "noisy" (masked) sequence
2. Iteratively denoise to reveal the final text
3. Generate all positions in parallel
4. Allow bidirectional context at every step

The key mathematical insight is adapting continuous diffusion (used for images) to the **discrete domain** of text tokens.

### The Core Framework: Masked Diffusion

For text, we use **absorbing-state diffusion** where:
- The "noise" is a special `[MASK]` token
- Forward process: gradually replace tokens with `[MASK]`
- Reverse process: learn to predict original tokens from masked context

This is formalized as a **Markov chain** over discrete states, which we'll explore in detail.

---

## 2. The Forward Diffusion Process

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/02_forward_diffusion_process.svg" alt="Forward Diffusion Process" width="100%">
</p>

The forward diffusion process defines how we **corrupt** clean data $x\_0$ into progressively noisier versions $x\_1, x\_2, \ldots, x\_T$, until we reach a fully masked sequence $x\_T$.

### 2.1 Markov Chain Formulation

The forward process is defined as a Markov chain:

```math
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
```

Each transition is defined by a **transition matrix** $Q\_t$:

```math
q(x_t \mid x_{t-1}) = \text{Cat}(x_t; x_{t-1} \cdot Q_t)
```

where $\text{Cat}$ denotes a categorical distribution.

### 2.2 Absorbing State Transition Matrix

For masked diffusion with `[MASK]` as the absorbing state, the transition matrix has a specific structure:

```math
Q_t[i, j] = (1 - \beta_t) \cdot \delta_{ij} + \beta_t \cdot \delta_{j, [M]}
```

Where:
- $\beta\_t$ is the **masking probability** at step $t$
- $\delta\_{ij}$ is the Kronecker delta (1 if $i=j$, 0 otherwise)
- $[M]$ is the index of the `[MASK]` token

**Interpretation of each entry:**

| Entry | Meaning |
|-------|---------|
| Diagonal $(1 - \beta\_t)$ | Token stays the same |
| Last column $(\beta\_t)$ | Token becomes `[MASK]` |
| Last row (absorbing) | Once masked, stays masked (probability 1) |

### 2.3 Marginal Distribution: The Key Theorem

**Theorem (Closed-Form Marginal):** The marginal distribution at time $t$ given $x\_0$ is:

```math
q(x_t \mid x_0) = \text{Cat}(x_t; x_0 \cdot \bar{Q}_t)
```

where $\bar{Q}\_t = Q\_1 \cdot Q\_2 \cdots Q\_t$ is the cumulative transition matrix.

**Proof:**

*Step 1:* By the chain rule of probability:
```math
q(x_t \mid x_0) = \sum_{x_1, \ldots, x_{t-1}} q(x_1 \mid x_0) \cdot q(x_2 \mid x_1) \cdots q(x_t \mid x_{t-1})
```

*Step 2:* Using the matrix multiplication property:
```math
q(x_t \mid x_0) = (x_0 \cdot Q_1) \cdot Q_2 \cdots Q_t = x_0 \cdot (Q_1 \cdot Q_2 \cdots Q_t) = x_0 \cdot \bar{Q}_t
```

*Step 3:* For absorbing-state diffusion, $\bar{Q}\_t$ simplifies to:
```math
\bar{Q}_t[i, j] = \alpha_t \cdot \delta_{ij} + (1 - \alpha_t) \cdot \delta_{j, [M]}
```

where $\alpha\_t = \prod\_{s=1}^{t} (1 - \beta\_s)$ is the **cumulative survival probability**.

### 2.4 Practical Interpretation

For any token at position $i$:

```math
P(x_t^i = x_0^i) = \alpha_t \quad \text{(token survives unmasked)}
P(x_t^i = [M]) = 1 - \alpha_t \quad \text{(token is masked)}
```

This gives us a simple sampling procedure:
1. Sample $t \sim \text{Uniform}(0, 1)$ in continuous time
2. Compute $\alpha\_t = 1 - t$ (linear schedule)
3. Independently mask each token with probability $1 - \alpha\_t$

### 2.5 Noise Schedules

The choice of $\beta\_t$ (or equivalently $\alpha\_t$) significantly impacts training:

| Schedule | Formula | Properties |
|----------|---------|------------|
| Linear | $\beta\_t = \beta\_1 + \frac{t-1}{T-1}(\beta\_T - \beta\_1)$ | Simple, but can be suboptimal |
| Cosine | $\alpha\_t = \cos^2\left(\frac{\pi t}{2T} + \text{offset}\right)$ | Smoother corruption |
| MDLM/Dream | $\alpha\_t = 1 - t$ for $t \in [0, 1]$ | Simple continuous-time formulation |

---

## 3. The Reverse Denoising Process

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/03_reverse_denoising_process.svg" alt="Reverse Denoising Process" width="100%">
</p>

The reverse process is where the magic happens: we learn to **undo** the corruption, generating text from noise.

### 3.1 The Goal

We want to learn $p\_\theta(x\_{t-1} \mid x\_t)$ that approximates the true reverse:

```math
p(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0) \, p(x_0 \mid x_t) \, dx_0
```

This integral is intractable because it requires marginalizing over all possible clean sequences $x\_0$.

### 3.2 The Posterior Distribution

**Key Insight:** Given both $x\_t$ and $x\_0$, the posterior $q(x\_{t-1} \mid x\_t, x\_0)$ is **tractable**!

**Derivation using Bayes' Rule:**

```math
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
```

By the Markov property: $q(x\_t \mid x\_{t-1}, x\_0) = q(x\_t \mid x\_{t-1})$

Substituting the known distributions:
- $q(x\_t \mid x\_{t-1}) = \text{Cat}(x\_t; x\_{t-1} \cdot Q\_t)$
- $q(x\_{t-1} \mid x\_0) = \text{Cat}(x\_{t-1}; x\_0 \cdot \bar{Q}\_{t-1})$
- $q(x\_t \mid x\_0) = \text{Cat}(x\_t; x\_0 \cdot \bar{Q}\_t)$

### 3.3 Posterior for Absorbing-State Diffusion

For masked diffusion, the posterior has a simple closed form:

**Case 1: $x\_t$ is NOT `[MASK]`** (meaning $x\_t = x\_0$)
```math
q(x_{t-1} \mid x_t, x_0) = \delta(x_{t-1}, x_t)
```

The token hasn't been masked yet, so it must stay as the original.

**Case 2: $x\_t$ IS `[MASK]`**
```math
q(x_{t-1} \mid x_t = [M], x_0) = \theta_t \cdot \delta(x_{t-1}, x_0) + (1 - \theta_t) \cdot \delta(x_{t-1}, [M])
```

With probability $\theta\_t$, we "unmask" to the original token; otherwise, we stay masked.

**The unmasking probability:**
```math
\theta_t = \frac{\alpha_{t-1} - \alpha_t}{1 - \alpha_t} = \frac{\beta_t \cdot \alpha_{t-1}}{1 - \alpha_t}
```

### 3.4 Neural Network Parameterization

We use a neural network $f\_\theta$ to approximate the reverse process.

**$x\_0$-Parameterization (used in Dream):**

Instead of directly predicting $x\_{t-1}$, we predict the clean data $x\_0$:

```math
\hat{x}_0 = f_\theta(x_t, t)
```

The network outputs logits over the vocabulary for each position:
```math
p_\theta(x_0^i \mid x_t) = \text{softmax}(f_\theta(x_t, t))^i
```

The reverse transition is then computed as:
```math
p_\theta(x_{t-1} \mid x_t) = \sum_{x_0} q(x_{t-1} \mid x_t, x_0) \cdot p_\theta(x_0 \mid x_t)
```

### 3.5 Architecture Differences from Autoregressive Models

| Feature | Autoregressive (GPT) | Diffusion (Dream) |
|---------|---------------------|-------------------|
| Attention | Causal (lower triangular mask) | Bidirectional (full attention) |
| Time input | None (position only) | Time embedding $E(t)$ added |
| Context | Only past tokens | All tokens (including future) |
| Generation | One token per forward pass | All tokens refined per step |

---

## 4. Training Objective: ELBO Derivation

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/04_elbo_training_objective.svg" alt="ELBO Training Objective" width="100%">
</p>

The training objective is derived from the **Evidence Lower Bound (ELBO)**, a fundamental concept in variational inference.

### 4.1 Maximum Likelihood Objective

We want to find parameters $\theta$ that maximize the log-likelihood of observed data:

```math
\max_\theta \mathbb{E}_{x_0 \sim p_{\text{data}}}[\log p_\theta(x_0)]
```

**The Problem:** Computing $p\_\theta(x\_0)$ requires integrating over all possible latent paths:

```math
p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}
```

This is intractable for high-dimensional data.

### 4.2 ELBO Derivation: Complete Proof

**Step 1: Introduce variational distribution**

We introduce the forward process $q(x\_{1:T} \mid x\_0)$ as our variational distribution:

```math
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T}) \, dx_{1:T} = \log \int p_\theta(x_{0:T}) \cdot \frac{q(x_{1:T} \mid x_0)}{q(x_{1:T} \mid x_0)} \, dx_{1:T}
```

**Step 2: Apply Jensen's inequality**

Since $\log$ is concave:

```math
\log p_\theta(x_0) = \log \mathbb{E}_q\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]
```

This gives us the **ELBO**:
```math
\log p_\theta(x_0) \geq \text{ELBO} = \mathbb{E}_q\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)\right]
```

**Step 3: Factor the joint distributions**

```math
p_\theta(x_{0:T}) = p(x_T) \cdot \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
```

### 4.3 ELBO Decomposition into KL Divergences

The ELBO decomposes into three interpretable terms:

```math
-\text{ELBO} = L_T + \sum_{t=2}^{T} L_{t-1} + L_0
```

| Term | Formula | Interpretation |
|------|---------|----------------|
| $L\_T$ | $D\_{KL}(q(x\_T \mid x\_0) \| p(x\_T))$ | Prior matching loss |
| $L\_{t-1}$ | $\mathbb{E}\_q[D\_{KL}(q(x\_{t-1} \mid x\_t, x\_0) \| p\_\theta(x\_{t-1} \mid x\_t))]$ | Denoising loss |
| $L\_0$ | $-\mathbb{E}\_q[\log p\_\theta(x\_0 \mid x\_1)]$ | Reconstruction loss |

**Key Insight:** For discrete diffusion, KL divergence simplifies to cross-entropy:

```math
D_{KL}(q \| p_\theta) = -\sum_x q(x) \log p_\theta(x) + \text{const} = H(q, p_\theta) - H(q)
```

### 4.4 Simplified MDLM Training Objective

For masked (absorbing-state) diffusion, the ELBO simplifies beautifully:

```math
\boxed{L_{\text{MDLM}} = \mathbb{E}_{t \sim U(0,1), x_0, x_t \sim q(\cdot \mid x_0)}\left[-\sum_{i: x_t^i = [M]} \log p_\theta(x_0^i \mid x_t)\right]}
```

**In plain English:**
1. Sample random time $t$ uniformly from $[0, 1]$
2. Sample clean data $x\_0$ from training set
3. Corrupt $x\_0$ to get $x\_t$ by masking $\sim(1-\alpha\_t)$ fraction of tokens
4. **Train to predict the original tokens at masked positions only**

This is remarkably similar to **BERT's masked language modeling (MLM)** objective, but with:
- Variable masking rate depending on $t$
- Time conditioning in the model

### 4.5 Continuous-Time ELBO

As $T \to \infty$, the discrete ELBO becomes an integral:

```math
L = \int_0^1 \mathbb{E}_{x_0, x_t}\left[-\lambda(t) \cdot \log p_\theta(x_0 \mid x_t)\right] dt
```

where $\lambda(t) = \frac{d}{dt}(1 - \alpha\_t) = -\alpha'\_t$ is the weighting function.

In practice, we use Monte Carlo estimation with $t \sim \text{Uniform}(0, 1)$.

---

## 5. Score Matching and Loss Functions

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/05_score_matching_loss.svg" alt="Score Matching Loss" width="100%">
</p>

### 5.1 What is the Score Function?

In continuous diffusion (for images), the **score function** is central:

```math
s(x, t) = \nabla_x \log p(x \mid t)
```

This is the gradient of the log-density with respect to the data $x$.

**The Problem for Text:** Tokens are discrete — there's no gradient to compute!

### 5.2 The Discrete "Score"

For discrete diffusion, we instead work with **probability ratios**:

```math
\text{score}(x_t, t) = \frac{p(x_0 \mid x_t)}{p([M] \mid x_t)}
```

**Interpretation:** How much more likely is a real token compared to `[MASK]`?

In practice, we work with log-probabilities (logits) directly:
```math
p_\theta(x_0 \mid x_t) = \text{softmax}(f_\theta(x_t, t))
```

### 5.3 Cross-Entropy Loss Derivation

**From KL to Cross-Entropy:**

*Step 1:* KL divergence for discrete distributions:
```math
D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) = \sum_v q(v) \log \frac{q(v)}{p_\theta(v)}
```

*Step 2:* Expand and separate:
```math
= \sum_v q(v) \log q(v) - \sum_v q(v) \log p_\theta(v) = -H(q) + H(q, p_\theta)
```

*Step 3:* For optimization, $H(q)$ is constant with respect to $\theta$:
```math
\nabla_\theta D_{KL} = \nabla_\theta H(q, p_\theta) = -\nabla_\theta \sum_v q(v) \log p_\theta(v)
```

*Step 4:* For masked diffusion with known $x\_0$:
```math
L = -\log p_\theta(x_0 \mid x_t) = \text{CrossEntropy}(\text{one-hot}(x_0), p_\theta(\cdot \mid x_t))
```

### 5.4 Position-Wise Loss

For a sequence of length $n$:

```math
L = \sum_{i=1}^{n} \mathbb{1}[x_t^i = [M]] \cdot \left(-\log p_\theta(x_0^i \mid x_t, t)\right)
```

**Key Insight:** We only compute loss on **masked positions**!
- If $x\_t^i \neq [M]$: No loss (position already correct)
- If $x\_t^i = [M]$: Compute cross-entropy between prediction and true $x\_0^i$

### 5.5 Loss Weighting Strategies

Different weighting schemes trade off likelihood vs sample quality:

| Strategy | Weight $w(t)$ | Properties |
|----------|---------------|------------|
| Uniform | $w(t) = 1$ | Simple, stable |
| SNR | $w(t) = \frac{\text{SNR}(t)}{\text{SNR}'(t)}$ | Theoretically optimal for ELBO |
| Min-SNR | $w(t) = \min(\text{SNR}(t), \gamma)$ | Best of both worlds ($\gamma \approx 5$) |

### 5.6 Complete Training Loss

```math
\boxed{L_\theta = \mathbb{E}_{x_0 \sim p_{\text{data}}, t \sim U(0,1), x_t \sim q(\cdot \mid x_0, t)}\left[\sum_{i: x_t^i = [M]} -\log \text{softmax}(f_\theta(x_t, t))_{x_0^i}\right]}
```

**Training Algorithm:**
```
repeat until convergence:
    1. Sample x₀ ~ training_data
    2. Sample t ~ Uniform(0, 1)
    3. Compute αₜ = 1 - t
    4. Create xₜ by masking each token with prob (1-αₜ)
    5. Compute loss: L = -∑_{masked} log p_θ(x₀ⁱ | xₜ, t)
    6. Update θ ← θ - η∇_θL
```

---

## 6. Sampling and Generation

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/03_Dream_7B_Diffusion_Large_Language_Models/svg_diagrams/06_sampling_generation.svg" alt="Sampling Generation" width="100%">
</p>

Now for the exciting part: **generating text** from a trained model!

### 6.1 Sampling Overview

**Starting Point:** All tokens are `[MASK]`
```math
x_T = [\text{M}, \text{M}, \text{M}, \ldots, \text{M}]
```

**End Point:** Coherent generated text
```math
x_0 = \text{"The cat sat on the mat"}
```

### 6.2 Basic Sampling Algorithm

```
Algorithm: MDLM/Dream Sampling
Input: Trained model f_θ, sequence length n, num_steps T
Output: Generated sequence x₀

1. Initialize: x_T ← [MASK, MASK, ..., MASK] (n tokens)

2. for t = T, T-1, ..., 1 do:
    a. Compute timestep: τ = t / T
    
    b. Predict clean tokens: logits = f_θ(xₜ, τ)
    
    c. For each masked position i where xₜⁱ = [MASK]:
        i.   Get probabilities: p(x₀ⁱ) = softmax(logitsⁱ)
        ii.  Compute unmasking prob: θₜ = (αₜ₋₁ - αₜ) / (1 - αₜ)
        iii. With probability θₜ: sample xₜ₋₁ⁱ ~ p(x₀ⁱ)
        iv.  Otherwise: xₜ₋₁ⁱ = [MASK]
    
    d. For unmasked positions: xₜ₋₁ⁱ = xₜⁱ

3. return x₀
```

### 6.3 Step-by-Step Example

Generating "The cat sat on mat" with $T=5$ steps:

| Step | τ | Sequence | Description |
|------|---|----------|-------------|
| t=5 | 1.0 | `[M] [M] [M] [M] [M]` | All masked |
| t=4 | 0.8 | `[M] cat [M] [M] [M]` | 1 token revealed |
| t=3 | 0.6 | `The cat [M] [M] mat` | 3 tokens revealed |
| t=2 | 0.4 | `The cat sat [M] mat` | 4 tokens revealed |
| t=1 | 0.2 | `The cat sat on mat` | All revealed |

**Key Observation:** Tokens are revealed **non-sequentially**! The model uses bidirectional context to inform predictions.

### 6.4 Advanced Sampling Strategies

#### Temperature Sampling
```math
p'(v) \propto p(v)^{1/\tau}
```

- $\tau < 1$: Sharper distribution (more deterministic)
- $\tau > 1$: Flatter distribution (more random)
- $\tau \to 0$: Becomes argmax

**Typical values:** $\tau = 0.7-0.9$ for coherence, $\tau = 1.0-1.2$ for creativity

#### Top-k / Top-p Sampling

- **Top-k:** Sample only from the $k$ highest probability tokens
- **Top-p (nucleus):** Sample from the smallest set with cumulative probability $\geq p$

**Typical values:** $k = 50$, $p = 0.95$

### 6.5 Parallel vs Sequential: Complexity Comparison

| Aspect | Diffusion (Dream) | Autoregressive (GPT) |
|--------|-------------------|---------------------|
| Positions per step | All $n$ | 1 |
| Total forward passes | $T$ | $n$ |
| Typical complexity | $O(T)$ where $T \in [16, 256]$ | $O(n)$ where $n$ can be $>1000$ |
| Context | Bidirectional | Unidirectional |

**Example:** For a 1000-token generation:
- GPT: 1000 forward passes
- Dream (T=64): 64 forward passes → **15.6× faster**

### 6.6 Conditional Generation

Diffusion naturally supports various conditional generation tasks:

#### Text Infilling
```
Input:  "The [M][M][M] sat on mat"
Output: "The cat sat on mat"
```
Simply initialize with partial masks and keep known tokens fixed.

#### Prompt Continuation
```
Input:  "The cat" + [M][M][M]
Output: "The cat sat on mat"
```
Set prefix as unmasked, append masked tokens, generate.

#### Classifier-Free Guidance
Scale logits based on conditional vs unconditional predictions:
```math
\tilde{\epsilon} = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
```

where $w > 1$ strengthens conditioning (typical: $w = 3-7$).

### 6.7 Quality vs Speed Trade-off

| Steps $T$ | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| 1-8 | Very fast | Lower | Real-time, drafts |
| 16-32 | Balanced | Good | General use |
| 64-256 | Slower | Best | High-quality generation |

**Dream 7B typical settings:**
- High quality: $T = 128-256$
- Fast inference: $T = 16-32$

---

## 7. Conclusion

### Summary of Key Mathematical Concepts

1. **Forward Process:** Markov chain that gradually masks tokens
```math
q(x_t \mid x_0) = \alpha_t \cdot \delta(x_t, x_0) + (1-\alpha_t) \cdot \delta(x_t, [M])
```math
2. **Reverse Process:** Neural network learns to unmask
```
p_\theta(x_{t-1} \mid x_t) \approx \sum_{x_0} q(x_{t-1} \mid x_t, x_0) \cdot p_\theta(x_0 \mid x_t)
```

3. **Training Objective:** Cross-entropy on masked positions
```math
L = -\sum_{i: x_t^i = [M]} \log p_\theta(x_0^i \mid x_t)
```math
4. **Sampling:** Iterative unmasking from full masks to text
```
x_T \to x_{T-1} \to \cdots \to x_1 \to x_0
```

### Why Diffusion LLMs Matter

| Advantage | Explanation |
|-----------|-------------|
| **Parallel Generation** | Generate all tokens simultaneously |
| **Bidirectional Context** | Each prediction sees the full sequence |
| **Flexible Inference** | Trade off speed vs quality with step count |
| **Natural Editing** | Infilling and revision are built-in |
| **Implicit Reasoning** | Multiple refinement passes enable "thinking" |

### The Future

Dream 7B represents a significant step toward **non-autoregressive language modeling**. As the field develops, we may see:

- Larger diffusion LLMs competing with AR models
- Hybrid approaches combining AR and diffusion
- Novel applications leveraging bidirectional generation
- Improved sampling algorithms for better quality/speed

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x\_0$ | Clean (original) data |
| $x\_t$ | Noisy data at time $t$ |
| $x\_T$ | Fully noisy (all masked) data |
| $[M]$ | MASK token |
| $Q\_t$ | Transition matrix at step $t$ |
| $\bar{Q}\_t$ | Cumulative transition matrix |
| $\beta\_t$ | Per-step masking probability |
| $\alpha\_t$ | Cumulative survival probability |
| $\theta\_t$ | Unmasking probability |
| $f\_\theta$ | Neural network (Transformer) |
| $p\_\theta$ | Model distribution |
| $q$ | Forward process distribution |

---

## References

1. **Dream 7B: Diffusion Large Language Models** - Ye et al., 2024
2. **Simple and Effective Masked Diffusion Language Models (MDLM)** - Sahoo et al., 2024
3. **Denoising Diffusion Probabilistic Models (DDPM)** - Ho et al., 2020
4. **Variational Diffusion Models** - Kingma et al., 2021
5. **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al., 2019

---

