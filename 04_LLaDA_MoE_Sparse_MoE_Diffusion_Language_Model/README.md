# LLaDA-MoE: A Sparse MoE Diffusion Language Model

> **Paper**: [arXiv:2509.24389](https://arxiv.org/abs/2509.24389)  
> **Authors**: Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, et al.  
> **Institutions**: Renmin University of China, Ant Group, Shanghai Jiao Tong University, Zhejiang University  
> **Models**: [HuggingFace Collection](https://huggingface.co/collections/inclusionAI/llada-68c141bca386b06b599cfe45)

---

## 📋 Table of Contents

| Section | Topic | Description |
|---------|-------|-------------|
| 01 | [Introduction](#01-introduction) | Overview and motivation |
| 02 | [Key Innovation](#02-key-innovation-moe--masked-diffusion) | Why MoE + Diffusion matters |
| 03 | [Architecture](#03-architecture) | Model structure and components |
| 04 | [Mathematical Foundations](#04-mathematical-foundations) | Core equations and notation |
| 05 | [Equation 1: Forward Process](#05-equation-1-forward-process---masking-noise) | How noise is added |
| 06 | [Equation 2: Pretrain Objective](#06-equation-2-pretraining-objective---elbo-upper-bound) | Training loss function |
| 07 | [Equation 3: MoE Routing](#07-equation-3-moe-routing-mechanism) | Expert selection math |
| 08 | [Equation 4: Auxiliary Losses](#08-equation-4-auxiliary-losses-for-load-balancing) | Load balancing math |
| 09 | [Equation 5: SFT Objective](#09-equation-5-supervised-fine-tuning-sft-objective) | Instruction tuning loss |
| 10 | [ELBO Derivation](#10-elbo-derivation-and-theoretical-proofs) | Complete proof |
| 11 | [Inference Algorithm](#11-inference-algorithm-and-generation) | How generation works |
| 12 | [Training Pipeline](#12-training-pipeline) | Multi-stage training |
| 13 | [Results](#13-results-and-benchmarks) | Performance comparison |
| 14 | [Conclusion](#14-conclusion) | Summary and future work |

---

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/01_Overview_LLaDA_MoE.svg" alt="Overview" width="100%">
</p>

---

## 01. Introduction

Large Language Models have achieved remarkable success, primarily through the **autoregressive (AR)** paradigm where tokens are generated left-to-right. However, **Masked Diffusion Models (MDMs)** offer an alternative approach with unique advantages: parallel token generation, bidirectional context, and improved controllability.

**LLaDA-MoE** represents a groundbreaking advancement: the **first masked diffusion language model trained from scratch with a Mixture-of-Experts (MoE) architecture**. This combination delivers exceptional efficiency—achieving state-of-the-art performance among diffusion language models while activating only **1.4B parameters** out of a total 7B during inference.

### 01.1 Problem Statement

| Challenge | Description |
|-----------|-------------|
| Dense Models are Expensive | 8B parameters = 8B active during inference |
| Diffusion Needs Multiple Steps | Each step = full forward pass |
| Scaling is Costly | Larger models = proportionally more compute |

### 01.2 Solution: LLaDA-MoE

| Solution | Benefit |
|----------|---------|
| Sparse MoE | Only 1.4B of 7B params active |
| 64 Experts, Top-8 | 12.5% activation rate |
| First of its Kind | Proves MoE works for diffusion LMs |

---

## 02. Key Innovation: MoE + Masked Diffusion

The core insight of LLaDA-MoE is that the efficiency benefits of MoE architecture can be successfully combined with the masked diffusion language modeling paradigm.

### 02.1 Efficiency Comparison

| Model | Total Params | Active Params | Efficiency Ratio |
|-------|--------------|---------------|------------------|
| LLaDA-8B | 8B | 8B (100%) | 1.0x (Baseline) |
| Dream-7B | 7B | 7B (100%) | 1.0x (Baseline) |
| Qwen2.5-3B | 3B | 3B (100%) | 2.7x |
| **LLaDA-MoE** | **7B** | **1.4B (20%)** | **5.7x** ✓ |

### 02.2 Why This Matters

With diffusion models requiring **T denoising steps** during generation, the efficiency gain multiplies:

```
Total Compute Savings = Efficiency Ratio × Number of Steps

Example:
- 32 steps × 5.7x efficiency = 182x total savings
- 64 steps × 5.7x efficiency = 365x total savings
```

---

## 03. Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/02_Architecture_MoE.svg" alt="Architecture" width="100%">
</p>

### 03.1 Model Specifications

| # | Component | Value | Description |
|---|-----------|-------|-------------|
| 1 | Layers | 16 | Transformer blocks |
| 2 | Hidden Dimension | 2048 | Model width |
| 3 | Attention Heads | 16 | Multi-head attention |
| 4 | Total Experts | 64 | Expert networks per layer |
| 5 | Active Experts (Top-K) | 8 | Selected per token |
| 6 | Expert Dimension | 1024 | FFN hidden size |
| 7 | RoPE Base | 50,000 | Position encoding |
| 8 | Total Parameters | 7B | Full model size |
| 9 | Active Parameters | 1.4B | Per-token compute |

### 03.2 MoE Layer Components

| # | Component | Function |
|---|-----------|----------|
| 1 | **Router** | Linear layer computing routing scores |
| 2 | **Softmax** | Normalizes scores to probabilities |
| 3 | **Top-K Selection** | Picks 8 highest-scoring experts |
| 4 | **Expert Networks** | 64 SwiGLU FFN modules |
| 5 | **Weighted Sum** | Combines selected expert outputs |

### 03.3 Key Architecture Components

| # | Component | Purpose |
|---|-----------|---------|
| 1 | RMSNorm | Efficient normalization (no mean centering) |
| 2 | SwiGLU Activation | Gated linear unit with Swish |
| 3 | RoPE Embeddings | Rotary position encoding |
| 4 | QK-LayerNorm | Stabilizes attention projections |

---

## 04. Mathematical Foundations

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/06_Mathematical_Foundations.svg" alt="Mathematical Foundations" width="100%">
</p>

### 04.1 Notation Table

| # | Symbol | Definition | Example |
|---|--------|------------|---------|
| 1 | y | Clean sequence | y ∈ {0, 1, ..., K-1}^L |
| 2 | L | Sequence length | L = 4096 |
| 3 | K | Vocabulary size | K = 32000 |
| 4 | M | Mask token | Special [MASK] symbol |
| 5 | t | Noise level | t ∈ [0, 1] |
| 6 | y_t | Noisy sequence | Partially masked |
| 7 | p_θ | Neural network | Parameterized predictor |
| 8 | p_data | Data distribution | Training corpus |
| 9 | N | Number of experts | N = 64 |
| 10 | k | Active experts | k = 8 |

### 04.2 Overview of Key Equations

| # | Equation | Purpose | Section |
|---|----------|---------|---------|
| 1 | Forward Process | Defines masking noise | §05 |
| 2 | Pretrain Objective | Training loss function | §06 |
| 3 | MoE Routing | Expert selection | §07 |
| 4 | Auxiliary Losses | Load balancing | §08 |
| 5 | SFT Objective | Instruction tuning | §09 |

---

## 05. Equation 1: Forward Process - Masking Noise

### 05.1 Definition

The forward process defines how clean data is corrupted by adding noise (masking):

**Joint Distribution**:

```math
q(y_t | t, y) = \prod_{i=1}^{L} q(y_t^i | t, y^i)
```

**Per-Token Distribution**:

```math
q(y_t^i | t, y^i) = \begin{cases}
1-t & \text{if } y_t^i = y^i \text{ (keep original token)} \\
t & \text{if } y_t^i = M \text{ (replace with mask)} \\
0 & \text{otherwise}
\end{cases}
```

### 05.2 Step-by-Step Interpretation

| Step | Action | Description |
|------|--------|-------------|
| 1 | Sample t | Draw t ~ Uniform[0, 1] |
| 2 | For each token i | Independent decision |
| 3 | With prob (1-t) | Keep original token y^i |
| 4 | With prob t | Replace with [MASK] |

### 05.3 Examples at Different Noise Levels

| # | Noise Level t | Sequence | % Masked |
|---|---------------|----------|----------|
| 1 | t = 0.0 | "The cat sat on the mat" | 0% |
| 2 | t = 0.2 | "The [M] sat on the mat" | ~20% |
| 3 | t = 0.5 | "The [M] sat [M] [M] mat" | ~50% |
| 4 | t = 0.8 | "[M] [M] [M] [M] [M] mat" | ~80% |
| 5 | t = 1.0 | "[M] [M] [M] [M] [M] [M]" | 100% |

### 05.4 Key Property

**Expected number of masked tokens**:

```math
\mathbb{E}[\text{\# masked}] = t \cdot L
```

This linear relationship is crucial for the 1/t weighting in the training objective.

---

## 06. Equation 2: Pretraining Objective - ELBO Upper Bound

### 06.1 The Loss Function

```math
\mathcal{L}_{\text{Pretrain}}(\theta) = -\mathbb{E}_{y \sim p_{\text{data}}} \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{y_t \sim q(y_t|t,y)} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbb{1}[y_t^i = M] \log p_\theta(y^i | y_t) \right]
```

### 06.2 Breaking Down the Components

| # | Component | Symbol | Meaning |
|---|-----------|--------|---------|
| 1 | Data expectation | E_{y ~ p_data} | Average over training data |
| 2 | Time expectation | E_{t ~ U[0,1]} | Average over noise levels |
| 3 | Noise expectation | E_{y_t ~ q} | Average over masked versions |
| 4 | Importance weight | 1/t | Corrects for varying # masks |
| 5 | Mask indicator | 𝟙[y_t^i = M] | Only count masked positions |
| 6 | Log probability | log p_θ(y^i \| y_t) | Model's prediction score |

### 06.3 Why the 1/t Weighting?

| # | Without 1/t | With 1/t |
|---|-------------|----------|
| 1 | High t → many masks → dominates loss | Each t weighted equally |
| 2 | Low t → few masks → ignored | Unbiased gradient estimates |
| 3 | Biased training | Valid ELBO bound |

**Mathematical Justification**:

Since E[# masked] = t · L:

```math
\frac{1}{t} \cdot \mathbb{E}[\text{\# masked}] = \frac{1}{t} \cdot tL = L \text{ (constant!)}
```

### 06.4 Training Algorithm

```
Algorithm: Pretrain Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Sample batch y ~ p_data
Step 2: Sample noise level t ~ Uniform[0, 1]
Step 3: Create masked sequence y_t using q(y_t | t, y)
Step 4: Forward pass: p = p_θ(· | y_t)
Step 5: Compute loss only on masked positions
Step 6: Apply 1/t weight
Step 7: Backpropagate and update θ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 07. Equation 3: MoE Routing Mechanism

### 07.1 The Routing Equations

**Step 1 - Compute Probabilities**:

```math
p_t = \text{Softmax}(\text{Router}(h_t))
```

**Step 2 - Weighted Expert Combination**:

```math
o_t = \sum_{i \in \text{TopK}(p_t)} p_{t,i} \cdot E_i(h_t)
```

### 07.2 Component Breakdown

| # | Component | Formula | Dimensions |
|---|-----------|---------|------------|
| 1 | Hidden state | h_t | ℝ^2048 |
| 2 | Router | W_r · h_t | ℝ^64 (one per expert) |
| 3 | Raw logits | z_t = Router(h_t) | ℝ^64 |
| 4 | Probabilities | p_t = softmax(z_t) | ℝ^64, sum = 1 |
| 5 | Selection | TopK(p_t, k=8) | 8 indices |
| 6 | Expert output | E_i(h_t) | ℝ^2048 |
| 7 | Final output | Σ p_{t,i} · E_i(h_t) | ℝ^2048 |

### 07.3 Routing Flow Diagram

```
Input: h_t ∈ ℝ^2048
          ↓
+-----------------------------+
| Router: W_r ∈ ℝ^(64×2048)   |  Step 1
| z_t = W_r · h_t ∈ ℝ^64      |
+-----------------------------+
          ↓
+-----------------------------+
| Softmax: p_t = exp(z_t) /   |  Step 2
|          Σ exp(z_t)          |
+-----------------------------+
          ↓
+-----------------------------+
| TopK: Select 8 highest      |  Step 3
| indices from p_t            |
+-----------------------------+
          ↓
+-----------------------------+
| Experts: Run E_i(h_t) for   |  Step 4
| each selected expert i      |
+-----------------------------+
          ↓
+-----------------------------+
| Combine: o_t = Σ p_i·E_i(h) |  Step 5
+-----------------------------+
          ↓
Output: o_t ∈ ℝ^2048
```

### 07.4 Renormalization

Selected weights are renormalized to sum to 1:

```math
\tilde{p}_{t,i} = \frac{p_{t,i}}{\sum_{j \in \text{TopK}(p_t)} p_{t,j}}
```

---

## 08. Equation 4: Auxiliary Losses for Load Balancing

Without regularization, routing collapses to using only a few experts. Two losses prevent this:

### 08.1 Load Balancing Loss (L_LB)

```math
\mathcal{L}_{\text{LB}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
```

| # | Symbol | Definition | Formula |
|---|--------|------------|---------|
| 1 | N | Number of experts | 64 |
| 2 | f_i | Selection frequency | (1/T) Σ_t 𝟙[i ∈ TopK(p_t)] |
| 3 | P_i | Average routing prob | (1/T) Σ_t p_{t,i} |
| 4 | T | Batch size (tokens) | Variable |

### 08.2 Why L_LB Works

| # | Condition | f_i · P_i | Result |
|---|-----------|-----------|--------|
| 1 | Expert i overused | High × High | Large penalty |
| 2 | Expert i underused | Low × Low | Small penalty |
| 3 | Balanced | 1/N × 1/N | Minimum penalty |

**Minimum achieved when**: f_i = P_i = 1/N for all experts (uniform distribution).

### 08.3 Router Z-Loss (L_Z)

```math
\mathcal{L}_{Z} = \frac{1}{T} \sum_{t=1}^{T} \left( \log \sum_{j=1}^{N} e^{z_{t,j}} \right)^2
```

| # | Purpose | Effect |
|---|---------|--------|
| 1 | Prevents large logits | Keeps routing "soft" |
| 2 | Avoids winner-take-all | Multiple experts can learn |
| 3 | Stabilizes training | Prevents gradient explosion |

### 08.4 Loss Weights (Practical Values)

| # | Loss | Weight | Contribution |
|---|------|--------|--------------|
| 1 | L_Pretrain | 1.0 | Main objective |
| 2 | L_LB | 0.01 | Load balancing |
| 3 | L_Z | 0.001 | Logit regularization |

**Total Loss**:

```math
\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Pretrain}} + 0.01 \cdot \mathcal{L}_{\text{LB}} + 0.001 \cdot \mathcal{L}_{Z}
```

---

## 09. Equation 5: Supervised Fine-Tuning (SFT) Objective

### 09.1 The SFT Loss

```math
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim p_{\text{data}}} \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{y_t \sim q(y_t|t,y)} \left[ \frac{1}{t} \sum_{i=1}^{|y|} \mathbb{1}[y_t^i = M] \log p_\theta(y^i | x, y_t) \right]
```

### 09.2 Key Difference from Pretraining

| # | Component | Pretraining | SFT |
|---|-----------|-------------|-----|
| 1 | Input | Sequence y only | Prompt x + Response y |
| 2 | Masking | All tokens can be masked | Only response y is masked |
| 3 | Conditioning | y_t only | x (clean) + y_t (masked) |
| 4 | Goal | General LM | Instruction following |

### 09.3 SFT Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prompt (x):     "What is the capital of France?"
                 ↑ Never masked - always visible as context

Response (y):   "The capital of France is Paris."
                 ↑ Masked at rate t

If t = 0.5:     "The [M] of [M] is Paris."
                 ↑ Model learns to predict "capital" and "France"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 09.4 SFT Training Algorithm

```
Algorithm: SFT Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Sample (prompt x, response y) pair
Step 2: Sample t ~ Uniform[0, 1]
Step 3: Mask response: y_t ~ q(y_t | t, y)
Step 4: Keep prompt clean: input = [x, y_t]
Step 5: Forward pass: p = p_θ(· | x, y_t)
Step 6: Compute loss on masked response positions
Step 7: Apply 1/t weight and backpropagate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 10. ELBO Derivation and Theoretical Proofs

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/07_ELBO_Derivation_Proof.svg" alt="ELBO Derivation" width="100%">
</p>

### 10.1 Goal

Maximize the log-likelihood of data:

```math
\max_\theta \mathbb{E}_{y \sim p_{\text{data}}} [\log p_\theta(y)]
```

### 10.2 Problem

Direct computation is intractable:

```math
p_\theta(y) = \int p_\theta(y | y_t) p(y_t) dy_t
```

### 10.3 Solution: ELBO Derivation (Step-by-Step)

| Step | Action | Formula |
|------|--------|---------|
| 1 | Start with marginal | log p_θ(y) = log ∫ p_θ(y, y_t) dy_t |
| 2 | Importance sampling | = log ∫ q(y_t\|y) · [p_θ(y, y_t) / q(y_t\|y)] dy_t |
| 3 | Rewrite as expectation | = log E_{y_t ~ q} [p_θ(y, y_t) / q(y_t\|y)] |
| 4 | Apply Jensen's inequality | ≥ E_{y_t ~ q} [log(p_θ(y, y_t) / q(y_t\|y))] |
| 5 | Expand with chain rule | = E_{y_t ~ q} [log p_θ(y\|y_t) + log p_θ(y_t) - log q(y_t\|y)] |
| 6 | Simplify for MDM | ≈ E_{t, y_t} [(1/t) Σ_i 𝟙[y_t^i=M] log p_θ(y^i\|y_t)] |

### 10.4 Jensen's Inequality Proof

For concave function f (like log):

```math
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
```

Therefore:

```math
\log \mathbb{E}[X] \geq \mathbb{E}[\log X]
```

### 10.5 Theoretical Properties

| # | Property | Statement | Meaning |
|---|----------|-----------|---------|
| 1 | Valid Lower Bound | -L_Pretrain ≤ log p_θ(y) | Minimizing loss increases likelihood |
| 2 | Tightness | Equality when q = p_θ posterior | Bound can be exact |
| 3 | Unbiased Gradients | E[∇L] = ∇ELBO | Correct optimization direction |
| 4 | Consistency | As L→0, p_θ→p_data | Model learns true distribution |

### 10.6 Why ELBO Works for MDM

| # | Fact | Implication |
|---|------|-------------|
| 1 | ELBO lower bounds log p(y) | Maximizing ELBO → Maximizing likelihood |
| 2 | ELBO is tractable | Can compute with Monte Carlo samples |
| 3 | 1/t makes it unbiased | All noise levels contribute equally |
| 4 | Only masked positions matter | Efficient computation |

---

## 11. Inference Algorithm and Generation

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/08_Inference_Algorithm.svg" alt="Inference Algorithm" width="100%">
</p>

### 11.1 Sampling Algorithm

```
Algorithm: Generate(prompt x, steps T, response_length L)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT:
  x = prompt tokens (conditioning context)
  T = number of denoising steps (e.g., 32)
  L = length of response to generate

INITIALIZE:
  y = [M, M, ..., M]  # L mask tokens

FOR t = T, T-1, ..., 1:
    Step 1: Forward pass
            p = p_θ(· | x, y)           # Get token probabilities
    
    Step 2: Sample tokens
            For each masked position i:
              ŷ^i ~ Categorical(p^i)    # Sample from distribution
              conf_i = max_k p^i_k      # Record confidence
    
    Step 3: Compute unmask count
            n = ⌊L × (1/t - 1/(t+1))⌋   # How many to unmask
    
    Step 4: Select by confidence
            Select top n positions by conf_i
    
    Step 5: Update sequence
            y^i = ŷ^i for selected positions  # Unmask

RETURN: y = final generated response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 11.2 Generation Example (T=4 steps, L=8 tokens)

| Step | t | Sequence | % Unmasked |
|------|---|----------|------------|
| Init | 4 | [M][M][M][M][M][M][M][M] | 0% |
| 1 | 4 | [The][M][M][M][M][M][M][.] | 25% |
| 2 | 3 | [The][cat][M][M][the][M][M][.] | 50% |
| 3 | 2 | [The][cat][sat][M][the][M][mat][.] | 75% |
| 4 | 1 | [The][cat][sat][on][the][cozy][mat][.] | 100% |

### 11.3 Confidence-Based Selection

| # | Step | Action |
|---|------|--------|
| 1 | Compute confidence | conf_i = max_k p_θ(k \| x, y_t) |
| 2 | Rank positions | Sort by confidence (highest first) |
| 3 | Select top-n | Unmask n positions with highest conf |
| 4 | Keep rest masked | Low confidence → need more refinement |

**Intuition**: "Easy" tokens (high confidence) are finalized first; "hard" tokens are revised over multiple steps.

### 11.4 Comparison: Diffusion vs Autoregressive

| # | Aspect | Autoregressive | Masked Diffusion |
|---|--------|----------------|------------------|
| 1 | Direction | Left-to-right only | Any order |
| 2 | Parallelism | 1 token/step | All tokens/step |
| 3 | Forward passes | L passes for L tokens | T passes (T << L) |
| 4 | Revision | ❌ No (tokens are final) | ✅ Yes (remasking) |
| 5 | Context | Causal (past only) | Full (bidirectional) |
| 6 | Generation | [The] → [The][cat] → ... | [MMM] → [The][M][sat] → ... |

---

## 12. Training Pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/04_Training_Pipeline.svg" alt="Training Pipeline" width="100%">
</p>

### 12.1 Overview

| Stage | Name | Tokens | Context | Purpose |
|-------|------|--------|---------|---------|
| 1 | Pretrain Stage 1 | 10T | 4k | Base language modeling |
| 2 | Pretrain Stage 2 | 10T | 4k | Math/code emphasis |
| 3 | Annealing Stage 1 | 500B | 4k | High-quality data |
| 4 | Annealing Stage 2 | 500B | 8k | Context extension |
| 5 | SFT Stage | - | 8k | Instruction tuning |

### 12.2 Stage Details

#### 12.2.1 Pretrain Stage 1 (10T tokens)
| Item | Value |
|------|-------|
| Data | Large mixed text corpus |
| Context | 4,096 tokens |
| Purpose | Basic language understanding |
| RoPE Base | 10,000 |

#### 12.2.2 Pretrain Stage 2 (10T tokens)
| Item | Value |
|------|-------|
| Data | Resampled corpus |
| Emphasis | ↑ Mathematics, ↑ Code |
| Purpose | Reasoning abilities |
| RoPE Base | 10,000 |

#### 12.2.3 Annealing Stage 1 (500B tokens)
| Item | Value |
|------|-------|
| Init | Best checkpoint from Stage 2 |
| Data | High-quality only |
| LR Schedule | Decay |
| Purpose | Quality refinement |

#### 12.2.4 Annealing Stage 2 (500B tokens, 8k)
| Item | Value |
|------|-------|
| Context | 4k → 8k (extended) |
| RoPE Base | 10,000 → 50,000 |
| Purpose | Long sequence support |

#### 12.2.5 SFT Stage
| Item | Value |
|------|-------|
| Data | Question-answer pairs |
| Masking | Response only (Eq. 5) |
| Purpose | Instruction following |

### 12.3 Variable-Length Training

To handle train-test distribution mismatch:

```
For 99% of steps:
    Use full 4k context
    
For 1% of steps:
    Sample ℓ ~ Uniform[8, 4096]
    Truncate input to ℓ tokens
```

### 12.4 Output Models

| # | Model | Description |
|---|-------|-------------|
| 1 | LLaDA-MoE-7B-A1B-Base | After Annealing Stage 2 |
| 2 | LLaDA-MoE-7B-A1B-Instruct | After SFT |

---

## 13. Results and Benchmarks

<p align="center">
  <img src="https://raw.githubusercontent.com/Gaurav14cs17/Diffusion-Language-Model/main/04_LLaDA_MoE_Sparse_MoE_Diffusion_Language_Model/images/05_Results_Benchmarks.svg" alt="Results" width="100%">
</p>

### 13.1 Key Finding

> **LLaDA-MoE beats 8B dense diffusion models with only 1.4B active parameters!**

### 13.2 Base Model Comparison

| # | Task | LLaDA-MoE (1.4B) | LLaDA-8B | Dream-7B | Qwen2.5-3B |
|---|------|------------------|----------|----------|------------|
| 1 | MMLU | 64.59 | 65.90 | **69.50** | 67.98 |
| 2 | MMLU-Pro | 39.16 | 41.80 | **48.15** | 35.50 |
| 3 | BBH | 52.71 | 49.80 | **57.90** | 56.50 |
| 4 | GSM8K | 66.41 | 70.70 | 77.79 | **78.17** |
| 5 | MATH | **36.10** | 27.30 | 39.60 | 40.94 |
| 6 | HumanEval | 45.73 | 33.50 | **57.90** | 57.93 |
| | **Average** | **46.94** | 43.53 | 46.66 | 50.34 |

### 13.3 Instruct Model Comparison

| # | Comparison | Result |
|---|------------|--------|
| 1 | vs LLaDA-8B-Instruct | **LLaDA-MoE wins**: MMLU-Pro (52.4 vs 49.4), HumanEval (60.4 vs 55.5), MBPP (58.8 vs 44.6) |
| 2 | vs Dream-v0-Instruct | **Competitive**: Better on MMLU-Pro, LiveCodeBench |
| 3 | vs Qwen2.5-3B-Instruct | **Comparable**: Similar with fewer active params |

### 13.4 Efficiency Analysis

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPUTE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model          | Active Params | Relative Cost | Efficiency
---------------+---------------+---------------+------------
LLaDA-8B       | 8.0B          | 8.0x          | 1.0x
Dream-7B       | 7.0B          | 7.0x          | 1.1x
Qwen2.5-3B     | 3.0B          | 3.0x          | 2.7x
LLaDA-MoE      | 1.4B          | 1.4x          | 5.7x ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITH DIFFUSION STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Steps  | LLaDA-8B Cost | LLaDA-MoE Cost | Total Savings
-------+---------------+----------------+---------------
10     | 80.0x         | 14.0x          | 5.7x
20     | 160.0x        | 28.0x          | 5.7x
50     | 400.0x        | 70.0x          | 5.7x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 14. Conclusion

### 14.1 Summary of Contributions

| # | Contribution | Description |
|---|--------------|-------------|
| 1 | **First MoE + Diffusion LM** | Pretrained from scratch with sparse MoE |
| 2 | **5.7x Efficiency** | 1.4B active vs 8B parameters |
| 3 | **State-of-the-Art** | Best among diffusion LMs at this size |
| 4 | **Comprehensive Training** | 20T+ tokens, 5 stages, context extension |

### 14.2 Key Takeaways

✅ **5.7x compute efficiency** (1.4B vs 8B active parameters)  
✅ **State-of-the-art** among diffusion language models  
✅ **Competitive with Qwen2.5-3B-Instruct** using fewer active parameters  
✅ **First MoE diffusion LM** trained from scratch  

### 14.3 Future Directions

| # | Direction | Description |
|---|-----------|-------------|
| 1 | Larger MoE | 100B+ total parameters |
| 2 | Better Routing | Improved mechanisms for diffusion |
| 3 | Combined Efficiency | With other optimization techniques |
| 4 | Multimodal | MoE diffusion for vision-language |

---

## Citation

```bibtex
@article{zhu2025llada-moe,
  title={LLaDA-MoE: A Sparse MoE Diffusion Language Model},
  author={Zhu, Fengqi and You, Zebin and Xing, Yipeng and Huang, Zenan and Liu, Lin and others},
  journal={arXiv preprint arXiv:2509.24389},
  year={2025}
}
```

---

## Resources

- 📄 **Paper**: [arXiv:2509.24389](https://arxiv.org/abs/2509.24389)
- 🤗 **Models**: [HuggingFace Collection](https://huggingface.co/collections/inclusionAI/llada-68c141bca386b06b599cfe45)
- 🏢 **Institutions**: Renmin University, Ant Group, SJTU, Zhejiang University

---

*Last updated: December 2024*
