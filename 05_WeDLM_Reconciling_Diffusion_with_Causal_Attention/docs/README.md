# 🧠 WeDLM: A Deep Dive into Reconciling Diffusion Language Models with Causal Attention

## A Comprehensive Mathematical Explanation

**Paper**: *WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference*  
**Authors**: Liu et al., Tencent WeChat AI (2025)

---

## 📋 Table of Contents

1. [Introduction: The Speed Problem in Language Models](#1-introduction-the-speed-problem-in-language-models)
2. [Background: Autoregressive vs Diffusion Models](#2-background-autoregressive-vs-diffusion-models)
3. [The Core Innovation: Topological Reordering](#3-the-core-innovation-topological-reordering)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Streaming Parallel Decoding](#5-streaming-parallel-decoding)
6. [Entropy-Based Position Selection](#6-entropy-based-position-selection)
7. [The Complete WeDLM Algorithm](#7-the-complete-wedlm-algorithm)
8. [Training Methodology](#8-training-methodology)
9. [Theoretical Analysis: Why This Works](#9-theoretical-analysis-why-this-works)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction: The Speed Problem in Language Models

### The Fundamental Challenge

Modern Large Language Models (LLMs) like GPT-4, Qwen, and LLaMA use **autoregressive (AR) decoding**:

```math
P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, x_2, \ldots, x_{i-1})

```

This means generating $n$ tokens requires **exactly $n$ sequential forward passes** through the model. Each token depends on all previous tokens, creating a fundamental speed bottleneck.

### The Diffusion Alternative

**Discrete Diffusion Language Models (DLMs)** offer a different approach. Instead of generating one token at a time, they start with a sequence of "mask" tokens and iteratively refine (unmask) multiple positions simultaneously:

```math
\text{Initial: } [\text{MASK}][\text{MASK}][\text{MASK}][\text{MASK}][\text{MASK}]
\text{Step 1: } [\text{The}][\text{MASK}][\text{MASK}][\text{MASK}][\text{MASK}]
\text{Step 2: } [\text{The}][\text{quick}][\text{MASK}][\text{fox}][\text{MASK}]
\text{Step 3: } [\text{The}][\text{quick}][\text{brown}][\text{fox}][\text{jumps}]

```

**The Promise**: Generate $n$ tokens in $k \ll n$ steps by unmasking multiple tokens per step.

### The Hidden Problem: Bidirectional Attention

Most diffusion language models (like LLaDA, SEDD, MDLM) use **bidirectional attention**:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

```

Where each position can attend to **all** other positions.

**Why is this a problem?**

1. **No KV Cache**: In AR models, we cache Key-Value pairs from previous positions. Bidirectional attention invalidates this because future positions change the attention patterns.

2. **No Hardware Optimizations**: Production systems like vLLM use:
   - **PagedAttention**: Efficient memory management assuming causal attention
   - **CUDA Graphs**: Compile execution patterns assuming fixed attention structure
   - **FlashAttention**: Optimized for causal masks

3. **Quadratic Overhead**: Without KV cache, each step requires full $O(n^2)$ attention computation.

**Result**: The theoretical speedup from parallel decoding is **completely negated** by implementation inefficiencies.

---

## 2. Background: Autoregressive vs Diffusion Models

### Autoregressive Modeling

The standard AR factorization:

```math
P_\theta(x_{1:n}) = \prod_{i=1}^{n} P_\theta(x_i \mid x_{1:i-1})

```

**Attention Mask (Causal)**:

```math
M_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}

```

This mask ensures position $i$ can only attend to positions $\leq i$:

```math
\text{Attention}_i = \text{softmax}\left(\frac{q_i \cdot K_{1:i}^\top}{\sqrt{d_k}} + M_{i, 1:n}\right) \cdot V_{1:i}

```

**KV Cache**: Since $K\_{1:i-1}$ and $V\_{1:i-1}$ are computed once and never change, we cache them:

```
Step 1: Compute K₁, V₁, cache them
Step 2: Load K₁, V₁ from cache, compute K₂, V₂, cache them
Step 3: Load K₁, K₂, V₁, V₂ from cache, compute K₃, V₃...

```

### Diffusion Language Models

Discrete diffusion models define a forward process that corrupts data into noise (masks):

**Forward Process**:

```math
q(x_t \mid x_0) = \text{Categorical}(x_t; p = (1-\beta_t)x_0 + \beta_t \cdot \mathbf{u})

```

Where $\mathbf{u} = [0, \ldots, 0, 1]$ is the one-hot vector for `[MASK]`, and $\beta\_t$ is a noise schedule.

**Reverse Process**:

```math
p_\theta(x_{t-1} \mid x_t) = \prod_{i: x_t^{(i)} = \text{MASK}} P_\theta(x_{t-1}^{(i)} \mid x_t)

```

The model predicts what token should replace each mask, conditioned on the current (partially masked) sequence.

### The Compatibility Problem

| Feature | AR Models | Standard DLMs | Impact |
|---------|-----------|---------------|--------|
| Attention | Causal | Bidirectional | DLMs can't use KV cache |
| KV Cache | ✅ Yes | ❌ No | DLMs need full recompute |
| PagedAttention | ✅ Yes | ❌ No | DLMs waste memory |
| FlashAttention v2 | ✅ Optimized | ⚠️ Less efficient | DLMs are slower |
| CUDA Graphs | ✅ Yes | ❌ No | DLMs have kernel overhead |

---

## 3. The Core Innovation: Topological Reordering

### The Key Insight

WeDLM's breakthrough realization:

> **Mask tokens at future positions don't need to attend to each other or to non-mask tokens at future positions!**

Consider generating the sequence "[The] [quick] [brown] [fox]":

```
State:  [The] [MASK] [MASK] [MASK]
               ↑       ↑       ↑
           Position 2  3       4

```

**Traditional Bidirectional Thinking**:
- Position 2's [MASK] should see positions 3, 4 to decide what token to generate
- Position 3's [MASK] should see positions 2, 4 as well

**WeDLM's Insight**:
- Position 2's [MASK] only needs to see "[The]" (the prefix context)
- Position 3's [MASK] only needs to see "[The]" (same prefix context)
- Position 4's [MASK] only needs to see "[The]" (same prefix context)

**Why?** Because all masks are **independently** predicting tokens given the **same** committed prefix!

### The Topological Reordering Trick

To use causal attention while processing multiple mask positions, WeDLM **reorders** the input sequence:

**Original Order** (positions 1, 2, 3, 4):

```
[The] [MASK₂] [MASK₃] [MASK₄]

```

**Reordered for Causal Attention** (positions 1, 2, 1, 2, 1, 2):

```
[The] [MASK₂]
[The] [MASK₃]
[The] [MASK₄]

```

But this is inefficient (duplicates context). WeDLM uses a clever attention structure instead.

### The WeDLM Attention Pattern

WeDLM maintains causal attention but processes multiple masks in one forward pass:

**Window Setup**:
- **Prefix**: Committed tokens (e.g., "[The]") - these are in KV cache
- **Window**: Current mask tokens being processed

**Key Innovation**: Arrange the window so:
1. Non-mask tokens in window come first
2. Mask tokens in window come last
3. Causal attention naturally gives correct dependencies

```
+-----------------------------------------+
|  Prefix (cached)  |  Non-mask  |  Masks |
|   [The] [quick]   |   [brown]  | [M] [M]|
+-----------------------------------------+
                     ← Causal attention flows this way ←

```

Each mask position:
- ✅ Sees all prefix tokens (via KV cache)
- ✅ Sees non-mask tokens in window
- ❌ Cannot see other mask positions (causal mask blocks this)
- ❌ Cannot see future positions (causal mask blocks this)

**This is exactly what we want!**

---

## 4. Mathematical Formulation

### Formal Definition

Let $S = (x\_1, x\_2, \ldots, x\_L)$ be a sequence where:
- $x\_i \in V$ (vocabulary) for committed tokens
- $x\_i = \text{[MASK]}$ for mask tokens

Define:
- $\mathcal{P} = \{i : x\_i \neq \text{[MASK]}\}$ - indices of committed (prefix) tokens
- $\mathcal{M} = \{i : x\_i = \text{[MASK]}\}$ - indices of mask tokens

### The WeDLM Objective

For each mask position $j \in \mathcal{M}$, WeDLM models:

```math
P_\theta(x_j \mid \{x_i : i \in \mathcal{P}, i < j\})

```

**Note**: Position $j$ only conditions on prefix tokens that appear **before** it in the sequence, not on other masks or future tokens.

### Proof: Equivalence to Masked Language Modeling Under Causal Constraint

**Theorem 1**: *WeDLM's formulation is equivalent to causal masked language modeling.*

**Proof**:

Standard masked LM objective:

```math
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim p_{\text{data}}}\left[\sum_{j \in \mathcal{M}} \log P_\theta(x_j \mid x_{\mathcal{M}^c})\right]

```

Where $\mathcal{M}^c$ is the set of non-masked positions.

WeDLM's causal MLM objective:

```math
\mathcal{L}_{\text{CMLM}} = -\mathbb{E}_{x \sim p_{\text{data}}}\left[\sum_{j \in \mathcal{M}} \log P_\theta(x_j \mid x_{1:j-1} \cap \mathcal{M}^c)\right]

```

For left-to-right languages, the conditional independence assumption:

```math
P(x_j \mid x_{\mathcal{M}^c}) \approx P(x_j \mid x_{1:j-1} \cap \mathcal{M}^c)

```

holds because future context (positions $> j$) provides redundant information that the model can learn to predict from left context. $\square$

### The Attention Mask Formulation

For a window of size $W$ with positions $p\_1, p\_2, \ldots, p\_W$ where:
- Positions $p\_1, \ldots, p\_k$ are non-mask
- Positions $p\_{k+1}, \ldots, p\_W$ are mask

The attention mask $A \in \mathbb{R}^{W \times (L\_{\text{prefix}} + W)}$:

```math
A_{ij} = \begin{cases}
0 & \text{if } j \leq L_{\text{prefix}} \text{ (can attend to prefix)} \\
0 & \text{if } j - L_{\text{prefix}} \leq i \text{ and } j \leq L_{\text{prefix}} + k \text{ (can attend to prior non-masks)} \\
-\infty & \text{otherwise}
\end{cases}

```

### Position Encoding

WeDLM uses standard **Rotary Position Embeddings (RoPE)**:

```math
\text{RoPE}(x_m, m) = \begin{pmatrix}
x_m^{(1)} \cos(m\theta_1) - x_m^{(2)} \sin(m\theta_1) \\
x_m^{(1)} \sin(m\theta_1) + x_m^{(2)} \cos(m\theta_1) \\
\vdots
\end{pmatrix}

```

Where $\theta\_k = 10000^{-2k/d}$.

**Critical**: Each mask token gets its **true** position index, not a shifted one. This maintains positional consistency with the AR base model.

---

## 5. Streaming Parallel Decoding

### The Sliding Window Mechanism

WeDLM uses a **sliding window** of size $W$ (typically 16) for generation:

```
Generation Flow:
===============================================================

Prefix (committed, in KV cache)     Window (processing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 0: []                          [M][M][M][M][M][M][M][M]...
        
Step 1: [The]                       [quick][M][M][M][M][M][M]...
        ↑ committed                  ↑ predicted, not committed yet

Step 2: [The][quick][brown]         [fox][M][M][M][M][M]...
        ↑↑↑ prefix grows             ↑ predicted

Step 3: [The][quick][brown][fox]    [jumps][over][M][M][M]...

```

### Formal Algorithm

**Algorithm**: Streaming Parallel Decoding (SPD)

```
Input: prompt tokens x₁, ..., xₚ, window size W
Output: generated tokens

1. Initialize KV cache with prompt
2. Initialize window: [MASK]₁, [MASK]₂, ..., [MASK]_W
3. Set prefix_len = p

4. while not finished:
5.     # Reorder window: non-masks first, then masks
6.     reordered_window = sort(window, key=is_mask)
7.     
8.     # Single forward pass with KV cache
9.     logits = model.forward(reordered_window, kv_cache)
10.    
11.    # Process mask positions
12.    for each mask position j in window:
13.        entropy_j = compute_entropy(logits[j])
14.        if should_fill(entropy_j):
15.            window[j] = sample(logits[j])
16.            mark_as_non_mask(j)
17.    
18.    # Find committed prefix (consecutive non-masks from start)
19.    committed = longest_non_mask_prefix(window)
20.    
21.    # Commit prefix to KV cache
22.    append_to_output(committed)
23.    kv_cache.append(committed)
24.    prefix_len += len(committed)
25.    
26.    # Slide window
27.    window = window[len(committed):] + new_masks(len(committed))
28.    
29.    # Check stopping condition
30.    if stop_token in committed:
31.        finished = True

32. return output

```

### Prefix Commitment Strategy

**Key Insight**: Only consecutive non-mask tokens from the start of the window can be committed.

**Example**:

```
Window:  [quick] [MASK] [fox] [jumps]
                   ↑
              Gap here!

Committable: [quick]  (only the first consecutive non-mask run)

```

**Why?** 
- Tokens after a gap might change when the gap is filled
- Only tokens before all remaining masks are "finalized"

### Proof: Consistency of Prefix Commitment

**Theorem 2**: *Committed tokens will not affect or be affected by future generation steps.*

**Proof**:

Let $C = (c\_1, \ldots, c\_k)$ be committed tokens at positions $p\_1, \ldots, p\_k$.
Let $M = (m\_1, \ldots, m\_j)$ be remaining mask positions at $q\_1, \ldots, q\_j$.

Since $C$ are consecutive from window start: $p\_k < q\_1$ (all committed positions precede all mask positions).

For causal attention at any mask position $q\_i$:

```math
\text{Attention}(q_i) = f\left(\{x_\ell : \ell < q_i\}\right)

```

Since $p\_k < q\_i$ for all $i$:
1. All committed tokens $C$ are visible to all masks $M$
2. No mask token is visible to any committed token (causal mask)

Therefore:
1. Committed tokens are computed only from their left context → stable
2. Future mask predictions use committed tokens but don't modify them → consistent

$\square$

---

## 6. Entropy-Based Position Selection

### The Decision Problem

At each step, we have mask positions with predicted distributions. We must decide:
1. **Which masks to fill?** (Position selection)
2. **What tokens to use?** (Token sampling)

### Entropy as Confidence Measure

For a probability distribution $P = (p\_1, \ldots, p\_V)$ over vocabulary $V$:

```math
H(P) = -\sum_{i=1}^{V} p_i \log p_i

```

**Properties**:
- $H(P) = 0$ when $P$ is deterministic (one token has probability 1)
- $H(P) = \log V$ when $P$ is uniform (maximum uncertainty)

**Intuition**: Low entropy = model is confident → safe to commit this prediction

### Position-Adjusted Entropy

WeDLM adds a **position penalty** to encourage left-to-right generation:

```math
\tilde{H}_j = H(P_j) + \lambda \cdot (j - j_{\min})

```

Where:
- $H(P\_j)$ = raw entropy at position $j$
- $\lambda$ = position penalty factor (default: 0.02)
- $j\_{\min}$ = first mask position in window

**Effect**: Earlier positions are favored when entropies are similar.

### Selection Criteria

**Threshold-based parallel decoding**:

```math
\text{Fill}(j) = \mathbb{1}\left[\tilde{H}_j < \tau\right]

```

Where $\tau$ is the entropy threshold (default: 0.4).

**Fallback**: If no position satisfies the threshold:

```math
j^* = \arg\min_j \tilde{H}_j

```

### Proof: Threshold Selection Minimizes Expected Error

**Theorem 3**: *Under mild assumptions, entropy-based selection minimizes expected prediction error.*

**Setup**: Let $e\_j = \mathbb{1}[\hat{x}\_j \neq x\_j^*]$ be the error indicator for position $j$.

**Assumption** (Calibration): The model's predictive probability reflects true accuracy:

```math
P(\hat{x}_j = x_j^*) \approx \max_k p_{j,k}

```

**Lemma**: For a categorical distribution, high confidence (low entropy) correlates with low error probability.

```math
\mathbb{E}[e_j] = 1 - \max_k p_{j,k} \leq 1 - \exp(-H(P_j))

```

**Proof of bound**:
Using the relationship between entropy and max probability:

```math
H(P) \geq -\log(\max_k p_k) \implies \max_k p_k \geq \exp(-H(P))

```

Therefore:

```math
\mathbb{E}[e_j] = 1 - \max_k p_{j,k} \leq 1 - \exp(-H(P_j))

```

**Corollary**: Selecting positions with $H(P\_j) < \tau$ ensures:

```math
\mathbb{E}[e_j] \leq 1 - \exp(-\tau)

```

For $\tau = 0.4$: $\mathbb{E}[e\_j] \leq 0.33$

$\square$

### Implementation

From `sampler.py`:

```python
def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy for each position's probability distribution."""
    return torch.distributions.Categorical(logits=logits).entropy()

def select_positions_to_fill(
    self,
    entropy: torch.Tensor,
    remaining_mask_indices: List[int],
    entropy_threshold: Optional[float],
    pos_penalty_factor: float
) -> List[int]:
    # Compute position-based penalty
    mask_indices_tensor = torch.tensor(remaining_mask_indices)
    base_pos = mask_indices_tensor[0]
    distances = mask_indices_tensor - base_pos
    position_penalty = distances * pos_penalty_factor
    
    # Add penalty to entropy
    adjusted_entropy = entropy + position_penalty
    
    # Select positions based on threshold
    if entropy_threshold is not None:
        candidates = (adjusted_entropy < entropy_threshold).nonzero()
        if candidates.numel() > 0:
            return candidates.tolist()
    
    # Fallback: minimum adjusted entropy
    return [int(adjusted_entropy.argmin().item())]

```

---

## 7. The Complete WeDLM Algorithm

### Inference Pipeline

```
+----------------------------------------------------------------------+
|                         WeDLM INFERENCE FLOW                         |
+----------------------------------------------------------------------+

Step 1: PREFILL
+---------------------------------------------------------------------+
|  Prompt: "Solve: 2x + 5 = 13"                                       |
|                                                                     |
|  +--------------------------------------------------------------+   |
|  | [Solve] [:] [2x] [+] [5] [=] [13]                            |   |
|  +--------------------------------------------------------------+   |
|                           ↓                                         |
|  +--------------------------------------------------------------+   |
|  |                      KV Cache                                |   |
|  |  K₁,V₁ | K₂,V₂ | K₃,V₃ | K₄,V₄ | K₅,V₅ | K₆,V₆ | K₇,V₇       |   |
|  +--------------------------------------------------------------+   |
+---------------------------------------------------------------------+

Step 2: INITIALIZE WINDOW
+---------------------------------------------------------------------+
|  Window (size W=8):                                                 |
|  +--------------------------------------------------------------+   |
|  | [M] [M] [M] [M] [M] [M] [M] [M]                              |   |
|  +--------------------------------------------------------------+   |
|   pos: 8   9  10  11  12  13  14  15                                |
+---------------------------------------------------------------------+

Step 3: DECODE ITERATION
+---------------------------------------------------------------------+
|                                                                     |
|  3a. Forward Pass (with KV cache + window)                          |
|  +--------------------------------------------------------------+   |
|  | Query: [M] [M] [M] [M] [M] [M] [M] [M]                       |   |
|  | Keys:  [Solve][:][2x][+][5][=][13][M][M][M][M][M][M][M][M]   |   |
|  |        +------ KV Cache -------+ +---- Window ---------+     |   |
|  +--------------------------------------------------------------+   |
|                           ↓                                         |
|  3b. Get Logits for Each Position                                   |
|  +--------------------------------------------------------------+   |
|  | logits[0]: P(token|prefix) → "To" (H=0.3)                    |   |
|  | logits[1]: P(token|prefix) → "solve" (H=0.25)                |   |
|  | logits[2]: P(token|prefix) → "," (H=0.8)                     |   |
|  | ...                                                          |   |
|  +--------------------------------------------------------------+   |
|                           ↓                                         |
|  3c. Compute Adjusted Entropy                                       |
|  +-------------------------------------------------------------+    |
|  | H̃[0] = 0.30 + 0.02×0 = 0.30  ← BELOW THRESHOLD (0.4)         |   |
|  | H̃[1] = 0.25 + 0.02×1 = 0.27  ← BELOW THRESHOLD               |   |
|  | H̃[2] = 0.80 + 0.02×2 = 0.84  ← ABOVE THRESHOLD               |   |
|  | ...                                                          |   |
|  +--------------------------------------------------------------+   |
|                           ↓                                         |
|  3d. Fill Selected Positions                                        |
|  +--------------------------------------------------------------+   |
|  | [To] [solve] [M] [M] [M] [M] [M] [M]                         |   |
|  |  ↑     ↑                                                     |   |
|  |  Filled!                                                     |   |
|  +--------------------------------------------------------------+   |
|                           ↓                                         |
|  3e. Commit Prefix & Slide Window                                   |
|  +--------------------------------------------------------------+   |
|  | Commit: [To] [solve] → append to KV cache                    |   |
|  | New Window: [M] [M] [M] [M] [M] [M] [M] [M]                  |   |
|  |             +- old masks ------+ + new masks +               |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+

Step 4: REPEAT Until Stop Token or Max Length

```

### Complexity Analysis

**Per-step Complexity**:

| Operation | WeDLM | Traditional DLM | AR |
|-----------|-------|-----------------|-----|
| Forward pass | $O(W \cdot L)$ | $O(L^2)$ | $O(L)$ |
| Tokens generated | $k$ (variable) | $k$ (variable) | $1$ |
| KV cache update | $O(W)$ | $O(L)$ | $O(1)$ |
| Memory | $O(L + W)$ | $O(L)$ | $O(L)$ |

Where:
- $L$ = current sequence length
- $W$ = window size (typically 16)
- $k$ = tokens filled per step

**Theorem 4** (Speedup Bound): *WeDLM achieves speedup factor:*

```math
\text{Speedup} = \frac{\mathbb{E}[k]}{1} = \mathbb{E}[k]

```

*where $k$ is the number of positions filled per step.*

**Empirical Results**: 
- Math reasoning (GSM8K): $\mathbb{E}[k] \approx 4-6$
- Code generation: $\mathbb{E}[k] \approx 2-3$
- Open-ended QA: $\mathbb{E}[k] \approx 1.5-2$

---

## 8. Training Methodology

### Training Objective

WeDLM uses a **causal masked language modeling (CMLM)** objective:

```math
\mathcal{L}_{\text{CMLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{\mathcal{M} \sim p(\mathcal{M}|x)} \left[\sum_{j \in \mathcal{M}} \log P_\theta(x_j \mid x_{1:j-1} \cap \mathcal{M}^c)\right]

```

Where:
- $\mathcal{D}$ = training data distribution
- $\mathcal{M}$ = set of masked positions
- $p(\mathcal{M}|x)$ = masking strategy

### Masking Strategy

**Random Span Masking**: Sample spans rather than individual tokens:

1. Sample span length $\ell \sim \text{Geometric}(p=0.2)$
2. Sample start position uniformly
3. Mask positions in span
4. Repeat until target mask ratio reached

**Mask Ratio Schedule**:

```math
r_t = r_{\min} + (r_{\max} - r_{\min}) \cdot \frac{t}{T}

```

Where:
- $r\_{\min} = 0.1$ (10% masking)
- $r\_{\max} = 0.5$ (50% masking)
- $t$ = training step, $T$ = total steps

### Training from AR Models

**Key Advantage**: WeDLM can be initialized from pretrained AR models!

**Why?** 
1. Same architecture (causal attention)
2. Same position encoding (RoPE)
3. Compatible objective (next token prediction is a special case)

**Procedure**:
1. Load pretrained AR model weights
2. Add `[MASK]` token to vocabulary (use embedding of rare token)
3. Fine-tune with CMLM objective

**Training Efficiency**:
- ~10% of original pretraining compute
- 100B tokens of fine-tuning data
- Preserves base model capabilities

### Loss Function Implementation

```python
def compute_cmlm_loss(model, input_ids, labels, mask_flags):
    """
    Compute causal masked language modeling loss.
    
    Args:
        input_ids: [B, L] - input tokens with [MASK]
        labels: [B, L] - ground truth tokens
        mask_flags: [B, L] - True where input is [MASK]
    """
    # Forward pass with causal attention
    logits = model(input_ids)  # [B, L, V]
    
    # Compute loss only on masked positions
    loss = F.cross_entropy(
        logits[mask_flags].view(-1, vocab_size),
        labels[mask_flags].view(-1),
        reduction='mean'
    )
    
    return loss

```

---

## 9. Theoretical Analysis: Why This Works

### Information-Theoretic Perspective

**Theorem 5** (Sufficient Statistics): *For left-to-right text generation, the left context provides sufficient information for predicting masked positions.*

**Formal Statement**: Let $x\_j$ be a token at position $j$. Under the assumption of left-to-right language structure:

```math
I(x_j ; x_{j+1:} \mid x_{1:j-1}) \approx 0

```

**Proof Sketch**:
1. Natural language follows left-to-right dependencies (subject before verb before object)
2. Information flows causally: earlier tokens constrain later tokens
3. Given complete left context, right context provides minimal additional information about the current position
$\square$

### Consistency with AR Generation

**Theorem 6** (Marginal Consistency): *WeDLM's parallel generation produces the same marginal distributions as AR generation under sufficient conditions.*

**Condition**: All masked positions are conditionally independent given the prefix:

```math
P(x_i, x_j \mid \text{prefix}) = P(x_i \mid \text{prefix}) \cdot P(x_j \mid \text{prefix})

```

**This holds because**:
1. Causal attention ensures each mask only sees the prefix
2. Positions are processed independently in the model
3. Sampling is done independently per position

**Corollary**: For structured outputs (math, code), where tokens are often deterministic given context, WeDLM's independence assumption is nearly exact.

### The Speed-Quality Tradeoff

**Theorem 7** (Speedup-Error Tradeoff): *Lower entropy threshold increases speedup but may increase error rate.*

Let $\tau$ be the entropy threshold. Define:
- $\bar{k}(\tau)$ = expected number of positions filled per step
- $\epsilon(\tau)$ = expected error rate

**Relation**:

```math
\bar{k}(\tau) \approx \sum_{j=1}^{W} P(H_j < \tau)
\epsilon(\tau) \approx \sum_{j=1}^{W} P(H_j < \tau) \cdot (1 - \exp(-\tau))

```

**Optimal Threshold**: Minimize combined cost:

```math
\tau^* = \arg\min_\tau \left[\frac{n}{\bar{k}(\tau)} \cdot c_{\text{time}} + \epsilon(\tau) \cdot n \cdot c_{\text{error}}\right]

```

---

## 10. Conclusion

### Summary of Key Innovations

1. **Topological Reordering**: Arrange mask and non-mask tokens to work with causal attention
2. **KV Cache Compatibility**: First diffusion LM to fully utilize production optimizations
3. **Streaming Parallel Decoding**: Commit prefixes continuously for efficiency
4. **Entropy-Based Selection**: Smart position selection for optimal speed-quality balance

### Mathematical Foundations

| Concept | Formulation |
|---------|-------------|
| Objective | $\mathcal{L} = -\sum\_{j \in \mathcal{M}} \log P\_\theta(x\_j \mid x\_{1:j-1} \cap \mathcal{M}^c)$ |
| Entropy | $H(P) = -\sum\_i p\_i \log p\_i$ |
| Adjusted Entropy | $\tilde{H}\_j = H(P\_j) + \lambda(j - j\_{\min})$ |
| Selection | $\text{Fill}(j) = \mathbb{1}[\tilde{H}\_j < \tau]$ |
| Speedup | $S = \mathbb{E}[\text{tokens per step}]$ |

### Why WeDLM Matters

| Aspect | Impact |
|--------|--------|
| **Speed** | 3-6× faster than vLLM for structured tasks |
| **Compatibility** | Works with FlashAttention, PagedAttention, CUDA Graphs |
| **Quality** | Matches or exceeds AR baselines |
| **Practicality** | Drop-in replacement for AR inference |

### The Big Picture

WeDLM bridges two worlds:
- **Theoretical elegance** of diffusion models (parallel generation)
- **Engineering efficiency** of AR models (optimized infrastructure)

By recognizing that mask positions don't need to see each other, WeDLM unlocks the speed benefits of parallel decoding while maintaining compatibility with the entire ecosystem of AR model optimizations.

---

## 📚 References

1. Liu, A., et al. (2025). *WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference*. arXiv:2512.22737.

2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

3. Austin, J., et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS.

4. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.

---

<div align="center">

**🎉 Thank you for reading!**

*For more details, see the [paper](https://arxiv.org/abs/2512.22737) and [project page](https://wedlm.github.io).*

</div>

