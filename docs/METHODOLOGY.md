# Methodology: Cross-Domain Plant Classification

## Problem Statement

### Challenge
Plant classification across domains faces a significant domain gap:
- **Source Domain (Herbarium)**: Dried, flattened specimens with controlled backgrounds
- **Target Domain (Field)**: Natural photos with varying lighting, backgrounds, and perspectives
- **Asymmetry**: Only 60/100 classes have paired field images for training
- **Test Set**: Contains only field images for all 100 classes

### Goals
1. Learn domain-invariant features for 60 paired classes
2. Generalize to 40 unpaired classes using only herbarium images
3. Achieve high accuracy on field test images

## Approach Overview

Our hybrid approach combines three complementary learning paradigms:

```
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Model                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────┐    │
│  │  Backbone │───▶│ Classification│    │   Domain    │    │
│  │  (DINOv2, │    │     Head      │    │Discriminator│    │
│  │  ResNet,  │    └──────────────┘    │   (GRL)     │    │
│  │EfficientNet)│                       └─────────────┘    │
│  └───────────┘                                             │
│       │                                                     │
│       ├────────────────┬──────────────────┐                │
│       │                │                  │                │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Projection │  │Prototypical │  │   Features  │        │
│  │    Head    │  │   Network   │  │             │        │
│  │(Contrastive)│  │ (Unpaired) │  │             │        │
│  └────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Multi-Backbone Architecture

We support multiple backbones to accommodate different computational budgets:

**DINOv2 (Recommended)**
- Self-supervised vision transformer pre-trained on diverse images
- Strong transfer learning capabilities
- Variants: ViT-S/14 (384d), ViT-B/14 (768d), ViT-L/14 (1024d)
- Advantages: Excellent feature quality, robust to domain shift

**ResNet50**
- Classical CNN architecture
- Lightweight and fast
- 2048-dimensional features
- Advantages: Low memory, well-understood

**EfficientNet-B3**
- Efficient compound scaling
- 1536-dimensional features
- Advantages: Good accuracy/efficiency trade-off

### 2. Hybrid Learning Strategy

#### A. For Paired Classes (60 classes with herbarium + field)

**Supervised Contrastive Learning (SupCon)**
```
Goal: Align herbarium and field features in embedding space

Positive pairs: (herbarium, field) of same class
Negative pairs: Different classes

Loss = -log[exp(sim(anchor, pos)/τ) / Σexp(sim(anchor, neg)/τ)]

where:
  sim(·,·) = cosine similarity
  τ = temperature (0.07)
```

Key benefits:
- Pulls same-class features together across domains
- Pushes different-class features apart
- Learns domain-invariant representations

**Domain Adversarial Training**
```
Gradient Reversal Layer (GRL):
  Forward: f(x) = x
  Backward: ∂L/∂x = -λ·∂L/∂x

Domain discriminator tries to classify domain
Feature extractor tries to fool discriminator
→ Domain-invariant features
```

Lambda scheduling (DANN paper):
```
λ(p) = 2 / (1 + exp(-γ·p)) - 1
where p = epoch / max_epochs
```

#### B. For Unpaired Classes (40 classes, herbarium only)

**Prototypical Networks**
```
For each unpaired class k:
  Prototype: μ_k ∈ R^d (learnable or computed)
  
Classification via distance:
  P(y=k|x) ∝ exp(-d(f(x), μ_k) / τ)
  
where:
  f(x) = backbone features
  d(·,·) = cosine or Euclidean distance
  τ = temperature
```

**Transductive Refinement** (at test time):
```
Iteratively update prototypes using unlabeled test samples:
  
For t = 1 to T:
  1. Compute soft assignments: q_nk ∝ exp(-d(x_n, μ_k))
  2. Update prototypes: μ_k' = Σ_n q_nk·x_n / Σ_n q_nk
  3. Blend: μ_k = α·μ_k + (1-α)·μ_k'
```

Benefits:
- Adapts prototypes to test distribution
- Improves generalization without labels

### 3. Combined Loss Function

```
L_total = α·L_cls + β·L_contrast + γ·L_proto + δ·L_domain

Components:
1. L_cls: Cross-entropy for all 100 classes
2. L_contrast: SupCon for 60 paired classes
3. L_proto: Prototypical loss for 40 unpaired classes
4. L_domain: Domain adversarial loss (GRL)

Default weights:
  α = 1.0, β = 0.5, γ = 0.3, δ = 0.1
```

### 4. Multi-Stage Training

**Why multi-stage?**
- Gradual curriculum prevents early-stage confusion
- Stable convergence with complex objectives
- Better final performance than single-stage

**Stage 1: Warm-up (Epochs 0-24)**
```
Active: L_cls only
Goal: Learn basic visual features
Data: All herbarium images
```

**Stage 2: Contrastive (Epochs 25-49)**
```
Active: L_cls + L_contrast
Goal: Align herbarium-field features
Data: Both herbarium and field (paired classes)
```

**Stage 3: Prototypical (Epochs 50-74)**
```
Active: L_cls + L_contrast + L_proto
Goal: Handle unpaired classes
Data: All data
```

**Stage 4: Fine-tuning (Epochs 75-99)**
```
Active: All losses
Goal: Joint optimization with domain adaptation
Data: All data
```

## Domain-Specific Augmentation

### Herbarium Images (Strong Augmentation)
```python
Rationale: Simulate field conditions
- Random crop & resize
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian/Motion blur
- CoarseDropout (simulate occlusion)
- Rotation (±30°)
```

### Field Images (Light Augmentation)
```python
Rationale: Preserve natural appearance
- Random crop & resize
- Horizontal flip
- Mild color jitter
- Minimal rotation (±15°)
```

### MixUp/CutMix (Optional)
```
Mix herbarium and field images of same class:
  x_mixed = λ·x_herb + (1-λ)·x_field
  λ ~ Beta(α, α)

Benefits:
- Smooth decision boundaries
- Additional regularization
- Cross-domain interpolation
```

## Key Design Decisions

### 1. Why DINOv2?
- Self-supervised pre-training on 142M images
- No need for ImageNet labels
- Strong performance on fine-grained classification
- Robust to domain shift

### 2. Why Prototypical Networks for Unpaired Classes?
- Natural fit for few-shot/zero-shot scenarios
- Interpretable (distance-based)
- Can leverage transductive refinement
- No need for paired data

### 3. Why Multi-Stage Training?
- Prevents gradient conflicts early in training
- Each stage builds on previous foundations
- Empirically superior to joint training
- Easier to debug and tune

### 4. Why Domain Adversarial Learning?
- Explicit domain-invariance objective
- Proven effective for domain adaptation
- Complements contrastive learning
- Gradient reversal is elegant and efficient

## Theoretical Insights

### Domain Adaptation Perspective
```
Goal: Minimize target error ε_T

Upper bound (Ben-David et al.):
  ε_T ≤ ε_S + d_H(D_S, D_T) + λ*

where:
  ε_S = source error (herbarium)
  d_H = H-divergence (domain distance)
  λ* = ideal joint error

Our approach:
- L_cls minimizes ε_S
- L_contrast + L_domain minimize d_H
- L_proto handles unpaired classes
```

### Contrastive Learning Perspective
```
SupCon creates well-separated clusters:
- Same class, different domains → close
- Different classes → far
- Temperature τ controls concentration
```

### Metric Learning Perspective
```
Prototypical networks learn a metric space:
- Each class has a prototype
- Classification = nearest prototype
- Transduction refines metrics for test data
```

## Implementation Details

### Feature Normalization
```python
# L2 normalization before distance computation
features = F.normalize(features, dim=1)
prototypes = F.normalize(prototypes, dim=1)

# Ensures cosine similarity ∈ [-1, 1]
similarity = features @ prototypes.T
```

### Temperature Scaling
```python
# Sharpens probability distributions
logits = similarity / temperature

# Lower τ → more confident predictions
# Higher τ → smoother distributions
```

### Gradient Checkpointing
```python
# Trade computation for memory
# Essential for large models (DINOv2-ViT-L)
model.set_grad_checkpointing(True)
```

### Mixed Precision
```python
# AMP for faster training
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
```

## Evaluation Protocol

### Metrics
1. **Top-k Accuracy**: k ∈ {1, 3, 5}
2. **Per-Class Accuracy**: Fine-grained analysis
3. **F1-Score**: Macro and weighted
4. **Domain Gap**: Mean distance between herbarium/field features
5. **Confusion Matrix**: Visual error analysis

### Separate Analysis
- Paired classes (60): Measure cross-domain transfer
- Unpaired classes (40): Measure generalization
- Overall (100): Aggregate performance

### Domain Gap Computation
```python
For each paired class k:
  μ_herb = mean(herbarium features)
  μ_field = mean(field features)
  gap_k = ||μ_herb - μ_field||_2

Domain gap = mean(gap_k)
```

Lower gap → better domain alignment

## References

1. Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
2. Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
3. Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
4. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2023
