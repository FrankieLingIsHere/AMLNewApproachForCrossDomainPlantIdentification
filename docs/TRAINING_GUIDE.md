# Training Guide: Tips and Best Practices

## Quick Start Configurations

### 1. Minimal Setup (Limited GPU)
```bash
python train.py \
    --backbone dinov2-vit-s \
    --batch_size 8 \
    --epochs 50 \
    --mixed_precision \
    --gradient_checkpointing \
    --augmentation_strength weak
```

**Expected:**
- Memory: ~4GB GPU
- Time: ~2-3 hours on single GPU
- Accuracy: ~60-65%

### 2. Standard Setup (Medium GPU)
```bash
python train.py \
    --backbone dinov2-vit-b \
    --batch_size 32 \
    --epochs 100 \
    --mixed_precision \
    --training_mode multi_stage \
    --early_stopping \
    --patience 15 \
    --tensorboard
```

**Expected:**
- Memory: ~12GB GPU
- Time: ~8-10 hours on single GPU
- Accuracy: ~68-72%

### 3. High Performance (Large GPU)
```bash
python train.py \
    --backbone dinov2-vit-l \
    --batch_size 64 \
    --epochs 100 \
    --mixed_precision \
    --gradient_checkpointing \
    --training_mode multi_stage \
    --augmentation_strength strong \
    --use_mixup \
    --balance_domains \
    --early_stopping
```

**Expected:**
- Memory: ~24GB GPU
- Time: ~12-15 hours on single GPU
- Accuracy: ~73-77%

## Hyperparameter Tuning

### Learning Rate

**Default**: `1e-4` (works well for most cases)

**Too high symptoms:**
- Loss spikes or diverges
- Unstable training
- Poor convergence

**Too low symptoms:**
- Very slow convergence
- Gets stuck in local minima
- Underfitting

**Recommended ranges:**
- DINOv2: `5e-5` to `2e-4`
- ResNet50: `1e-4` to `5e-4`
- EfficientNet: `5e-5` to `2e-4`

**Learning rate scheduling:**
```bash
# Cosine annealing (recommended)
--scheduler cosine

# Step decay
--scheduler step

# Reduce on plateau
--scheduler plateau
```

### Batch Size

**Trade-offs:**
- Larger: Better gradient estimates, faster training, more memory
- Smaller: More updates, can escape local minima, less memory

**Recommendations:**
| Backbone | Min Batch | Recommended | Max Batch |
|----------|-----------|-------------|-----------|
| DINOv2-ViT-S | 8 | 16-32 | 64 |
| DINOv2-ViT-B | 16 | 32-48 | 96 |
| DINOv2-ViT-L | 16 | 32-64 | 128 |
| ResNet50 | 16 | 32-64 | 128 |
| EfficientNet-B3 | 16 | 32-64 | 128 |

**Memory-saving tips:**
```bash
# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision

# Reduce batch size, increase accumulation
# (not implemented, but can be added)
```

### Loss Weights

**Default values:**
```
α (classification) = 1.0    # Base task
β (contrastive) = 0.5       # Cross-domain alignment
γ (prototypical) = 0.3      # Unpaired classes
δ (domain adversarial) = 0.1 # Domain confusion
```

**Tuning guidelines:**

**If unpaired classes perform poorly:**
```bash
--gamma 0.5  # Increase prototypical weight
```

**If domain gap is large:**
```bash
--beta 0.7   # Increase contrastive weight
--delta 0.2  # Increase domain adversarial weight
```

**If overall accuracy is low:**
```bash
--alpha 1.5  # Increase classification weight
```

**Balanced configuration:**
```bash
--alpha 1.0 --beta 0.5 --gamma 0.3 --delta 0.1
```

**Aggressive domain adaptation:**
```bash
--alpha 1.0 --beta 0.8 --gamma 0.4 --delta 0.3
```

### Temperature

**Contrastive learning temperature:**
- Default: `0.07`
- Lower (0.05): Sharper, more confident
- Higher (0.1): Smoother, more conservative

**Effect on training:**
```python
# Lower temperature
logits / 0.05  # More emphasis on hard negatives

# Higher temperature  
logits / 0.1   # More uniform weighting
```

**Recommendation**: Start with 0.07, adjust if:
- Training unstable → increase to 0.1
- Need sharper boundaries → decrease to 0.05

### Augmentation Strength

**Weak** (`--augmentation_strength weak`)
- Minimal transformations
- Faster training
- May underfit
- Use for: Quick experiments

**Medium** (`--augmentation_strength medium`) - **Recommended**
- Balanced augmentation
- Good generalization
- Standard training time
- Use for: Default training

**Strong** (`--augmentation_strength strong`)
- Heavy transformations
- Best generalization
- Slower training
- Use for: High-accuracy goals

**MixUp** (`--use_mixup --mixup_alpha 0.2`)
- Interpolates between samples
- Improves cross-domain transfer
- Slight slowdown
- Recommended for paired classes

## Training Strategies

### Multi-Stage vs Single-Stage

**Multi-Stage** (`--training_mode multi_stage`) - **Recommended**

Advantages:
- More stable convergence
- Better final accuracy
- Easier to debug
- Curriculum learning effect

Disadvantages:
- Fixed stage schedule
- Longer total training time

**Single-Stage** (`--training_mode single_stage`)

Advantages:
- Simpler
- Potentially faster convergence
- Flexible weighting from start

Disadvantages:
- Can be unstable
- Requires careful tuning
- May get stuck

**Custom stage durations:**
Modify in `train.py`:
```python
if epoch < 20:      # Shorter stage 1
    current_stage = 1
elif epoch < 40:    # Shorter stage 2
    current_stage = 2
# ...
```

### Backbone Freezing

**When to freeze:**
```bash
--freeze_backbone --unfreeze_epoch 10
```

Advantages:
- Faster initial training
- Prevents overfitting early
- Preserves pretrained features

**When NOT to freeze:**
- Small dataset
- Very different domain (herbarium)
- Fine-tuning required

**Recommendation:** 
- Freeze for first 5-10 epochs
- Then unfreeze for full adaptation

### Early Stopping

```bash
--early_stopping --patience 15
```

**How it works:**
- Monitors validation accuracy
- Stops if no improvement for N epochs
- Saves best model automatically

**Patience guidelines:**
- Small dataset: 10-15 epochs
- Large dataset: 15-20 epochs
- Aggressive: 5-10 epochs

**Tips:**
- Always use with multi-stage training
- Combine with checkpoint saving
- Monitor multiple metrics if possible

## Common Issues and Solutions

### 1. Out of Memory (OOM)

**Solutions:**
```bash
# Reduce batch size
--batch_size 16

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision

# Smaller backbone
--backbone dinov2-vit-s
```

### 2. Poor Convergence

**Symptoms:** Loss not decreasing

**Solutions:**
```bash
# Increase learning rate
--lr 2e-4

# Change optimizer
--optimizer adamw

# More epochs
--epochs 150

# Check data loading
# Verify augmentations are working
```

### 3. Overfitting

**Symptoms:** Train acc high, test acc low

**Solutions:**
```bash
# Stronger augmentation
--augmentation_strength strong

# Add MixUp
--use_mixup

# Increase dropout
--dropout 0.2

# Early stopping
--early_stopping --patience 10
```

### 4. Underfitting

**Symptoms:** Both train and test acc low

**Solutions:**
```bash
# Larger model
--backbone dinov2-vit-l

# More epochs
--epochs 150

# Higher learning rate
--lr 2e-4

# Weaker augmentation
--augmentation_strength weak
```

### 5. Domain Gap Still Large

**Symptoms:** Good on herbarium, poor on field

**Solutions:**
```bash
# Increase contrastive weight
--beta 0.8

# Increase domain adversarial weight
--delta 0.3

# Balance domains in batches
--balance_domains

# Stronger herbarium augmentation
--augmentation_strength strong
```

### 6. Unpaired Classes Performing Poorly

**Symptoms:** Low accuracy on 40 unpaired classes

**Solutions:**
```bash
# Increase prototypical weight
--gamma 0.5

# Use transductive refinement (in evaluate.py)
# Ensure sufficient herbarium data
```

## Monitoring Training

### TensorBoard

```bash
# Start training with TensorBoard
python train.py --tensorboard

# In another terminal
tensorboard --logdir results/tensorboard

# Open browser: http://localhost:6006
```

**What to monitor:**
1. **Loss curves**: Should decrease smoothly
2. **Learning rate**: Should follow schedule
3. **Accuracy**: Should increase over time
4. **Domain gap**: Should decrease (if logging)

### Console Output

```
Epoch 25 | train/loss: 0.8234 | train/class_loss: 0.5123 | ...
```

**Healthy training signs:**
- Losses decreasing
- Classification loss dominates early
- Contrastive loss becomes important in stage 2
- Steady improvement in accuracy

**Warning signs:**
- Loss oscillating wildly
- NaN or inf values
- No improvement after many epochs
- Accuracy stuck at random guess level

## Advanced Techniques

### Learning Rate Warmup

Currently implemented as warmup epochs. For custom warmup:

```python
# In train.py
def get_warmup_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr
```

### Gradient Accumulation

For simulating larger batch sizes:

```python
# Not currently implemented, but can add:
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Custom Loss Scheduling

Gradually adjust loss weights:

```python
# Example: Increase contrastive weight over time
beta = args.beta * min(1.0, epoch / 50)
criterion.update_weights(beta=beta)
```

## Reproducibility

**Ensure reproducible results:**

```bash
python train.py \
    --seed 42 \
    --device cuda \
    --device_id 0
```

**Note:** Perfect reproducibility on GPU is hard due to:
- CUDA non-determinism
- Parallel operations
- Floating-point arithmetic

**For research:**
- Run multiple seeds (42, 123, 456)
- Report mean and std dev
- Save all hyperparameters

## Experiment Tracking

### Recommended workflow

1. **Baseline run:**
```bash
python train.py --exp_name baseline
```

2. **Hyperparameter sweep:**
```bash
for lr in 5e-5 1e-4 2e-4; do
    python train.py --lr $lr --exp_name lr_$lr
done
```

3. **Compare results:**
```python
# Load metrics from results/*_best_metrics.json
# Compare test accuracy
```

### Logging best practices

- Use descriptive `--exp_name`
- Save config with results
- Track hardware and software versions
- Note any manual interventions

## Performance Optimization

### Speed up training

1. **More workers:**
```bash
--num_workers 8  # Match CPU cores
```

2. **Larger batch size:**
```bash
--batch_size 64  # If memory allows
```

3. **Mixed precision:**
```bash
--mixed_precision  # ~2x speedup
```

4. **Disable evaluation during training:**
```bash
--eval_interval 10  # Evaluate less frequently
```

### Reduce memory usage

1. **Gradient checkpointing:**
```bash
--gradient_checkpointing
```

2. **Smaller backbone:**
```bash
--backbone dinov2-vit-s
```

3. **Smaller batch size:**
```bash
--batch_size 16
```

## Troubleshooting Checklist

Before asking for help:

- [ ] Checked data loading (print batch shapes)
- [ ] Verified augmentations (visualize samples)
- [ ] Monitored loss curves
- [ ] Tried different learning rates
- [ ] Checked for NaN/inf in outputs
- [ ] Verified class balance
- [ ] Tested with smaller model first
- [ ] Read error messages carefully
- [ ] Checked GPU memory usage
- [ ] Verified checkpoint loading

## Recommended Workflow

1. **Quick sanity check** (10 min)
```bash
python train.py --backbone resnet50 --epochs 5 --batch_size 8
```

2. **Baseline run** (2-4 hours)
```bash
python train.py --backbone dinov2-vit-s --epochs 50
```

3. **Full training** (8-12 hours)
```bash
python train.py --backbone dinov2-vit-b --epochs 100 --training_mode multi_stage
```

4. **Best model** (12-24 hours)
```bash
python train.py --backbone dinov2-vit-l --epochs 100 --training_mode multi_stage --augmentation_strength strong --use_mixup
```

5. **Evaluation**
```bash
python evaluate.py --checkpoint_path checkpoints/best_model.pth
```
