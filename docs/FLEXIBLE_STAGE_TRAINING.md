# Flexible Stage Training - Quick Start Guide

This document demonstrates the new flexible stage training feature that allows you to test different training stages independently without waiting for previous stages to complete.

## What Changed?

Previously, multi-stage training was hardcoded to progress through all stages sequentially:
- Epochs 0-24: Stage 1
- Epochs 25-49: Stage 2
- Epochs 50-74: Stage 3
- Epochs 75-99: Stage 4

Now, you can:
- Start at any stage (1-4)
- End at any stage (1-4)
- Control how many epochs each stage gets
- Train a single stage repeatedly

## New Command Line Arguments

### `--start_stage` (1-4, default: 1)
The stage to start training from.

### `--end_stage` (1-4, default: 4) ✨ NEW
The stage to end training at. Allows you to train specific stage ranges.

### `--epochs_per_stage` (default: 25) ✨ NEW
Number of epochs to spend in each stage before transitioning to the next.

## Quick Examples

### 1. Test a Single Stage
```bash
# Test only Stage 3 (prototypical learning) for 50 epochs
python train.py --start_stage 3 --end_stage 3 --epochs 50

# Test only Stage 4 (full model) for 100 epochs
python train.py --start_stage 4 --end_stage 4 --epochs 100
```

**Use case**: Testing a specific component in isolation, debugging, or when you have a pretrained model from earlier stages.

### 2. Skip Earlier Stages
```bash
# Train only stages 2-4 (skip classification-only stage)
python train.py --start_stage 2 --end_stage 4 --epochs 75 --epochs_per_stage 25

# Train only stages 1-2 (focus on supervised and contrastive learning)
python train.py --start_stage 1 --end_stage 2 --epochs 60 --epochs_per_stage 30
```

**Use case**: When you want to focus on specific learning objectives or have time constraints.

### 3. Adjust Stage Duration
```bash
# Give each stage 30 epochs instead of 25
python train.py --epochs_per_stage 30 --epochs 120

# Quick experimentation with 5 epochs per stage
python train.py --epochs_per_stage 5 --epochs 20
```

**Use case**: Allowing more or less time for each stage to converge based on your observations.

### 4. Resume from Checkpoint at Different Stage
```bash
# Resume from checkpoint and train only stage 4
python train.py --resume_from checkpoints/stage2_model.pth \
  --start_stage 4 --end_stage 4 --epochs 50
```

**Use case**: You've trained stages 1-2, now you want to test the full model.

## Understanding Stage Transitions

### Example: Training Stages 2-4 with 25 epochs each
```bash
python train.py --start_stage 2 --end_stage 4 --epochs_per_stage 25 --epochs 75
```

**What happens:**
- Epochs 0-24: Stage 2 (Classification + Contrastive)
- Epochs 25-49: Stage 3 (+ Prototypical)
- Epochs 50-74: Stage 4 (+ Domain Adversarial)

### Example: Single Stage Training
```bash
python train.py --start_stage 3 --end_stage 3 --epochs 100
```

**What happens:**
- Epochs 0-99: Stage 3 (Classification + Contrastive + Prototypical)
- No stage transitions - stays in Stage 3 the entire time

## Stage Descriptions

### Stage 1: Foundation
- **Active**: Classification loss only
- **Purpose**: Learn basic feature representations from herbarium images
- **Best for**: Establishing a baseline supervised model

### Stage 2: Cross-Domain Alignment
- **Active**: Classification + Contrastive loss
- **Purpose**: Align herbarium and field features for paired classes
- **Best for**: Learning domain-invariant representations

### Stage 3: Handle Unpaired Classes
- **Active**: Classification + Contrastive + Prototypical loss
- **Purpose**: Extend to unpaired classes using prototypical learning
- **Best for**: Dealing with classes that only have herbarium samples

### Stage 4: Full Model
- **Active**: All losses (Classification + Contrastive + Prototypical + Domain Adversarial)
- **Purpose**: Fine-tune everything end-to-end with domain confusion
- **Best for**: Maximum performance and domain adaptation

## Configuration File Usage

You can also set these in a YAML config file:

```yaml
# config/my_experiment.yaml
training:
  epochs: 75
  training_mode: multi_stage
  start_stage: 2
  end_stage: 4
  epochs_per_stage: 25
```

Then run:
```bash
python train.py --config config/my_experiment.yaml
```

## Common Use Cases

### 1. Debugging a Specific Stage
```bash
# Something seems wrong with prototypical learning? Test it in isolation:
python train.py --start_stage 3 --end_stage 3 --epochs 30 --batch_size 16
```

### 2. Time-Constrained Training
```bash
# Only have 2 hours? Focus on the most important stages:
python train.py --start_stage 3 --end_stage 4 --epochs 50 --epochs_per_stage 25
```

### 3. Ablation Studies
```bash
# Test without domain adversarial loss:
python train.py --start_stage 1 --end_stage 3 --epochs 75

# Test without prototypical learning:
python train.py --start_stage 1 --end_stage 2 --epochs 50
```

### 4. Rapid Experimentation
```bash
# Quickly test all stages with minimal epochs:
python train.py --epochs_per_stage 3 --epochs 12 --batch_size 8
```

## Tips

1. **Monitor TensorBoard**: Use `--tensorboard` to visualize how different losses activate in each stage
2. **Save checkpoints**: Use `--save_interval 5` to save models at different stages
3. **Combine with other options**: All existing options (batch size, optimizer, etc.) work normally
4. **Start small**: Try 5-10 epochs per stage first to verify your configuration

## Getting Help

Run `python train.py --help` to see all available options.

For detailed examples with output explanations, see `docs/stage_training_examples.sh`.
