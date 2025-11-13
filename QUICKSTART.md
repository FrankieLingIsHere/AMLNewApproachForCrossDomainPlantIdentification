# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Minimal Example
```bash
python train.py
```

### Common Configurations

**Low Compute (4GB GPU)**
```bash
python train.py --backbone dinov2-vit-s --batch_size 8 --mixed_precision --gradient_checkpointing
```

**Medium Compute (12GB GPU)** - Recommended
```bash
python train.py --backbone dinov2-vit-b --batch_size 32 --mixed_precision --early_stopping
```

**High Compute (24GB GPU)**
```bash
python train.py --backbone dinov2-vit-l --batch_size 64 --mixed_precision --use_mixup --augmentation_strength strong
```

### Stage-Specific Training

**Test a specific stage independently:**
```bash
# Test only stage 3 (prototypical learning)
python train.py --start_stage 3 --end_stage 3 --epochs 50 --batch_size 32

# Test only stage 4 (full model with domain adversarial)
python train.py --start_stage 4 --end_stage 4 --epochs 50
```

**Train a subset of stages:**
```bash
# Skip stage 1, train stages 2-4
python train.py --start_stage 2 --end_stage 4 --epochs 75 --epochs_per_stage 25

# Train only stages 1-2
python train.py --start_stage 1 --end_stage 2 --epochs 60 --epochs_per_stage 30
```

**Custom stage duration:**
```bash
# Train all stages with 30 epochs each
python train.py --start_stage 1 --end_stage 4 --epochs_per_stage 30 --epochs 120
```

## Evaluation

```bash
python evaluate.py --checkpoint_path checkpoints/best_model.pth --backbone dinov2-vit-b
```

## Key Arguments

### Model
- `--backbone`: dinov2-vit-s/b/l, resnet50, efficientnet-b3 (default: dinov2-vit-b)
- `--pretrained`: Use pretrained weights
- `--freeze_backbone`: Freeze backbone initially
- `--gradient_checkpointing`: Save memory

### Training
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--training_mode`: multi_stage or single_stage
- `--start_stage`: Starting stage (1-4, default: 1)
- `--end_stage`: Ending stage (1-4, default: 4)
- `--epochs_per_stage`: Epochs per stage (default: 25)
- `--mixed_precision`: Enable AMP
- `--early_stopping`: Stop when no improvement

### Loss Weights
- `--alpha`: Classification weight (default: 1.0)
- `--beta`: Contrastive weight (default: 0.5)
- `--gamma`: Prototypical weight (default: 0.3)
- `--delta`: Domain adversarial weight (default: 0.1)

### Augmentation
- `--augmentation_strength`: weak, medium, strong
- `--use_mixup`: Enable MixUp
- `--balance_domains`: Balance herbarium/field in batches

### Logging
- `--tensorboard`: Enable TensorBoard
- `--output_dir`: Results directory
- `--checkpoint_dir`: Checkpoints directory
- `--exp_name`: Experiment name

## Monitoring

### TensorBoard
```bash
# Start training with TensorBoard
python train.py --tensorboard

# In another terminal
tensorboard --logdir results/tensorboard
```

### Checkpoints
- Best model: `checkpoints/best_model.pth`
- Periodic: `checkpoints/checkpoint_epoch_N.pth`

### Results
After evaluation, check:
- `results/overall_metrics.json`
- `results/confusion_matrix.png`
- `results/per_class_accuracy.csv`
- `results/paired_classes_metrics.json`
- `results/unpaired_classes_metrics.json`

## Expected Performance

| Backbone | Top-1 Acc | Time (100 epochs) | Memory |
|----------|-----------|-------------------|--------|
| DINOv2-ViT-S | 64-66% | 4-5 hours | 8 GB |
| DINOv2-ViT-B | 69-72% | 8-10 hours | 12 GB |
| DINOv2-ViT-L | 73-77% | 14-16 hours | 20 GB |
| ResNet50 | 58-62% | 3-4 hours | 6 GB |
| EfficientNet-B3 | 61-65% | 5-6 hours | 10 GB |

## Troubleshooting

### Out of Memory
```bash
--batch_size 16 --gradient_checkpointing --backbone dinov2-vit-s
```

### Slow Training
```bash
--mixed_precision --num_workers 8
```

### Poor Accuracy
```bash
--epochs 150 --augmentation_strength strong --use_mixup
```

### Overfitting
```bash
--early_stopping --patience 10 --dropout 0.2
```

## Files Overview

```
train.py              # Main training script
evaluate.py           # Evaluation script
test_system.py        # System tests
README.md             # Full documentation
requirements.txt      # Dependencies
config/               # Configuration files
models/               # Model implementations
losses/               # Loss functions
data/                 # Dataset and augmentation
utils/                # Logging and evaluation
docs/                 # Detailed documentation
```

## Tips

1. **Start small**: Test with `--epochs 5` first
2. **Monitor**: Use `--tensorboard` to track training
3. **Save often**: Checkpoints saved every 5 epochs by default
4. **Use defaults**: Default hyperparameters work well
5. **Read docs**: Check docs/ for detailed guides

## Support

- See README.md for detailed usage
- See TRAINING_GUIDE.md for hyperparameter tuning
- See METHODOLOGY.md for technical details
- See RESULTS.md for expected performance
- Run `python train.py --help` for all options
