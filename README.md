# Cross-Domain Plant Classification for Herbarium-Field Dataset

A complete PyTorch implementation for cross-domain plant classification, addressing the domain gap between herbarium specimens (dried, flattened) and field images (natural photos).

## 🌟 Features

- **Multi-Backbone Support**: DINOv2 (ViT-S/B/L), ResNet50, EfficientNet-B3
- **Hybrid Learning**: Combines supervised, contrastive, and prototypical learning
- **Domain Adaptation**: Gradient Reversal Layer for domain-invariant features
- **Flexible Training**: Multi-stage or single-stage training modes
- **Comprehensive Evaluation**: Top-k accuracy, confusion matrix, per-class metrics
- **Production Ready**: Mixed precision, gradient checkpointing, early stopping

## 📊 Dataset Structure

```
Herbarium_Field/
├── train/
│   ├── herbarium/     # 100 classes (all plants)
│   └── photo/         # 60 classes (paired plants with field images)
├── test/
│   └── *.jpg          # Field images only (all 100 classes)
└── list/
    ├── class_with_pairs.txt      # 60 paired class IDs
    ├── class_without_pairs.txt   # 40 unpaired class IDs
    ├── train.txt                 # Training samples
    ├── test.txt                  # Test samples
    └── species_list.txt          # Class names
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AMLNewApproachForCrossDomainPlantIdentification

# Install dependencies
pip install -r requirements.txt
```

### Training

**Lightweight (Low Compute)**
```bash
python train.py \
    --backbone dinov2-vit-s \
    --batch_size 16 \
    --mixed_precision \
    --gradient_checkpointing \
    --epochs 50
```

**Recommended (Medium Compute)**
```bash
python train.py \
    --backbone dinov2-vit-b \
    --batch_size 32 \
    --mixed_precision \
    --training_mode multi_stage \
    --epochs 100 \
    --early_stopping \
    --patience 15
```

**High Performance (High Compute)**
```bash
python train.py \
    --backbone dinov2-vit-l \
    --batch_size 64 \
    --mixed_precision \
    --training_mode multi_stage \
    --epochs 100 \
    --augmentation_strength strong \
    --use_mixup
```

### Evaluation

```bash
python evaluate.py \
    --backbone dinov2-vit-b \
    --checkpoint_path checkpoints/best_model.pth \
    --save_confusion_matrix
```

## 🏗️ Architecture

### Model Components

1. **Backbone** (`models/backbone.py`)
   - DINOv2-ViT-S/B/L (384/768/1024 dims)
   - ResNet50 (2048 dims)
   - EfficientNet-B3 (1536 dims)

2. **Hybrid Model** (`models/hybrid_model.py`)
   - Classification head (100 classes)
   - Contrastive projection head
   - Prototypical network (40 unpaired classes)
   - Domain discriminator

3. **Loss Functions** (`losses/`)
   - **Classification**: Cross-entropy for all 100 classes
   - **Contrastive**: SupCon for 60 paired classes (herbarium-field alignment)
   - **Prototypical**: Distance-based for 40 unpaired classes
   - **Domain Adversarial**: GRL-based domain confusion

### Multi-Stage Training

**Stage 1** (Epochs 0-24): Classification only
- Train on herbarium images
- Learn basic feature representations

**Stage 2** (Epochs 25-49): Add contrastive learning
- Align herbarium and field features for paired classes
- Encourage cross-domain similarity

**Stage 3** (Epochs 50-74): Add prototypical learning
- Handle unpaired classes with prototype-based classification
- Refine prototypes transductively

**Stage 4** (Epochs 75-99): Full end-to-end
- Enable domain adversarial training
- Fine-tune all components together

## 📋 Command Line Arguments

### Data
- `--data_dir`: Dataset directory (default: `Herbarium_Field`)
- `--batch_size`: Batch size (default: `32`)
- `--num_workers`: Data loading workers (default: `4`)
- `--balance_domains`: Balance herbarium/field in batches
- `--image_size`: Input size (default: `224`)

### Model
- `--backbone`: Architecture choice (default: `dinov2-vit-b`)
- `--pretrained`: Use pretrained weights
- `--freeze_backbone`: Freeze backbone initially
- `--unfreeze_epoch`: Epoch to unfreeze (default: `5`)
- `--gradient_checkpointing`: Enable memory-efficient training
- `--dropout`: Dropout rate (default: `0.1`)

### Training
- `--epochs`: Total epochs (default: `100`)
- `--lr`: Learning rate (default: `1e-4`)
- `--optimizer`: Optimizer choice (default: `adamw`)
- `--scheduler`: LR scheduler (default: `cosine`)
- `--training_mode`: `multi_stage` or `single_stage`
- `--mixed_precision`: Use AMP for faster training
- `--early_stopping`: Enable early stopping
- `--patience`: Early stopping patience (default: `15`)

### Loss Weights
- `--alpha`: Classification weight (default: `1.0`)
- `--beta`: Contrastive weight (default: `0.5`)
- `--gamma`: Prototypical weight (default: `0.3`)
- `--delta`: Domain adversarial weight (default: `0.1`)
- `--temperature`: Contrastive temperature (default: `0.07`)

### Augmentation
- `--augmentation_strength`: `weak`, `medium`, or `strong`
- `--use_mixup`: Enable MixUp augmentation
- `--mixup_alpha`: MixUp alpha (default: `0.2`)

### Logging
- `--output_dir`: Results directory (default: `results`)
- `--checkpoint_dir`: Checkpoints directory (default: `checkpoints`)
- `--log_interval`: Log every N batches (default: `10`)
- `--save_interval`: Save every N epochs (default: `5`)
- `--eval_interval`: Evaluate every N epochs (default: `5`)
- `--tensorboard`: Use TensorBoard logging
- `--exp_name`: Experiment name

### Device
- `--device`: `cuda` or `cpu`
- `--device_id`: GPU ID (default: `0`)
- `--seed`: Random seed (default: `42`)

### Resume
- `--resume_from`: Checkpoint path to resume training
- `--config`: YAML config file path

## 📊 Evaluation Metrics

The evaluation script computes and saves:

### Overall Metrics (`overall_metrics.json`)
- Top-1, Top-3, Top-5 accuracy
- F1-score (macro and weighted)
- Mean per-class accuracy
- Domain gap measurement

### Per-Class Metrics (`per_class_accuracy.csv`)
- Class ID, name, type (paired/unpaired)
- Accuracy, precision, recall, F1-score
- Support (number of samples)

### Paired Classes (`paired_classes_metrics.json`)
- Mean accuracy, F1, recall for 60 paired classes

### Unpaired Classes (`unpaired_classes_metrics.json`)
- Mean accuracy, F1, recall for 40 unpaired classes

### Confusion Matrix (`confusion_matrix.png`)
- 100x100 heatmap visualization

## 🔧 Configuration File

You can use a YAML config file instead of command-line arguments:

```yaml
# config/my_experiment.yaml
data:
  batch_size: 32
  augmentation_strength: medium

model:
  backbone: dinov2-vit-b
  pretrained: true

training:
  epochs: 100
  lr: 0.0001
  training_mode: multi_stage

loss:
  alpha: 1.0
  beta: 0.5
  gamma: 0.3
  delta: 0.1
```

```bash
python train.py --config config/my_experiment.yaml
```

## 📈 Expected Results

| Backbone | Top-1 Acc | Top-5 Acc | Paired Acc | Unpaired Acc |
|----------|-----------|-----------|------------|--------------|
| DINOv2-ViT-S | ~65% | ~85% | ~70% | ~55% |
| DINOv2-ViT-B | ~70% | ~88% | ~75% | ~60% |
| DINOv2-ViT-L | ~75% | ~90% | ~80% | ~65% |
| ResNet50 | ~60% | ~82% | ~65% | ~50% |
| EfficientNet-B3 | ~62% | ~84% | ~68% | ~52% |

*Note: Actual results depend on training hyperparameters and random initialization.*

## 🛠️ Advanced Features

### Gradient Checkpointing
```bash
python train.py --gradient_checkpointing
```
Reduces memory usage for large models (DINOv2-ViT-L).

### Mixed Precision Training
```bash
python train.py --mixed_precision
```
Speeds up training by ~2x with minimal accuracy loss.

### Domain Balancing
```bash
python train.py --balance_domains
```
Ensures equal herbarium/field samples in each batch.

### Resume Training
```bash
python train.py --resume_from checkpoints/checkpoint_epoch_50.pth
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{cross_domain_plant_classification,
  title={Cross-Domain Plant Classification for Herbarium-Field Dataset},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/repo}
}
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- DINOv2: Meta AI Research
- PyTorch Team
- Herbarium-Field Dataset Contributors
