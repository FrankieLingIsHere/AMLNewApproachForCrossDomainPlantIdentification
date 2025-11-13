"""Evaluation script for cross-domain plant classification."""

import os
import argparse
import torch
from tqdm import tqdm

from models.hybrid_model import HybridModel
from data.dataset import create_dataloaders
from utils.evaluation import Evaluator, evaluate_model
from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cross-Domain Plant Classification Evaluation')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='Herbarium_Field',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='dinov2-vit-b',
                       choices=['dinov2-vit-s', 'dinov2-vit-b', 'dinov2-vit-l', 
                               'resnet50', 'efficientnet-b3'],
                       help='Backbone architecture')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Evaluation settings
    parser.add_argument('--eval_only', action='store_true',
                       help='Evaluation only mode')
    parser.add_argument('--save_confusion_matrix', action='store_true', default=True,
                       help='Save confusion matrix visualization')
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 3, 5],
                       help='Top-k values for accuracy computation')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU device ID')
    
    return parser.parse_args()


def load_class_indices(data_dir, class_to_idx=None):
    """
    Load paired and unpaired class indices.
    
    Args:
        data_dir: Path to data directory
        class_to_idx: Optional class to index mapping from dataset.
                     If provided, uses this mapping instead of creating a new one.
                     This ensures consistency with the dataset's label encoding.
    
    Returns:
        Tuple of (paired_indices, unpaired_indices, all_classes)
    """
    paired_file = os.path.join(data_dir, 'list', 'class_with_pairs.txt')
    unpaired_file = os.path.join(data_dir, 'list', 'class_without_pairs.txt')
    
    with open(paired_file, 'r') as f:
        paired_classes = [line.strip() for line in f if line.strip()]
    
    with open(unpaired_file, 'r') as f:
        unpaired_classes = [line.strip() for line in f if line.strip()]
    
    # Use provided class_to_idx or create one
    if class_to_idx is None:
        # Create mapping to indices (fallback for backward compatibility)
        all_classes = sorted(paired_classes + unpaired_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    else:
        # Get all classes from the provided mapping
        all_classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
    
    paired_indices = [class_to_idx[cls] for cls in paired_classes if cls in class_to_idx]
    unpaired_indices = [class_to_idx[cls] for cls in unpaired_classes if cls in class_to_idx]
    
    return paired_indices, unpaired_indices, all_classes


def load_model(checkpoint_path, backbone, device, dropout=0.1):
    """Load model from checkpoint."""
    # Create model
    model = HybridModel(
        backbone_name=backbone,
        num_classes=100,
        num_paired_classes=60,
        num_unpaired_classes=40,
        pretrained=False,  # We're loading from checkpoint
        freeze_backbone=False,
        gradient_checkpointing=False,
        dropout=dropout,
        learnable_prototypes=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle MultiStageHybridModel wrapper
    if 'model.model.backbone.backbone' in list(checkpoint['model_state_dict'].keys())[0]:
        # Checkpoint has MultiStageHybridModel wrapper
        from models.hybrid_model import MultiStageHybridModel
        wrapped_model = MultiStageHybridModel(model)
        wrapped_model.load_state_dict(checkpoint['model_state_dict'])
        model = wrapped_model.model
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_id}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create logger
    logger = Logger(
        log_dir=args.output_dir,
        exp_name='evaluation',
        use_tensorboard=False
    )
    
    logger.info("=" * 80)
    logger.info("Cross-Domain Plant Classification Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Backbone: {args.backbone}")
    
    # Create dataloaders first
    logger.info("Creating dataloaders...")
    _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_strength='weak',  # No augmentation for test
        balance_domains=False,
        image_size=args.image_size
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Get class_to_idx from dataset (authoritative source for label mapping)
    dataset_class_to_idx = test_loader.dataset.class_to_idx
    logger.info(f"Dataset has {len(dataset_class_to_idx)} classes")
    
    # Load class indices using the dataset's class_to_idx mapping
    # This ensures paired/unpaired indices match the actual labels from the dataset
    paired_indices, unpaired_indices, class_names = load_class_indices(args.data_dir, class_to_idx=dataset_class_to_idx)
    logger.info(f"Paired classes: {len(paired_indices)}, Unpaired classes: {len(unpaired_indices)}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint_path}")
    model, checkpoint = load_model(args.checkpoint_path, args.backbone, device, args.dropout)
    
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        logger.info(f"Best accuracy in checkpoint: {checkpoint['best_acc']:.4f}")
    
    # Create evaluator
    evaluator = Evaluator(
        num_classes=100,
        paired_class_indices=paired_indices,
        unpaired_class_indices=unpaired_indices,
        class_names=class_names,
        top_k=args.top_k
    )
    
    # Evaluate
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(
        model, test_loader, device, evaluator, return_features=True
    )
    
    # Log results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    
    for key, value in test_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save detailed results
    logger.info("\nSaving detailed results...")
    evaluator.save_results(args.output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"Top-3 Accuracy: {test_metrics.get('top3_accuracy', 0.0):.4f}")
    logger.info(f"Top-5 Accuracy: {test_metrics.get('top5_accuracy', 0.0):.4f}")
    logger.info(f"Paired Classes Accuracy: {test_metrics.get('paired_accuracy', 0.0):.4f}")
    logger.info(f"Unpaired Classes Accuracy: {test_metrics.get('unpaired_accuracy', 0.0):.4f}")
    logger.info(f"F1-Score (Macro): {test_metrics.get('f1_macro', 0.0):.4f}")
    logger.info(f"Domain Gap: {test_metrics.get('domain_gap', 0.0):.4f}")
    
    logger.info("\nResults saved to:")
    logger.info(f"  - {os.path.join(args.output_dir, 'overall_metrics.json')}")
    logger.info(f"  - {os.path.join(args.output_dir, 'per_class_accuracy.csv')}")
    logger.info(f"  - {os.path.join(args.output_dir, 'paired_classes_metrics.json')}")
    logger.info(f"  - {os.path.join(args.output_dir, 'unpaired_classes_metrics.json')}")
    logger.info(f"  - {os.path.join(args.output_dir, 'top_k_accuracy.json')}")
    if args.save_confusion_matrix:
        logger.info(f"  - {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    logger.info("\nEvaluation completed!")
    logger.close()


if __name__ == '__main__':
    main()
