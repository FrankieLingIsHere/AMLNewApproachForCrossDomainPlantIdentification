"""Main training script for cross-domain plant classification."""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.hybrid_model import HybridModel, MultiStageHybridModel
from models.discriminator import compute_grl_lambda
from losses.combined_loss import HybridLoss
from data.dataset import create_dataloaders
from utils.logger import Logger, MetricTracker
from utils.evaluation import Evaluator, evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cross-Domain Plant Classification Training')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='Herbarium_Field',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--balance_domains', action='store_true',
                       help='Balance domains in batches')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='dinov2-vit-b',
                       choices=['dinov2-vit-s', 'dinov2-vit-b', 'dinov2-vit-l', 
                               'resnet50', 'efficientnet-b3'],
                       help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone during initial training')
    parser.add_argument('--unfreeze_epoch', type=int, default=5,
                       help='Epoch to unfreeze backbone')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--training_mode', type=str, default='multi_stage',
                       choices=['multi_stage', 'single_stage'],
                       help='Training mode')
    parser.add_argument('--start_stage', type=int, default=1,
                       choices=[1, 2, 3, 4],
                       help='Starting stage for multi-stage training')
    parser.add_argument('--end_stage', type=int, default=4,
                       choices=[1, 2, 3, 4],
                       help='Ending stage for multi-stage training (allows training specific stage ranges)')
    parser.add_argument('--epochs_per_stage', type=int, default=25,
                       help='Number of epochs per stage in multi-stage training')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Patience for early stopping')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Classification loss weight')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Contrastive loss weight')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='Prototypical loss weight')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='Domain adversarial loss weight')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive learning')
    
    # Augmentation settings
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                       choices=['weak', 'medium', 'strong'],
                       help='Augmentation strength')
    parser.add_argument('--use_mixup', action='store_true',
                       help='Use MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='MixUp alpha parameter')
    
    # Logging and checkpoints
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval (batches)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save interval (epochs)')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval (epochs)')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Use TensorBoard logging')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU device ID')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values (command line args take precedence)
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    arg_name = f"{key}_{subkey}" if key != 'training' else subkey
                    if hasattr(args, arg_name) and getattr(args, arg_name) == parser.get_default(arg_name):
                        setattr(args, arg_name, subvalue)
    
    return args


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args):
    """Create the hybrid model."""
    model = HybridModel(
        backbone_name=args.backbone,
        num_classes=100,
        num_paired_classes=60,
        num_unpaired_classes=40,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        gradient_checkpointing=args.gradient_checkpointing,
        dropout=args.dropout,
        learnable_prototypes=True
    )
    
    if args.training_mode == 'multi_stage':
        model = MultiStageHybridModel(model)
        model.set_stage(args.start_stage)
    
    return model


def create_optimizer(model, args):
    """Create optimizer."""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    return optimizer


def create_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    
    return scheduler


def load_class_indices(data_dir, class_to_idx=None):
    """
    Load paired and unpaired class indices.
    
    Args:
        data_dir: Path to data directory
        class_to_idx: Optional class to index mapping from dataset.
                     If provided, uses this mapping instead of creating a new one.
                     This ensures consistency with the dataset's label encoding.
    
    Returns:
        Tuple of (paired_indices, unpaired_indices)
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
    
    paired_indices = [class_to_idx[cls] for cls in paired_classes if cls in class_to_idx]
    unpaired_indices = [class_to_idx[cls] for cls in unpaired_classes if cls in class_to_idx]
    
    return paired_indices, unpaired_indices


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, 
                epoch, args, logger, current_stage):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker('loss', 'class_loss', 'contrast_loss', 'proto_loss', 'domain_loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images, labels, domains, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.mixed_precision:
            with autocast():
                outputs = model(images, return_features=True, return_projections=True)
                loss, loss_dict = criterion(outputs, labels, domains, stage=current_stage)
        else:
            outputs = model(images, return_features=True, return_projections=True)
            loss, loss_dict = criterion(outputs, labels, domains, stage=current_stage)
        
        # Backward pass
        if args.mixed_precision:
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        
        if args.scheduler == 'cosine':
            scheduler.step()
        
        # Update metrics
        metric_tracker.update(
            loss=loss.item(),
            class_loss=loss_dict.get('classification', 0.0),
            contrast_loss=loss_dict.get('contrastive', 0.0),
            proto_loss=loss_dict.get('prototypical', 0.0),
            domain_loss=loss_dict.get('domain_adversarial', 0.0)
        )
        
        # Update progress bar
        pbar.set_postfix(metric_tracker.get_metrics())
        
        # Log
        if batch_idx % args.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            logger.log_metrics(metric_tracker.get_metrics(), step, prefix='train/')
            logger.log_learning_rate(optimizer.param_groups[0]['lr'], step)
    
    return metric_tracker.get_metrics()


def main():
    args = parse_args()
    
    # Validate stage arguments
    if args.training_mode == 'multi_stage' and args.start_stage > args.end_stage:
        raise ValueError(f"start_stage ({args.start_stage}) cannot be greater than end_stage ({args.end_stage})")
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_id}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create logger
    logger = Logger(
        log_dir=args.output_dir,
        exp_name=args.exp_name,
        use_tensorboard=args.tensorboard
    )
    
    logger.info("=" * 80)
    logger.info("Cross-Domain Plant Classification Training")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")
    
    # Create dataloaders first
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_strength=args.augmentation_strength,
        balance_domains=args.balance_domains,
        image_size=args.image_size
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Get class_to_idx from dataset (authoritative source for label mapping)
    dataset_class_to_idx = train_loader.dataset.class_to_idx
    logger.info(f"Dataset has {len(dataset_class_to_idx)} classes")
    
    # Load class indices using the dataset's class_to_idx mapping
    # This ensures paired/unpaired indices match the actual labels from the dataset
    paired_indices, unpaired_indices = load_class_indices(args.data_dir, class_to_idx=dataset_class_to_idx)
    logger.info(f"Paired classes: {len(paired_indices)}, Unpaired classes: {len(unpaired_indices)}")
    
    # Debug: verify label distribution
    logger.info("Verifying dataset label distribution...")
    all_labels = []
    all_domains = []
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:  # Check first 10 batches
            break
        _, labels, domains, _ = batch
        all_labels.extend(labels.tolist())
        all_domains.extend(domains.tolist())
    
    unique_labels = set(all_labels)
    paired_count = sum(1 for l in all_labels if l in paired_indices)
    unpaired_count = sum(1 for l in all_labels if l in unpaired_indices)
    herbarium_count = sum(1 for d in all_domains if d == 0)
    field_count = sum(1 for d in all_domains if d == 1)
    
    logger.info(f"Sample from first 10 batches: {len(all_labels)} total samples")
    logger.info(f"  Unique labels: {len(unique_labels)}")
    logger.info(f"  Paired class samples: {paired_count}")
    logger.info(f"  Unpaired class samples: {unpaired_count}")
    logger.info(f"  Herbarium samples: {herbarium_count}")
    logger.info(f"  Field samples: {field_count}")

    
    # Create model
    logger.info(f"Creating model with backbone: {args.backbone}")
    model = create_model(args)
    model = model.to(device)
    
    # Create loss function
    criterion = HybridLoss(
        num_classes=100,
        num_paired_classes=60,
        num_unpaired_classes=40,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        temperature=args.temperature,
        paired_class_indices=paired_indices,
        unpaired_class_indices=unpaired_indices
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Create evaluator
    evaluator = Evaluator(
        num_classes=100,
        paired_class_indices=paired_indices,
        unpaired_class_indices=unpaired_indices
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume_from is not None and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Training mode: {args.training_mode}")
    if args.training_mode == 'multi_stage':
        logger.info(f"Stage range: {args.start_stage} to {args.end_stage}")
        logger.info(f"Epochs per stage: {args.epochs_per_stage}")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Update stage in multi-stage training
        current_stage = args.start_stage
        if args.training_mode == 'multi_stage':
            # Calculate which stage we're in based on epoch and epochs_per_stage
            # Only progress through stages in the requested range [start_stage, end_stage]
            num_stages_to_train = args.end_stage - args.start_stage + 1
            
            if num_stages_to_train == 1:
                # Training only one specific stage - stay in that stage
                current_stage = args.start_stage
            else:
                # Training multiple stages - distribute epochs across them
                epochs_per_stage_actual = args.epochs_per_stage
                stage_index = min(epoch // epochs_per_stage_actual, num_stages_to_train - 1)
                current_stage = args.start_stage + stage_index
            
            # Set the stage on the model at stage transitions
            if hasattr(model, 'set_stage'):
                # Check if we're at a stage boundary
                if num_stages_to_train > 1:
                    stage_boundary_epochs = [i * args.epochs_per_stage for i in range(num_stages_to_train)]
                    if epoch in stage_boundary_epochs:
                        model.set_stage(current_stage)
                        logger.info(f"Epoch {epoch}: Transitioning to stage {current_stage}")
                elif epoch == start_epoch:
                    # For single-stage training, set stage at the beginning
                    model.set_stage(current_stage)
                    logger.info(f"Epoch {epoch}: Training in stage {current_stage} only")
        
        # Update GRL lambda for domain adversarial training
        if hasattr(model, 'model'):
            lambda_val = compute_grl_lambda(epoch, args.epochs)
            model.model.set_domain_lambda(lambda_val)
        
        # Unfreeze backbone if needed
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            logger.info(f"Unfreezing backbone at epoch {epoch}")
            if hasattr(model, 'model'):
                model.model.unfreeze_backbone()
            else:
                model.unfreeze_backbone()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args, logger, current_stage
        )
        
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            logger.info("Evaluating...")
            eval_model = model.model if hasattr(model, 'model') else model
            test_metrics = evaluate_model(eval_model, test_loader, device, evaluator, return_features=True)
            logger.log_metrics(test_metrics, epoch, prefix='test/')
            logger.info(f"Epoch {epoch} - Test: {test_metrics}")
            
            # Check for improvement
            current_acc = test_metrics.get('accuracy', 0.0)
            is_best = logger.update_best_metric('test_accuracy', current_acc, mode='max')
            
            if is_best:
                best_acc = current_acc
                patience_counter = 0
                # Save best model
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'metrics': test_metrics
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if args.early_stopping and patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_model = model.model if hasattr(model, 'model') else model
    final_metrics = evaluate_model(eval_model, test_loader, device, evaluator, return_features=True)
    logger.info(f"Final Test Metrics: {final_metrics}")
    
    # Save final results
    evaluator.save_results(args.output_dir)
    logger.save_metrics()
    logger.save_best_metrics()
    
    logger.info("Training completed!")
    logger.close()


if __name__ == '__main__':
    main()
