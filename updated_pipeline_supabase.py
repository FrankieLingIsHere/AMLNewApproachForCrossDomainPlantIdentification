"""
Standalone training and evaluation pipeline with Supabase storage.
This is a complete standalone pipeline that uses Supabase for storing metrics and results.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
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

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase-py not installed. Install with: pip install supabase")


class SupabaseStorage:
    """Supabase storage for training metrics and results."""
    
    def __init__(self, url: str, key: str, table_prefix: str = 'ml_pipeline'):
        """Initialize Supabase storage.
        
        Args:
            url: Supabase project URL
            key: Supabase API key
            table_prefix: Prefix for table names
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py is not installed. Install with: pip install supabase")
        
        self.client: Client = create_client(url, key)
        self.table_prefix = table_prefix
        
        # Table names
        self.experiments_table = f"{table_prefix}_experiments"
        self.metrics_table = f"{table_prefix}_metrics"
        self.evaluations_table = f"{table_prefix}_evaluations"
        self.checkpoints_table = f"{table_prefix}_checkpoints"
        
        print(f"Connected to Supabase: {url}")
        print(f"Using table prefix: {table_prefix}")
    
    def create_experiment(self, exp_name: str, config: Dict) -> str:
        """Create a new experiment entry.
        
        Args:
            exp_name: Experiment name
            config: Experiment configuration
            
        Returns:
            Experiment ID (UUID from Supabase)
        """
        data = {
            'exp_name': exp_name,
            'backbone': config.get('backbone', 'unknown'),
            'batch_size': config.get('batch_size', 0),
            'learning_rate': config.get('lr', 0.0),
            'epochs': config.get('epochs', 0),
            'training_mode': config.get('training_mode', 'unknown'),
            'status': 'running',
            'config': json.dumps(config),
            'created_at': datetime.utcnow().isoformat()
        }
        
        response = self.client.table(self.experiments_table).insert(data).execute()
        
        if response.data and len(response.data) > 0:
            exp_id = response.data[0].get('id') or response.data[0].get('exp_name')
            return exp_id
        else:
            raise Exception("Failed to create experiment in Supabase")
    
    def log_metric(self, exp_id: str, epoch: int, stage: int, 
                   metric_name: str, metric_value: float, metric_type: str = 'train'):
        """Log a metric value.
        
        Args:
            exp_id: Experiment ID
            epoch: Current epoch
            stage: Current training stage
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (train/val/test)
        """
        data = {
            'exp_id': exp_id,
            'epoch': epoch,
            'stage': stage,
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'metric_type': metric_type,
            'recorded_at': datetime.utcnow().isoformat()
        }
        
        try:
            self.client.table(self.metrics_table).insert(data).execute()
        except Exception as e:
            print(f"Warning: Failed to log metric {metric_name}: {e}")
    
    def log_evaluation(self, exp_id: str, epoch: int, results: Dict):
        """Log evaluation results.
        
        Args:
            exp_id: Experiment ID
            epoch: Current epoch
            results: Dictionary of evaluation results
        """
        data = {
            'exp_id': exp_id,
            'epoch': epoch,
            'accuracy': float(results.get('accuracy', 0.0)),
            'top3_accuracy': float(results.get('top3_accuracy', 0.0)),
            'top5_accuracy': float(results.get('top5_accuracy', 0.0)),
            'f1_macro': float(results.get('f1_macro', 0.0)),
            'f1_weighted': float(results.get('f1_weighted', 0.0)),
            'paired_accuracy': float(results.get('paired_accuracy', 0.0)),
            'unpaired_accuracy': float(results.get('unpaired_accuracy', 0.0)),
            'results_json': json.dumps(results),
            'evaluated_at': datetime.utcnow().isoformat()
        }
        
        try:
            self.client.table(self.evaluations_table).insert(data).execute()
        except Exception as e:
            print(f"Warning: Failed to log evaluation: {e}")
    
    def save_checkpoint_info(self, exp_id: str, epoch: int, checkpoint_path: str, 
                           is_best: bool = False, metric_value: float = 0.0):
        """Save checkpoint information.
        
        Args:
            exp_id: Experiment ID
            epoch: Current epoch
            checkpoint_path: Path to checkpoint file
            is_best: Whether this is the best checkpoint
            metric_value: Metric value for this checkpoint
        """
        # Mark previous best as not best if this is best
        if is_best:
            try:
                # Update all previous checkpoints for this experiment
                self.client.table(self.checkpoints_table).update(
                    {'is_best': False}
                ).eq('exp_id', exp_id).execute()
            except Exception as e:
                print(f"Warning: Failed to update previous best checkpoints: {e}")
        
        data = {
            'exp_id': exp_id,
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'is_best': is_best,
            'metric_value': float(metric_value),
            'saved_at': datetime.utcnow().isoformat()
        }
        
        try:
            self.client.table(self.checkpoints_table).insert(data).execute()
        except Exception as e:
            print(f"Warning: Failed to save checkpoint info: {e}")
    
    def update_experiment_status(self, exp_id: str, status: str):
        """Update experiment status.
        
        Args:
            exp_id: Experiment ID
            status: New status (running/completed/failed)
        """
        try:
            self.client.table(self.experiments_table).update(
                {'status': status, 'updated_at': datetime.utcnow().isoformat()}
            ).eq('id', exp_id).execute()
        except Exception as e:
            # Try with exp_name if id doesn't work
            try:
                self.client.table(self.experiments_table).update(
                    {'status': status, 'updated_at': datetime.utcnow().isoformat()}
                ).eq('exp_name', exp_id).execute()
            except Exception as e2:
                print(f"Warning: Failed to update experiment status: {e2}")
    
    def get_experiment_metrics(self, exp_id: str) -> List[Dict]:
        """Get all metrics for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            List of metric dictionaries
        """
        try:
            response = self.client.table(self.metrics_table).select(
                'epoch, stage, metric_name, metric_value, metric_type, recorded_at'
            ).eq('exp_id', exp_id).order('epoch').execute()
            
            return response.data if response.data else []
        except Exception as e:
            print(f"Warning: Failed to get experiment metrics: {e}")
            return []
    
    def get_best_checkpoint(self, exp_id: str) -> Optional[Dict]:
        """Get best checkpoint for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Dictionary with checkpoint info or None
        """
        try:
            response = self.client.table(self.checkpoints_table).select(
                'epoch, checkpoint_path, metric_value, saved_at'
            ).eq('exp_id', exp_id).eq('is_best', True).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Warning: Failed to get best checkpoint: {e}")
            return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Standalone Pipeline with Supabase Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Supabase credentials
  python updated_pipeline_supabase.py \\
    --supabase_url https://xxx.supabase.co \\
    --supabase_key your-api-key \\
    --epochs 50

  # With environment variables (recommended for security)
  export SUPABASE_URL=https://xxx.supabase.co
  export SUPABASE_KEY=your-api-key
  python updated_pipeline_supabase.py --epochs 50
  
  # Full configuration
  python updated_pipeline_supabase.py \\
    --backbone dinov2-vit-b \\
    --batch_size 32 \\
    --epochs 100 \\
    --mixed_precision \\
    --exp_name my_experiment
        """
    )
    
    # Supabase settings
    parser.add_argument('--supabase_url', type=str, default=None,
                       help='Supabase project URL (or set SUPABASE_URL env var)')
    parser.add_argument('--supabase_key', type=str, default=None,
                       help='Supabase API key (or set SUPABASE_KEY env var)')
    parser.add_argument('--table_prefix', type=str, default='ml_pipeline',
                       help='Prefix for Supabase table names')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='Herbarium_Field',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='dinov2-vit-b',
                       choices=['dinov2-vit-s', 'dinov2-vit-b', 'dinov2-vit-l', 
                               'resnet50', 'efficientnet-b3'],
                       help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--training_mode', type=str, default='single_stage',
                       choices=['multi_stage', 'single_stage'],
                       help='Training mode')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use automatic mixed precision')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, 
                storage=None, exp_id=None, epoch=0, stage=1):
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for mixed precision
        storage: Supabase storage instance
        exp_id: Experiment ID
        epoch: Current epoch
        stage: Current training stage
        
    Returns:
        Dictionary of average metrics
    """
    model.train()
    tracker = MetricTracker('loss', 'cls_loss', 'accuracy')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images, labels, domains, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with autocast():
                outputs = model(images, return_features=True, return_projections=True)
                loss, loss_dict = criterion(outputs, labels, domains, stage=stage)
        else:
            outputs = model(images, return_features=True, return_projections=True)
            loss, loss_dict = criterion(outputs, labels, domains, stage=stage)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs['logits'], 1)
        accuracy = (preds == labels).float().mean().item()
        
        # Update metrics
        tracker.update(
            loss=loss.item(),
            cls_loss=loss_dict.get('classification_loss', 0.0),
            accuracy=accuracy
        )
        
        pbar.set_postfix({
            'loss': f"{tracker.meters['loss'].avg:.4f}",
            'acc': f"{tracker.meters['accuracy'].avg:.4f}"
        })
    
    metrics = tracker.get_metrics()
    
    # Log to Supabase
    if storage is not None and exp_id is not None:
        for metric_name, metric_value in metrics.items():
            storage.log_metric(exp_id, epoch, stage, metric_name, metric_value, 'train')
    
    return metrics


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Get Supabase credentials
    supabase_url = args.supabase_url or os.environ.get('SUPABASE_URL')
    supabase_key = args.supabase_key or os.environ.get('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("Error: Supabase credentials not provided!")
        print("Please provide --supabase_url and --supabase_key arguments")
        print("Or set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)
    
    # Initialize Supabase storage
    print("\n" + "="*70)
    print("Initializing Supabase Storage")
    print("="*70)
    
    try:
        storage = SupabaseStorage(supabase_url, supabase_key, args.table_prefix)
    except Exception as e:
        print(f"Error: Failed to connect to Supabase: {e}")
        print("\nPlease ensure:")
        print("1. Supabase URL and key are correct")
        print("2. supabase-py is installed: pip install supabase")
        print("3. Required tables exist in your Supabase project")
        sys.exit(1)
    
    # Create experiment
    exp_name = args.exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = vars(args)
    
    try:
        exp_id = storage.create_experiment(exp_name, config)
        print(f"✓ Experiment created: {exp_name}")
        print(f"  ID: {exp_id}")
    except Exception as e:
        print(f"Error: Failed to create experiment: {e}")
        sys.exit(1)
    
    try:
        # Create dataloaders
        print("\n" + "="*70)
        print("Loading Data")
        print("="*70)
        
        train_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentation_strength='medium',
            balance_domains=False,
            image_size=args.image_size
        )
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
        # Load class information
        class_to_idx = train_loader.dataset.class_to_idx
        paired_file = os.path.join(args.data_dir, 'list', 'class_with_pairs.txt')
        unpaired_file = os.path.join(args.data_dir, 'list', 'class_without_pairs.txt')
        
        with open(paired_file, 'r') as f:
            paired_classes = [line.strip() for line in f if line.strip()]
        with open(unpaired_file, 'r') as f:
            unpaired_classes = [line.strip() for line in f if line.strip()]
        
        paired_indices = [class_to_idx[cls] for cls in paired_classes if cls in class_to_idx]
        unpaired_indices = [class_to_idx[cls] for cls in unpaired_classes if cls in class_to_idx]
        
        num_classes = len(class_to_idx)
        print(f"✓ Classes: {num_classes} total ({len(paired_indices)} paired, {len(unpaired_indices)} unpaired)")
        
        # Create model
        print("\n" + "="*70)
        print("Creating Model")
        print("="*70)
        
        model = HybridModel(
            backbone_name=args.backbone,
            num_classes=num_classes,
            num_paired_classes=len(paired_indices),
            num_unpaired_classes=len(unpaired_indices),
            pretrained=args.pretrained,
            freeze_backbone=False,
            dropout=args.dropout
        )
        
        if args.training_mode == 'multi_stage':
            model = MultiStageHybridModel(model)
            model.set_stage(1)
        
        model = model.to(device)
        print(f"✓ Model: {args.backbone}")
        print(f"✓ Training mode: {args.training_mode}")
        
        # Create loss and optimizer
        criterion = HybridLoss(
            num_classes=num_classes,
            num_paired_classes=len(paired_indices),
            num_unpaired_classes=len(unpaired_indices),
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
            delta=0.1
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"✓ Optimizer: AdamW (lr={args.lr})")
        
        # Mixed precision scaler
        scaler = GradScaler() if args.mixed_precision else None
        if args.mixed_precision:
            print("✓ Mixed precision: enabled")
        
        # Create evaluator
        evaluator = Evaluator(
            num_classes=num_classes,
            paired_class_indices=paired_indices,
            unpaired_class_indices=unpaired_indices
        )
        
        # Training loop
        best_accuracy = 0.0
        print("\n" + "="*70)
        print(f"Training for {args.epochs} epochs")
        print("="*70)
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler, storage, exp_id, epoch, stage=1
            )
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                print("\nEvaluating...")
                model.eval()
                evaluator.reset()
                
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc='Evaluation'):
                        images, labels, _, _ = batch
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(images)
                        preds = torch.argmax(outputs['logits'], dim=1)
                        probs = torch.softmax(outputs['logits'], dim=1)
                        
                        evaluator.update(preds, labels, probs)
                
                eval_metrics = evaluator.compute_metrics()
                print(f"✓ Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"✓ F1 (Macro): {eval_metrics['f1_macro']:.4f}")
                
                # Log evaluation to Supabase
                storage.log_evaluation(exp_id, epoch, eval_metrics)
                
                # Save checkpoint if best
                if eval_metrics['accuracy'] > best_accuracy:
                    best_accuracy = eval_metrics['accuracy']
                    checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_name}_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_accuracy,
                    }, checkpoint_path)
                    storage.save_checkpoint_info(exp_id, epoch, checkpoint_path, 
                                               is_best=True, metric_value=best_accuracy)
                    print(f"✓ Saved best model: {checkpoint_path}")
        
        # Mark experiment as completed
        storage.update_experiment_status(exp_id, 'completed')
        
        print("\n" + "="*70)
        print("Training Completed!")
        print("="*70)
        print(f"✓ Best accuracy: {best_accuracy:.4f}")
        print(f"✓ Experiment: {exp_name}")
        print(f"✓ Results saved to Supabase")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        storage.update_experiment_status(exp_id, 'failed')
        raise


if __name__ == '__main__':
    main()
