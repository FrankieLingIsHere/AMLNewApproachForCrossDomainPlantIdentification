"""
Updated training and evaluation pipeline with SQLite storage.
This pipeline integrates training and evaluation with SQLite database for storing metrics and results.
"""

import os
import sys
import argparse
import sqlite3
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


class SQLiteStorage:
    """SQLite storage for training metrics and results."""
    
    def __init__(self, db_path: str = 'pipeline_results.db'):
        """Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database and create tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create experiments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_name TEXT UNIQUE NOT NULL,
                backbone TEXT,
                batch_size INTEGER,
                learning_rate REAL,
                epochs INTEGER,
                training_mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'running'
            )
        ''')
        
        # Create metrics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id INTEGER,
                epoch INTEGER,
                stage INTEGER,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (exp_id) REFERENCES experiments (id)
            )
        ''')
        
        # Create evaluation results table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id INTEGER,
                epoch INTEGER,
                accuracy REAL,
                top3_accuracy REAL,
                top5_accuracy REAL,
                f1_macro REAL,
                f1_weighted REAL,
                paired_accuracy REAL,
                unpaired_accuracy REAL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (exp_id) REFERENCES experiments (id)
            )
        ''')
        
        # Create checkpoints table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id INTEGER,
                epoch INTEGER,
                checkpoint_path TEXT,
                is_best BOOLEAN DEFAULT 0,
                metric_value REAL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (exp_id) REFERENCES experiments (id)
            )
        ''')
        
        self.conn.commit()
    
    def create_experiment(self, exp_name: str, config: Dict) -> int:
        """Create a new experiment entry.
        
        Args:
            exp_name: Experiment name
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        self.cursor.execute('''
            INSERT INTO experiments (exp_name, backbone, batch_size, learning_rate, 
                                   epochs, training_mode)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            exp_name,
            config.get('backbone', 'unknown'),
            config.get('batch_size', 0),
            config.get('lr', 0.0),
            config.get('epochs', 0),
            config.get('training_mode', 'unknown')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def log_metric(self, exp_id: int, epoch: int, stage: int, 
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
        self.cursor.execute('''
            INSERT INTO metrics (exp_id, epoch, stage, metric_name, metric_value, metric_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (exp_id, epoch, stage, metric_name, metric_value, metric_type))
        self.conn.commit()
    
    def log_evaluation(self, exp_id: int, epoch: int, results: Dict):
        """Log evaluation results.
        
        Args:
            exp_id: Experiment ID
            epoch: Current epoch
            results: Dictionary of evaluation results
        """
        self.cursor.execute('''
            INSERT INTO evaluation_results 
            (exp_id, epoch, accuracy, top3_accuracy, top5_accuracy, f1_macro, 
             f1_weighted, paired_accuracy, unpaired_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exp_id,
            epoch,
            results.get('accuracy', 0.0),
            results.get('top3_accuracy', 0.0),
            results.get('top5_accuracy', 0.0),
            results.get('f1_macro', 0.0),
            results.get('f1_weighted', 0.0),
            results.get('paired_accuracy', 0.0),
            results.get('unpaired_accuracy', 0.0)
        ))
        self.conn.commit()
    
    def save_checkpoint_info(self, exp_id: int, epoch: int, checkpoint_path: str, 
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
            self.cursor.execute('''
                UPDATE checkpoints SET is_best = 0 WHERE exp_id = ?
            ''', (exp_id,))
        
        self.cursor.execute('''
            INSERT INTO checkpoints (exp_id, epoch, checkpoint_path, is_best, metric_value)
            VALUES (?, ?, ?, ?, ?)
        ''', (exp_id, epoch, checkpoint_path, is_best, metric_value))
        self.conn.commit()
    
    def update_experiment_status(self, exp_id: int, status: str):
        """Update experiment status.
        
        Args:
            exp_id: Experiment ID
            status: New status (running/completed/failed)
        """
        self.cursor.execute('''
            UPDATE experiments SET status = ? WHERE id = ?
        ''', (status, exp_id))
        self.conn.commit()
    
    def get_experiment_metrics(self, exp_id: int) -> List[Dict]:
        """Get all metrics for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            List of metric dictionaries
        """
        self.cursor.execute('''
            SELECT epoch, stage, metric_name, metric_value, metric_type, recorded_at
            FROM metrics WHERE exp_id = ?
            ORDER BY epoch, metric_name
        ''', (exp_id,))
        
        columns = ['epoch', 'stage', 'metric_name', 'metric_value', 'metric_type', 'recorded_at']
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_best_checkpoint(self, exp_id: int) -> Optional[Dict]:
        """Get best checkpoint for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Dictionary with checkpoint info or None
        """
        self.cursor.execute('''
            SELECT epoch, checkpoint_path, metric_value, saved_at
            FROM checkpoints WHERE exp_id = ? AND is_best = 1
        ''', (exp_id,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'epoch': row[0],
                'checkpoint_path': row[1],
                'metric_value': row[2],
                'saved_at': row[3]
            }
        return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Updated Pipeline with SQLite Storage')
    
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
    
    # Storage settings
    parser.add_argument('--db_path', type=str, default='pipeline_results.db',
                       help='Path to SQLite database')
    
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
        storage: SQLite storage instance
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
    
    # Log to database
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
    
    # Initialize SQLite storage
    storage = SQLiteStorage(args.db_path)
    print(f"SQLite database: {args.db_path}")
    
    # Create experiment
    exp_name = args.exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = vars(args)
    exp_id = storage.create_experiment(exp_name, config)
    print(f"Experiment ID: {exp_id}, Name: {exp_name}")
    
    try:
        # Create dataloaders
        print("Loading data...")
        train_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentation_strength='medium',
            balance_domains=False,
            image_size=args.image_size
        )
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
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
        print(f"Classes: {num_classes} total ({len(paired_indices)} paired, {len(unpaired_indices)} unpaired)")
        
        # Create model
        print("Creating model...")
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
        print(f"Model: {args.backbone}")
        
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
        
        # Mixed precision scaler
        scaler = GradScaler() if args.mixed_precision else None
        
        # Create evaluator
        evaluator = Evaluator(
            num_classes=num_classes,
            paired_class_indices=paired_indices,
            unpaired_class_indices=unpaired_indices
        )
        
        # Training loop
        best_accuracy = 0.0
        print(f"\nStarting training for {args.epochs} epochs...")
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler, storage, exp_id, epoch, stage=1
            )
            
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
                print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"F1 (Macro): {eval_metrics['f1_macro']:.4f}")
                
                # Log evaluation to database
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
        print(f"\n{'='*70}")
        print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
        print(f"Results saved to database: {args.db_path}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        storage.update_experiment_status(exp_id, 'failed')
        raise
    
    finally:
        storage.close()


if __name__ == '__main__':
    main()
