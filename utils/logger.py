"""Logging utilities for training and evaluation."""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import json


class Logger:
    """
    Logger for training and evaluation.
    
    Supports:
    - Console logging
    - File logging
    - TensorBoard logging
    - Metrics tracking
    """
    
    def __init__(
        self,
        log_dir: str,
        exp_name: str = None,
        use_tensorboard: bool = True,
        log_to_file: bool = True
    ):
        self.log_dir = log_dir
        self.exp_name = exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_tensorboard = use_tensorboard
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logger
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_file = os.path.join(log_dir, f'{self.exp_name}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            tb_dir = os.path.join(log_dir, 'tensorboard', self.exp_name)
            self.writer = SummaryWriter(tb_dir)
        
        # Metrics storage
        self.metrics = {}
        self.best_metrics = {}
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        Log metrics to console, file, and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current step (epoch or iteration)
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Log to console
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f"Step {step} | {prefix}{metric_str}")
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{prefix}{key}', value, step)
        
        # Store metrics
        if step not in self.metrics:
            self.metrics[step] = {}
        self.metrics[step].update({f'{prefix}{k}': v for k, v in metrics.items()})
    
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        if self.writer is not None:
            self.writer.add_scalar('learning_rate', lr, step)
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """
        Log images to TensorBoard.
        
        Args:
            images: Dictionary of image names to tensors [C, H, W] or [B, C, H, W]
            step: Current step
        """
        if self.writer is not None:
            for name, img in images.items():
                if img.dim() == 3:
                    self.writer.add_image(name, img, step)
                elif img.dim() == 4:
                    self.writer.add_images(name, img, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram to TensorBoard."""
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard."""
        if self.writer is not None:
            try:
                self.writer.add_graph(model, input_tensor)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")
    
    def update_best_metric(self, metric_name: str, value: float, mode: str = 'max') -> bool:
        """
        Update best metric value.
        
        Args:
            metric_name: Name of the metric
            value: Current value
            mode: 'max' or 'min'
            
        Returns:
            True if this is a new best value
        """
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
            return True
        
        if mode == 'max':
            is_best = value > self.best_metrics[metric_name]
        else:
            is_best = value < self.best_metrics[metric_name]
        
        if is_best:
            self.best_metrics[metric_name] = value
        
        return is_best
    
    def save_metrics(self, filepath: str = None):
        """Save metrics to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f'{self.exp_name}_metrics.json')
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.info(f"Metrics saved to {filepath}")
    
    def save_best_metrics(self, filepath: str = None):
        """Save best metrics to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f'{self.exp_name}_best_metrics.json')
        
        with open(filepath, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
        
        self.info(f"Best metrics saved to {filepath}")
    
    def close(self):
        """Close logger and writer."""
        if self.writer is not None:
            self.writer.close()
        
        # Remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class MetricTracker:
    """Track multiple metrics using AverageMeter."""
    
    def __init__(self, *metric_names):
        self.meters = {name: AverageMeter(name) for name in metric_names}
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key in self.meters:
                if isinstance(value, tuple):
                    self.meters[key].update(value[0], value[1])
                else:
                    self.meters[key].update(value)
    
    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get average values of all metrics."""
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def __str__(self):
        return ' | '.join([str(meter) for meter in self.meters.values()])


if __name__ == "__main__":
    # Test logger
    print("Testing Logger...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir, exp_name='test_exp', use_tensorboard=False)
        
        # Test logging
        logger.info("This is an info message")
        logger.warning("This is a warning")
        
        # Test metrics
        for epoch in range(5):
            metrics = {
                'loss': 1.0 / (epoch + 1),
                'accuracy': 0.5 + 0.1 * epoch
            }
            logger.log_metrics(metrics, epoch, prefix='train/')
        
        # Test best metric
        is_best = logger.update_best_metric('accuracy', 0.9, mode='max')
        print(f"Is best: {is_best}")
        
        # Save metrics
        logger.save_metrics()
        logger.save_best_metrics()
        
        logger.close()
    
    # Test AverageMeter
    print("\nTesting AverageMeter...")
    meter = AverageMeter('loss')
    for i in range(10):
        meter.update(1.0 / (i + 1))
    print(meter)
    
    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker('loss', 'accuracy', 'f1')
    for i in range(5):
        tracker.update(loss=1.0 / (i + 1), accuracy=0.5 + 0.1 * i, f1=0.6 + 0.08 * i)
    print(tracker)
    print(tracker.get_metrics())
