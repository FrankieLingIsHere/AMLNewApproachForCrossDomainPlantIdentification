"""Evaluation utilities with comprehensive metrics and visualization."""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from tqdm import tqdm


class Evaluator:
    """
    Comprehensive evaluator for cross-domain plant classification.
    
    Computes:
    - Top-k accuracy (1, 3, 5)
    - Per-class accuracy
    - Confusion matrix
    - F1-score (macro and weighted)
    - Separate metrics for paired and unpaired classes
    - Domain gap measurement
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        paired_class_indices: List[int] = None,
        unpaired_class_indices: List[int] = None,
        class_names: List[str] = None,
        top_k: List[int] = [1, 3, 5]
    ):
        self.num_classes = num_classes
        self.paired_class_indices = paired_class_indices or list(range(60))
        self.unpaired_class_indices = unpaired_class_indices or list(range(60, 100))
        self.class_names = class_names
        self.top_k = top_k
        
        # Storage for predictions
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_domains = []
        self.all_features = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor = None,
        domains: torch.Tensor = None,
        features: torch.Tensor = None
    ):
        """
        Update evaluator with batch predictions.
        
        Args:
            predictions: Predicted class indices [B]
            labels: Ground truth labels [B]
            probabilities: Class probabilities [B, num_classes]
            domains: Domain labels [B]
            features: Feature vectors [B, feature_dim]
        """
        self.all_preds.append(predictions.cpu())
        self.all_labels.append(labels.cpu())
        
        if probabilities is not None:
            self.all_probs.append(probabilities.cpu())
        if domains is not None:
            self.all_domains.append(domains.cpu())
        if features is not None:
            self.all_features.append(features.cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Concatenate all predictions
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = (preds == labels).mean()
        
        # Top-k accuracy
        if len(self.all_probs) > 0:
            probs = torch.cat(self.all_probs).numpy()
            for k in self.top_k:
                top_k_preds = np.argsort(probs, axis=1)[:, -k:]
                top_k_acc = np.mean([labels[i] in top_k_preds[i] for i in range(len(labels))])
                metrics[f'top{k}_accuracy'] = top_k_acc
        
        # F1 scores
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Per-class accuracy
        per_class_acc = []
        for cls in range(self.num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        metrics['mean_per_class_accuracy'] = np.mean(per_class_acc)
        
        # Metrics for paired classes
        paired_mask = np.isin(labels, self.paired_class_indices)
        if paired_mask.sum() > 0:
            metrics['paired_accuracy'] = (preds[paired_mask] == labels[paired_mask]).mean()
            metrics['paired_f1_macro'] = f1_score(
                labels[paired_mask], preds[paired_mask], 
                average='macro', zero_division=0
            )
        
        # Metrics for unpaired classes
        unpaired_mask = np.isin(labels, self.unpaired_class_indices)
        if unpaired_mask.sum() > 0:
            metrics['unpaired_accuracy'] = (preds[unpaired_mask] == labels[unpaired_mask]).mean()
            metrics['unpaired_f1_macro'] = f1_score(
                labels[unpaired_mask], preds[unpaired_mask],
                average='macro', zero_division=0
            )
        
        # Domain gap (if features available)
        if len(self.all_features) > 0 and len(self.all_domains) > 0:
            features = torch.cat(self.all_features).numpy()
            domains = torch.cat(self.all_domains).numpy()
            domain_gap = self._compute_domain_gap(features, domains, labels)
            metrics['domain_gap'] = domain_gap
        
        return metrics
    
    def _compute_domain_gap(
        self,
        features: np.ndarray,
        domains: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute domain gap as average distance between herbarium and field features
        of the same class.
        
        Args:
            features: Feature vectors [N, feature_dim]
            domains: Domain labels [N]
            labels: Class labels [N]
            
        Returns:
            Average domain gap
        """
        gaps = []
        
        # Only consider paired classes
        for cls in self.paired_class_indices:
            herb_mask = (labels == cls) & (domains == 0)
            field_mask = (labels == cls) & (domains == 1)
            
            if herb_mask.sum() > 0 and field_mask.sum() > 0:
                herb_features = features[herb_mask]
                field_features = features[field_mask]
                
                # Compute mean features
                herb_mean = herb_features.mean(axis=0)
                field_mean = field_features.mean(axis=0)
                
                # Euclidean distance
                gap = np.linalg.norm(herb_mean - field_mean)
                gaps.append(gap)
        
        return np.mean(gaps) if gaps else 0.0
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()
        return confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
    
    def plot_confusion_matrix(
        self,
        save_path: str,
        normalize: bool = True,
        figsize: Tuple[int, int] = (20, 20),
        cmap: str = 'Blues'
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            cmap: Colormap
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        plt.figure(figsize=figsize)
        
        # Plot with reduced labels for better visibility
        if self.num_classes > 50:
            # Show every 5th label
            tick_marks = np.arange(0, self.num_classes, 5)
            labels = [str(i) if i % 5 == 0 else '' for i in range(self.num_classes)]
        else:
            tick_marks = np.arange(self.num_classes)
            labels = [str(i) for i in range(self.num_classes)]
        
        sns.heatmap(
            cm,
            annot=False,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            square=True,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_per_class_metrics(self) -> pd.DataFrame:
        """Get detailed per-class metrics."""
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()
        
        per_class_data = []
        
        for cls in range(self.num_classes):
            mask = labels == cls
            if mask.sum() == 0:
                continue
            
            cls_preds = preds[mask]
            cls_labels = labels[mask]
            
            accuracy = (cls_preds == cls_labels).mean()
            support = mask.sum()
            
            # Precision and recall
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_type = 'paired' if cls in self.paired_class_indices else 'unpaired'
            class_name = self.class_names[cls] if self.class_names else f'Class_{cls}'
            
            per_class_data.append({
                'class_id': cls,
                'class_name': class_name,
                'class_type': class_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            })
        
        return pd.DataFrame(per_class_data)
    
    def save_results(self, output_dir: str):
        """
        Save all evaluation results.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Overall metrics
        metrics = self.compute_metrics()
        with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Per-class metrics
        per_class_df = self.get_per_class_metrics()
        per_class_df.to_csv(os.path.join(output_dir, 'per_class_accuracy.csv'), index=False)
        
        # Paired classes metrics
        paired_df = per_class_df[per_class_df['class_type'] == 'paired']
        paired_metrics = {
            'mean_accuracy': paired_df['accuracy'].mean(),
            'mean_f1': paired_df['f1_score'].mean(),
            'mean_recall': paired_df['recall'].mean()
        }
        with open(os.path.join(output_dir, 'paired_classes_metrics.json'), 'w') as f:
            json.dump(paired_metrics, f, indent=2)
        
        # Unpaired classes metrics
        unpaired_df = per_class_df[per_class_df['class_type'] == 'unpaired']
        unpaired_metrics = {
            'mean_accuracy': unpaired_df['accuracy'].mean(),
            'mean_f1': unpaired_df['f1_score'].mean(),
            'mean_recall': unpaired_df['recall'].mean()
        }
        with open(os.path.join(output_dir, 'unpaired_classes_metrics.json'), 'w') as f:
            json.dump(unpaired_metrics, f, indent=2)
        
        # Top-k accuracy
        top_k_results = {f'top{k}': metrics.get(f'top{k}_accuracy', 0.0) for k in self.top_k}
        with open(os.path.join(output_dir, 'top_k_accuracy.json'), 'w') as f:
            json.dump(top_k_results, f, indent=2)
        
        # Confusion matrix
        self.plot_confusion_matrix(os.path.join(output_dir, 'confusion_matrix.png'))
        
        print(f"Results saved to {output_dir}")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    evaluator: Evaluator,
    return_features: bool = False
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run on
        evaluator: Evaluator instance
        return_features: Whether to extract and store features
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    evaluator.reset()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels, domains, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            
            # Forward pass
            outputs = model(images, return_features=return_features, return_projections=False)
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Update evaluator
            features = outputs.get('features') if return_features else None
            evaluator.update(preds, labels, probs, domains, features)
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    return metrics


if __name__ == "__main__":
    # Test evaluator
    print("Testing Evaluator...")
    
    num_classes = 100
    num_samples = 500
    
    # Create dummy predictions
    preds = torch.randint(0, num_classes, (num_samples,))
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = torch.randn(num_samples, num_classes).softmax(dim=1)
    domains = torch.randint(0, 2, (num_samples,))
    features = torch.randn(num_samples, 768)
    
    # Create evaluator
    evaluator = Evaluator(num_classes=num_classes)
    
    # Update in batches
    batch_size = 32
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        evaluator.update(
            preds[i:end_idx],
            labels[i:end_idx],
            probs[i:end_idx],
            domains[i:end_idx],
            features[i:end_idx]
        )
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test per-class metrics
    per_class_df = evaluator.get_per_class_metrics()
    print(f"\nPer-class metrics shape: {per_class_df.shape}")
    print(per_class_df.head())
    
    # Test saving results
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator.save_results(tmpdir)
        print(f"\nResults saved to {tmpdir}")
        print("Files:", os.listdir(tmpdir))
