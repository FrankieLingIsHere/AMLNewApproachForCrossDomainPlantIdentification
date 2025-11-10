"""Combined loss function for the hybrid model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from losses.contrastive_loss import SupervisedContrastiveLoss
from losses.prototypical_loss import CombinedPrototypicalLoss


class HybridLoss(nn.Module):
    """
    Combined loss function for hybrid cross-domain plant classification.
    
    Combines:
    - Classification loss (cross-entropy for all 100 classes)
    - Contrastive loss (SupCon for paired classes with cross-domain alignment)
    - Prototypical loss (for unpaired classes)
    - Domain adversarial loss (for domain-invariant features)
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        num_paired_classes: int = 60,
        num_unpaired_classes: int = 40,
        alpha: float = 1.0,  # Classification weight
        beta: float = 0.5,   # Contrastive weight
        gamma: float = 0.3,  # Prototypical weight
        delta: float = 0.1,  # Domain adversarial weight
        temperature: float = 0.07,
        paired_class_indices: list = None,
        unpaired_class_indices: list = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_paired_classes = num_paired_classes
        self.num_unpaired_classes = num_unpaired_classes
        
        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Store class indices
        self.paired_class_indices = paired_class_indices or list(range(num_paired_classes))
        self.unpaired_class_indices = unpaired_class_indices or list(range(num_paired_classes, num_classes))
        
        # Individual loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.prototypical_loss = CombinedPrototypicalLoss(temperature=temperature)
        self.domain_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_labels: torch.Tensor = None,
        stage: int = 4
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs containing:
                - 'logits': Classification logits [B, num_classes]
                - 'features': Backbone features [B, feature_dim]
                - 'projections': Contrastive projections [B, proj_dim]
                - 'domain_pred': Domain predictions [B, 2]
                - 'proto_logits': Prototypical logits [B, num_unpaired_classes]
            labels: Ground truth class labels [B]
            domain_labels: Domain labels [B] (0=herbarium, 1=field)
            stage: Current training stage (1-4)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['logits'].device)
        
        # 1. Classification Loss (always active)
        class_loss = self.classification_loss(outputs['logits'], labels)
        loss_dict['classification'] = class_loss.item()
        total_loss += self.alpha * class_loss
        
        # 2. Contrastive Loss (stage >= 2, for paired classes only)
        if stage >= 2 and 'projections' in outputs and domain_labels is not None:
            # Filter for paired classes
            paired_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in self.paired_class_indices:
                paired_mask |= (labels == idx)
            
            if paired_mask.sum() > 1:  # Need at least 2 samples
                paired_projections = outputs['projections'][paired_mask]
                paired_labels = labels[paired_mask]
                paired_domains = domain_labels[paired_mask]
                
                contrastive_loss_val = self.contrastive_loss(
                    paired_projections, paired_labels, paired_domains
                )
                loss_dict['contrastive'] = contrastive_loss_val.item()
                total_loss += self.beta * contrastive_loss_val
            else:
                loss_dict['contrastive'] = 0.0
        else:
            loss_dict['contrastive'] = 0.0
        
        # 3. Prototypical Loss (stage >= 3, for unpaired classes only)
        if stage >= 3 and 'features' in outputs and 'proto_logits' in outputs:
            # Filter for unpaired classes
            unpaired_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in self.unpaired_class_indices:
                unpaired_mask |= (labels == idx)
            
            if unpaired_mask.sum() > 0:
                # Map global labels to local unpaired indices
                local_labels = torch.zeros_like(labels[unpaired_mask])
                for i, global_idx in enumerate(self.unpaired_class_indices):
                    local_labels[labels[unpaired_mask] == global_idx] = i
                
                # Use prototypical logits for unpaired classes
                proto_loss_val = self.classification_loss(
                    outputs['proto_logits'][unpaired_mask],
                    local_labels
                )
                loss_dict['prototypical'] = proto_loss_val.item()
                total_loss += self.gamma * proto_loss_val
            else:
                loss_dict['prototypical'] = 0.0
        else:
            loss_dict['prototypical'] = 0.0
        
        # 4. Domain Adversarial Loss (stage >= 4)
        if stage >= 4 and 'domain_pred' in outputs and domain_labels is not None:
            domain_loss_val = self.domain_loss(outputs['domain_pred'], domain_labels)
            loss_dict['domain_adversarial'] = domain_loss_val.item()
            total_loss += self.delta * domain_loss_val
        else:
            loss_dict['domain_adversarial'] = 0.0
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def update_weights(self, alpha: float = None, beta: float = None, 
                      gamma: float = None, delta: float = None):
        """Update loss weights."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta


if __name__ == "__main__":
    # Test combined loss
    print("Testing HybridLoss...")
    
    batch_size = 16
    num_classes = 100
    num_paired = 60
    num_unpaired = 40
    feature_dim = 768
    proj_dim = 256
    
    # Create sample outputs
    outputs = {
        'logits': torch.randn(batch_size, num_classes),
        'features': torch.randn(batch_size, feature_dim),
        'projections': torch.randn(batch_size, proj_dim),
        'domain_pred': torch.randn(batch_size, 2),
        'proto_logits': torch.randn(batch_size, num_unpaired)
    }
    
    labels = torch.randint(0, num_classes, (batch_size,))
    domain_labels = torch.randint(0, 2, (batch_size,))
    
    # Create loss function
    hybrid_loss = HybridLoss(
        num_classes=num_classes,
        num_paired_classes=num_paired,
        num_unpaired_classes=num_unpaired,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        delta=0.1
    )
    
    # Test different stages
    for stage in [1, 2, 3, 4]:
        print(f"\nStage {stage}:")
        total_loss, loss_dict = hybrid_loss(outputs, labels, domain_labels, stage=stage)
        print(f"  Total loss: {total_loss.item():.4f}")
        for key, value in loss_dict.items():
            if key != 'total':
                print(f"  {key}: {value:.4f}")
