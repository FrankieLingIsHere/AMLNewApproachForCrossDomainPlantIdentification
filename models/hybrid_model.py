"""
Hybrid model integrating:
- Backbone (DINOv2, ResNet, EfficientNet)
- Classification head
- Contrastive learning head
- Prototypical network for unpaired classes
- Domain discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.backbone import BackboneWrapper
from models.discriminator import DomainDiscriminator
from models.prototypical import PrototypicalNetwork


class HybridModel(nn.Module):
    """
    Hybrid model for cross-domain plant classification.
    
    Handles both paired classes (herbarium + field) and unpaired classes (herbarium only).
    """
    
    def __init__(
        self,
        backbone_name: str = 'dinov2-vit-b',
        num_classes: int = 100,
        num_paired_classes: int = 60,
        num_unpaired_classes: int = 40,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        dropout: float = 0.1,
        learnable_prototypes: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_paired_classes = num_paired_classes
        self.num_unpaired_classes = num_unpaired_classes
        
        # Backbone
        self.backbone = BackboneWrapper(
            backbone_name=backbone_name,
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing,
            freeze=freeze_backbone
        )
        feature_dim = self.backbone.feature_dim
        
        # Feature projection for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 256)  # Projection dimension
        )
        
        # Classification head (for all 100 classes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Prototypical network for unpaired classes
        self.prototypical_net = PrototypicalNetwork(
            feature_dim=feature_dim,
            num_unpaired_classes=num_unpaired_classes,
            learnable_prototypes=learnable_prototypes
        )
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=1024
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_projections: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input images [B, C, H, W]
            return_features: Whether to return backbone features
            return_projections: Whether to return contrastive projections
            
        Returns:
            Dictionary containing:
                - 'logits': Classification logits [B, num_classes]
                - 'features': Backbone features [B, feature_dim] (if requested)
                - 'projections': Contrastive projections [B, 256] (if requested)
                - 'domain_pred': Domain predictions [B, 2]
                - 'proto_logits': Prototypical logits for unpaired classes [B, num_unpaired_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Domain prediction
        domain_pred = self.domain_discriminator(features)
        
        # Prototypical logits (for unpaired classes)
        proto_logits = self.prototypical_net(features)
        
        outputs = {
            'logits': logits,
            'domain_pred': domain_pred,
            'proto_logits': proto_logits
        }
        
        if return_features:
            outputs['features'] = features
        
        if return_projections:
            projections = self.projection_head(features)
            projections = F.normalize(projections, dim=1)
            outputs['projections'] = projections
        
        return outputs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        return self.backbone(x)
    
    def get_projections(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized projections for contrastive learning."""
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        self.backbone.freeze()
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        self.backbone.unfreeze()
    
    def set_domain_lambda(self, lambda_: float):
        """Update domain adversarial lambda."""
        self.domain_discriminator.set_lambda(lambda_)
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9):
        """Update prototypical network prototypes (for non-learnable prototypes)."""
        self.prototypical_net.update_prototypes(features, labels, momentum)


class MultiStageHybridModel(nn.Module):
    """
    Wrapper for multi-stage training.
    Controls which components are active in each stage.
    """
    
    def __init__(self, model: HybridModel):
        super().__init__()
        self.model = model
        self.current_stage = 1
    
    def set_stage(self, stage: int):
        """
        Set the current training stage.
        
        Stage 1: Classification only (herbarium)
        Stage 2: Classification + Contrastive (paired classes)
        Stage 3: Classification + Contrastive + Prototypical (all classes)
        Stage 4: Full end-to-end (all losses including domain adversarial)
        """
        self.current_stage = stage
        print(f"Setting training stage to: {stage}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass based on current stage."""
        if self.current_stage == 1:
            # Stage 1: Only classification
            return self.model(x, return_features=False, return_projections=False)
        elif self.current_stage == 2:
            # Stage 2: Classification + contrastive
            return self.model(x, return_features=True, return_projections=True)
        elif self.current_stage >= 3:
            # Stage 3+: Everything
            return self.model(x, return_features=True, return_projections=True)
        
        return self.model(x, **kwargs)
    
    def get_active_losses(self) -> Tuple[bool, bool, bool, bool]:
        """
        Get which losses are active in current stage.
        
        Returns:
            Tuple of (classification, contrastive, prototypical, domain_adversarial)
        """
        if self.current_stage == 1:
            return (True, False, False, False)
        elif self.current_stage == 2:
            return (True, True, False, False)
        elif self.current_stage == 3:
            return (True, True, True, False)
        else:  # Stage 4
            return (True, True, True, True)


if __name__ == "__main__":
    # Test hybrid model
    print("Testing HybridModel...")
    
    model = HybridModel(
        backbone_name='dinov2-vit-b',
        num_classes=100,
        num_paired_classes=60,
        num_unpaired_classes=40,
        pretrained=False
    )
    
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    
    outputs = model(x, return_features=True, return_projections=True)
    
    print(f"Input: {x.shape}")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # Test multi-stage wrapper
    print("\nTesting MultiStageHybridModel...")
    multi_stage_model = MultiStageHybridModel(model)
    
    for stage in [1, 2, 3, 4]:
        multi_stage_model.set_stage(stage)
        active_losses = multi_stage_model.get_active_losses()
        print(f"Stage {stage} - Active losses: Classification={active_losses[0]}, "
              f"Contrastive={active_losses[1]}, Prototypical={active_losses[2]}, "
              f"Domain Adversarial={active_losses[3]}")
