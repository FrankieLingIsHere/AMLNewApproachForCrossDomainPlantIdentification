"""
Flexible backbone loader supporting multiple architectures:
- DINOv2 (ViT-S/B/L/14)
- ResNet50
- EfficientNet-B3
"""

import torch
import torch.nn as nn
from typing import Tuple


class BackboneLoader:
    """Factory class for loading different backbone architectures."""
    
    @staticmethod
    def get_backbone(backbone_name: str, pretrained: bool = True, gradient_checkpointing: bool = False) -> Tuple[nn.Module, int]:
        """
        Load a backbone model and return the model + feature dimension.
        
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            gradient_checkpointing: Whether to enable gradient checkpointing
            
        Returns:
            Tuple of (backbone_model, feature_dim)
        """
        backbone_name = backbone_name.lower()
        
        if backbone_name.startswith('dinov2'):
            return BackboneLoader._load_dinov2(backbone_name, pretrained, gradient_checkpointing)
        elif backbone_name == 'resnet50':
            return BackboneLoader._load_resnet50(pretrained, gradient_checkpointing)
        elif backbone_name == 'efficientnet-b3':
            return BackboneLoader._load_efficientnet_b3(pretrained, gradient_checkpointing)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    @staticmethod
    def _load_dinov2(backbone_name: str, pretrained: bool, gradient_checkpointing: bool) -> Tuple[nn.Module, int]:
        """Load DINOv2 model with fallback handling."""
        # Map backbone names to model variants
        variant_map = {
            'dinov2-vit-s': 'dinov2_vits14',
            'dinov2-vit-b': 'dinov2_vitb14',
            'dinov2-vit-l': 'dinov2_vitl14',
        }
        
        if backbone_name not in variant_map:
            raise ValueError(f"Unknown DINOv2 variant: {backbone_name}")
        
        model_name = variant_map[backbone_name]
        
        # Get feature dimension
        feature_dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
        }
        feature_dim = feature_dim_map[model_name]
        
        # Try loading from timm first (more reliable), with fallback to torch.hub
        model = None
        import timm
        
        # Map to timm model names
        timm_model_map = {
            'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
            'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
        }
        
        timm_model_name = timm_model_map.get(model_name)
        
        # Try with pretrained first, fall back to non-pretrained if download fails
        if pretrained:
            try:
                print(f"Loading {timm_model_name} from timm with pretrained weights...")
                model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0,  # Remove classifier head
                    dynamic_img_size=True,  # Allow flexible image sizes
                )
                print(f"Successfully loaded pretrained model from timm")
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights ({str(e)})")
                print(f"Falling back to non-pretrained initialization...")
                pretrained = False
        
        # If pretrained failed or was False, load without pretrained weights
        if not pretrained:
            try:
                print(f"Loading {timm_model_name} from timm without pretrained weights...")
                model = timm.create_model(
                    timm_model_name,
                    pretrained=False,
                    num_classes=0,  # Remove classifier head
                    dynamic_img_size=True,  # Allow flexible image sizes
                )
                print(f"Successfully loaded model without pretrained weights")
            except Exception as e:
                print(f"Warning: Failed to load from timm ({str(e)})")
                # Last resort: try torch.hub
                try:
                    print(f"Trying torch.hub as last resort...")
                    model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False, trust_repo=True, skip_validation=True)
                    print(f"Successfully loaded from torch.hub")
                except Exception as hub_error:
                    raise RuntimeError(f"Could not load DINOv2 model from timm or torch.hub: {str(hub_error)}")
        
        if model is None:
            raise RuntimeError(f"Failed to load model {model_name}")
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        
        return model, feature_dim
    
    @staticmethod
    def _load_resnet50(pretrained: bool, gradient_checkpointing: bool) -> Tuple[nn.Module, int]:
        """Load ResNet50 from torchvision."""
        from torchvision.models import resnet50, ResNet50_Weights
        
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=None)
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
        
        # Note: ResNet doesn't natively support gradient checkpointing in the same way
        # For production, you could wrap blocks with torch.utils.checkpoint
        
        return model, feature_dim
    
    @staticmethod
    def _load_efficientnet_b3(pretrained: bool, gradient_checkpointing: bool) -> Tuple[nn.Module, int]:
        """Load EfficientNet-B3 from timm."""
        import timm
        
        model = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=''  # We'll handle pooling ourselves
        )
        
        feature_dim = 1536  # EfficientNet-B3 feature dimension
        
        if gradient_checkpointing:
            model.set_grad_checkpointing(True)
        
        return model, feature_dim


class BackboneWrapper(nn.Module):
    """Wrapper to ensure consistent output format across different backbones."""
    
    def __init__(self, backbone_name: str, pretrained: bool = True, 
                 gradient_checkpointing: bool = False, freeze: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = BackboneLoader.get_backbone(
            backbone_name, pretrained, gradient_checkpointing
        )
        
        # Global average pooling for spatial outputs
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Features of shape [B, feature_dim]
        """
        features = self.backbone(x)
        
        # Handle different output formats
        if len(features.shape) == 4:  # [B, C, H, W]
            features = self.pool(features)
            features = features.flatten(1)
        elif len(features.shape) == 2:  # [B, C]
            pass  # Already in the right format (DINOv2 outputs this)
        elif len(features.shape) == 3:  # [B, N, C] (ViT with cls token)
            # Take the CLS token (first token)
            features = features[:, 0]
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return features


if __name__ == "__main__":
    # Test different backbones
    print("Testing backbones...")
    
    backbones = ['dinov2-vit-s', 'dinov2-vit-b', 'dinov2-vit-l', 'resnet50', 'efficientnet-b3']
    
    for backbone_name in backbones:
        print(f"\n{backbone_name}:")
        try:
            model = BackboneWrapper(backbone_name, pretrained=False)
            x = torch.randn(2, 3, 224, 224)
            out = model(x)
            print(f"  Input: {x.shape}, Output: {out.shape}, Feature dim: {model.feature_dim}")
        except Exception as e:
            print(f"  Error: {e}")
