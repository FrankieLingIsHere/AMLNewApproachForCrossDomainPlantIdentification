"""Prototypical Networks for handling unpaired classes (herbarium-only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for unpaired classes.
    Learns class prototypes in the embedding space.
    """
    
    def __init__(self, feature_dim: int, num_unpaired_classes: int, 
                 learnable_prototypes: bool = True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_unpaired_classes = num_unpaired_classes
        self.learnable_prototypes = learnable_prototypes
        
        # Initialize prototypes
        if learnable_prototypes:
            # Learnable prototypes (initialized randomly, trained end-to-end)
            self.prototypes = nn.Parameter(
                torch.randn(num_unpaired_classes, feature_dim)
            )
            # Normalize prototypes
            with torch.no_grad():
                self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
        else:
            # Non-learnable prototypes (computed from features, updated during training)
            self.register_buffer('prototypes', torch.zeros(num_unpaired_classes, feature_dim))
            self.register_buffer('prototype_counts', torch.zeros(num_unpaired_classes))
    
    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute distances from features to prototypes.
        
        Args:
            features: Input features [B, feature_dim]
            temperature: Temperature for softmax scaling
            
        Returns:
            Logits [B, num_unpaired_classes] (negative distances)
        """
        # Normalize features and prototypes for cosine similarity
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        
        # Compute cosine similarity (equivalent to negative euclidean distance in normalized space)
        logits = torch.matmul(features_norm, prototypes_norm.t()) / temperature
        
        return logits
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9):
        """
        Update prototypes using moving average (for non-learnable prototypes).
        
        Args:
            features: Input features [B, feature_dim]
            labels: Class labels [B] (relative to unpaired classes)
            momentum: Momentum for moving average
        """
        if self.learnable_prototypes:
            return  # Don't update learnable prototypes this way
        
        with torch.no_grad():
            features_norm = F.normalize(features, dim=1)
            
            for i in range(self.num_unpaired_classes):
                mask = labels == i
                if mask.sum() > 0:
                    # Compute mean of features for this class
                    class_features = features_norm[mask].mean(dim=0)
                    
                    # Update prototype with momentum
                    if self.prototype_counts[i] > 0:
                        self.prototypes[i] = momentum * self.prototypes[i] + (1 - momentum) * class_features
                    else:
                        self.prototypes[i] = class_features
                    
                    self.prototype_counts[i] += 1
                    
                    # Normalize prototype
                    self.prototypes[i] = F.normalize(self.prototypes[i], dim=0)
    
    def get_prototypes(self) -> torch.Tensor:
        """Get the current prototypes."""
        return self.prototypes


class TransductivePrototypicalRefinement(nn.Module):
    """
    Transductive refinement for prototypes during inference.
    Uses unlabeled test samples to refine prototypes.
    """
    
    def __init__(self, num_iterations: int = 5, alpha: float = 0.5):
        super().__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
    
    def refine(self, prototypes: torch.Tensor, test_features: torch.Tensor, 
               temperature: float = 1.0) -> torch.Tensor:
        """
        Refine prototypes using test features.
        
        Args:
            prototypes: Initial prototypes [num_classes, feature_dim]
            test_features: Test features [N, feature_dim]
            temperature: Temperature for softmax
            
        Returns:
            Refined prototypes [num_classes, feature_dim]
        """
        refined_prototypes = prototypes.clone()
        
        for _ in range(self.num_iterations):
            # Compute soft assignments
            features_norm = F.normalize(test_features, dim=1)
            prototypes_norm = F.normalize(refined_prototypes, dim=1)
            
            logits = torch.matmul(features_norm, prototypes_norm.t()) / temperature
            soft_assignments = F.softmax(logits, dim=1)  # [N, num_classes]
            
            # Update prototypes
            weighted_features = torch.matmul(soft_assignments.t(), features_norm)  # [num_classes, feature_dim]
            assignment_sums = soft_assignments.sum(dim=0, keepdim=True).t()  # [num_classes, 1]
            
            new_prototypes = weighted_features / (assignment_sums + 1e-8)
            new_prototypes = F.normalize(new_prototypes, dim=1)
            
            # Blend with original prototypes
            refined_prototypes = self.alpha * refined_prototypes + (1 - self.alpha) * new_prototypes
            refined_prototypes = F.normalize(refined_prototypes, dim=1)
        
        return refined_prototypes


if __name__ == "__main__":
    # Test prototypical network
    print("Testing PrototypicalNetwork...")
    
    feature_dim = 768
    num_unpaired_classes = 40
    batch_size = 16
    
    # Test learnable prototypes
    proto_net = PrototypicalNetwork(feature_dim, num_unpaired_classes, learnable_prototypes=True)
    
    features = torch.randn(batch_size, feature_dim)
    logits = proto_net(features)
    
    print(f"Input features: {features.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Prototypes: {proto_net.get_prototypes().shape}")
    
    # Test transductive refinement
    print("\nTesting TransductivePrototypicalRefinement...")
    
    refiner = TransductivePrototypicalRefinement()
    test_features = torch.randn(100, feature_dim)
    prototypes = proto_net.get_prototypes()
    
    refined_prototypes = refiner.refine(prototypes, test_features)
    print(f"Original prototypes: {prototypes.shape}")
    print(f"Refined prototypes: {refined_prototypes.shape}")
