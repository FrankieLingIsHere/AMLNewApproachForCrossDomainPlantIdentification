"""Prototypical loss for unpaired classes."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    """
    Prototypical loss for classification using distance to prototypes.
    
    Computes cross-entropy loss using distances to class prototypes.
    """
    
    def __init__(self, temperature: float = 1.0, distance_metric: str = 'cosine'):
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric
    
    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prototypical loss.
        
        Args:
            features: Input features [B, feature_dim]
            prototypes: Class prototypes [num_classes, feature_dim]
            labels: Ground truth labels [B]
            
        Returns:
            Scalar loss value
        """
        # Normalize for cosine distance
        if self.distance_metric == 'cosine':
            features = F.normalize(features, dim=1)
            prototypes = F.normalize(prototypes, dim=1)
            
            # Cosine similarity (higher is better, so we negate for distance)
            logits = torch.matmul(features, prototypes.t()) / self.temperature
        else:  # Euclidean
            # Compute squared Euclidean distances
            distances = torch.cdist(features, prototypes, p=2).pow(2)
            # Convert to logits (negative distance)
            logits = -distances / self.temperature
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class PrototypicalAlignmentLoss(nn.Module):
    """
    Alignment loss to ensure features are close to their class prototypes.
    
    Encourages features to be similar to their corresponding prototypes.
    """
    
    def __init__(self, distance_metric: str = 'cosine'):
        super().__init__()
        self.distance_metric = distance_metric
    
    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss.
        
        Args:
            features: Input features [B, feature_dim]
            prototypes: Class prototypes [num_classes, feature_dim]
            labels: Ground truth labels [B]
            
        Returns:
            Scalar loss value
        """
        # Get the prototype for each sample
        batch_prototypes = prototypes[labels]  # [B, feature_dim]
        
        if self.distance_metric == 'cosine':
            features = F.normalize(features, dim=1)
            batch_prototypes = F.normalize(batch_prototypes, dim=1)
            
            # Cosine similarity
            similarity = (features * batch_prototypes).sum(dim=1)
            # Loss: negative similarity (we want to maximize similarity)
            loss = -similarity.mean()
        else:  # Euclidean
            # Mean squared distance
            loss = F.mse_loss(features, batch_prototypes)
        
        return loss


class PrototypeConsistencyLoss(nn.Module):
    """
    Consistency loss for prototypes.
    
    Encourages prototypes to be well-separated and diverse.
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype consistency loss.
        
        Args:
            prototypes: Class prototypes [num_classes, feature_dim]
            
        Returns:
            Scalar loss value
        """
        # Normalize prototypes
        prototypes_norm = F.normalize(prototypes, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.t())
        
        # Remove diagonal (self-similarity)
        num_prototypes = prototypes.shape[0]
        mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=prototypes.device)
        
        similarities = similarity_matrix[mask]
        
        # Penalize high similarities between different prototypes
        # We want prototypes to be dissimilar
        loss = F.relu(similarities - (-self.margin)).mean()
        
        return loss


class CombinedPrototypicalLoss(nn.Module):
    """
    Combined prototypical loss with classification, alignment, and consistency.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        distance_metric: str = 'cosine',
        use_alignment: bool = True,
        use_consistency: bool = True,
        alignment_weight: float = 0.1,
        consistency_weight: float = 0.01
    ):
        super().__init__()
        
        self.proto_loss = PrototypicalLoss(temperature, distance_metric)
        self.use_alignment = use_alignment
        self.use_consistency = use_consistency
        
        if use_alignment:
            self.alignment_loss = PrototypicalAlignmentLoss(distance_metric)
            self.alignment_weight = alignment_weight
        
        if use_consistency:
            self.consistency_loss = PrototypeConsistencyLoss()
            self.consistency_weight = consistency_weight
    
    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined prototypical loss.
        
        Args:
            features: Input features [B, feature_dim]
            prototypes: Class prototypes [num_classes, feature_dim]
            labels: Ground truth labels [B]
            
        Returns:
            Scalar loss value
        """
        # Main classification loss
        loss = self.proto_loss(features, prototypes, labels)
        
        # Alignment loss
        if self.use_alignment:
            align_loss = self.alignment_loss(features, prototypes, labels)
            loss = loss + self.alignment_weight * align_loss
        
        # Consistency loss
        if self.use_consistency:
            consist_loss = self.consistency_loss(prototypes)
            loss = loss + self.consistency_weight * consist_loss
        
        return loss


if __name__ == "__main__":
    # Test prototypical losses
    print("Testing Prototypical Losses...")
    
    batch_size = 16
    feature_dim = 768
    num_classes = 40
    
    features = torch.randn(batch_size, feature_dim)
    prototypes = torch.randn(num_classes, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test basic prototypical loss
    print("\n1. PrototypicalLoss:")
    proto_loss = PrototypicalLoss()
    loss = proto_loss(features, prototypes, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test alignment loss
    print("\n2. PrototypicalAlignmentLoss:")
    align_loss = PrototypicalAlignmentLoss()
    loss = align_loss(features, prototypes, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test consistency loss
    print("\n3. PrototypeConsistencyLoss:")
    consist_loss = PrototypeConsistencyLoss()
    loss = consist_loss(prototypes)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n4. CombinedPrototypicalLoss:")
    combined_loss = CombinedPrototypicalLoss()
    loss = combined_loss(features, prototypes, labels)
    print(f"  Loss: {loss.item():.4f}")
