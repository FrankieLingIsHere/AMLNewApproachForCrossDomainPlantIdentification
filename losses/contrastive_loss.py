"""Supervised Contrastive Loss for paired classes (herbarium-field alignment)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    From: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    
    For each anchor, pull positive samples (same class, different domain) closer
    and push negative samples (different class) away.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                domain_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Normalized projection features [B, feature_dim]
            labels: Class labels [B]
            domain_labels: Optional domain labels [B] (0=herbarium, 1=field)
                          If provided, encourages cross-domain alignment
            
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Create label mask: 1 if same class, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        
        # If domain labels provided, modify mask to prioritize cross-domain positives
        if domain_labels is not None:
            domain_labels = domain_labels.contiguous().view(-1, 1)
            same_domain_mask = torch.eq(domain_labels, domain_labels.t()).float().to(device)
            # Cross-domain mask: same class, different domain
            cross_domain_mask = mask * (1 - same_domain_mask)
            # Give higher weight to cross-domain pairs
            mask = cross_domain_mask * 2.0 + mask * same_domain_mask
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class TripletContrastiveLoss(nn.Module):
    """
    Triplet loss variant for cross-domain alignment.
    
    Creates triplets: (herbarium, field_positive, field_negative)
    where herbarium and field_positive are from the same class.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss for cross-domain alignment.
        
        Args:
            features: Normalized features [B, feature_dim]
            labels: Class labels [B]
            domain_labels: Domain labels [B] (0=herbarium, 1=field)
            
        Returns:
            Scalar loss value
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Separate herbarium and field samples
        herbarium_mask = domain_labels == 0
        field_mask = domain_labels == 1
        
        if herbarium_mask.sum() == 0 or field_mask.sum() == 0:
            # No cross-domain pairs available
            return torch.tensor(0.0, device=features.device)
        
        herbarium_features = features[herbarium_mask]
        herbarium_labels = labels[herbarium_mask]
        field_features = features[field_mask]
        field_labels = labels[field_mask]
        
        # Compute pairwise distances
        distances = torch.cdist(herbarium_features, field_features, p=2)
        
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        # For each herbarium sample
        for i, h_label in enumerate(herbarium_labels):
            # Find positive (same class in field)
            positive_mask = field_labels == h_label
            if positive_mask.sum() == 0:
                continue
            
            # Find negative (different class in field)
            negative_mask = field_labels != h_label
            if negative_mask.sum() == 0:
                continue
            
            # Get distances
            positive_distances = distances[i, positive_mask]
            negative_distances = distances[i, negative_mask]
            
            # Compute triplet loss
            pos_dist = positive_distances.min()
            neg_dist = negative_distances.min()
            
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            loss += triplet_loss
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used in SimCLR and similar self-supervised methods.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            features: Normalized features [2*B, feature_dim]
                     First B samples are one augmentation, next B are another
            
        Returns:
            Scalar loss value
        """
        batch_size = features.shape[0] // 2
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Create positive pairs mask
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        mask = mask.repeat(2, 2)
        mask = ~mask  # Invert to exclude self-similarity
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_indices = torch.arange(batch_size, device=features.device)
        pos_i = torch.cat([pos_indices, pos_indices + batch_size])
        pos_j = torch.cat([pos_indices + batch_size, pos_indices])
        
        # Compute loss
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Get positive logits
        positive_logits = log_prob[torch.arange(2 * batch_size, device=features.device), 
                                   torch.cat([pos_j])]
        
        loss = -positive_logits.mean()
        
        return loss


if __name__ == "__main__":
    # Test losses
    print("Testing Contrastive Losses...")
    
    batch_size = 16
    feature_dim = 256
    num_classes = 10
    
    # Test SupCon
    print("\n1. SupervisedContrastiveLoss:")
    supcon_loss = SupervisedContrastiveLoss()
    
    features = torch.randn(batch_size, feature_dim)
    features = F.normalize(features, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    domain_labels = torch.randint(0, 2, (batch_size,))
    
    loss = supcon_loss(features, labels, domain_labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Triplet
    print("\n2. TripletContrastiveLoss:")
    triplet_loss = TripletContrastiveLoss()
    
    loss = triplet_loss(features, labels, domain_labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test NT-Xent
    print("\n3. NTXentLoss:")
    ntxent_loss = NTXentLoss()
    
    # Create two augmentations
    features_aug = torch.randn(2 * batch_size, feature_dim)
    features_aug = F.normalize(features_aug, dim=1)
    
    loss = ntxent_loss(features_aug)
    print(f"  Loss: {loss.item():.4f}")
