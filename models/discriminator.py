"""Domain discriminator with Gradient Reversal Layer (GRL) for domain-adversarial training."""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer - reverses gradients during backprop."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer module."""
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Update the lambda value (typically scheduled during training)."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for domain adversarial training.
    Takes features and predicts whether they're from herbarium or field domain.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 1024, lambda_=1.0):
        super().__init__()
        
        self.grl = GradientReversalLayer(lambda_)
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim // 2, 2)  # Binary: herbarium vs field
        )
    
    def forward(self, features):
        """
        Forward pass through discriminator.
        
        Args:
            features: Input features [B, feature_dim]
            
        Returns:
            Domain predictions [B, 2] (logits for herbarium/field)
        """
        reversed_features = self.grl(features)
        domain_pred = self.discriminator(reversed_features)
        return domain_pred
    
    def set_lambda(self, lambda_):
        """Update the GRL lambda value."""
        self.grl.set_lambda(lambda_)


def compute_grl_lambda(epoch: int, max_epochs: int, gamma: float = 10.0):
    """
    Compute the lambda value for GRL based on training progress.
    Gradually increases from 0 to 1 following the schedule from DANN paper.
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        gamma: Controls the shape of the schedule
        
    Returns:
        Lambda value
    """
    p = float(epoch) / float(max_epochs)
    lambda_ = 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0
    return float(lambda_)


if __name__ == "__main__":
    # Test domain discriminator
    print("Testing DomainDiscriminator...")
    
    feature_dim = 768
    batch_size = 16
    
    discriminator = DomainDiscriminator(feature_dim)
    
    # Test forward pass
    features = torch.randn(batch_size, feature_dim)
    domain_pred = discriminator(features)
    
    print(f"Input features: {features.shape}")
    print(f"Domain predictions: {domain_pred.shape}")
    
    # Test lambda scheduling
    print("\nTesting lambda scheduling:")
    for epoch in [0, 25, 50, 75, 100]:
        lambda_ = compute_grl_lambda(epoch, 100)
        print(f"Epoch {epoch}/100: lambda = {lambda_:.4f}")
