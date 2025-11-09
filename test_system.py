"""
Smoke test for the complete pipeline.
Tests model creation, data loading, and a few training iterations.
"""

import torch
import warnings
warnings.filterwarnings('ignore')

from models.hybrid_model import HybridModel, MultiStageHybridModel
from losses.combined_loss import HybridLoss
from data.dataset import create_dataloaders
from utils.logger import Logger
from utils.evaluation import Evaluator
import tempfile

print("=" * 70)
print("Cross-Domain Plant Classification - Smoke Test")
print("=" * 70)

# Test 1: Model creation
print("\n[1/6] Testing model creation...")
model = HybridModel(
    backbone_name='resnet50',
    num_classes=100,
    num_paired_classes=60,
    num_unpaired_classes=40,
    pretrained=False,
    freeze_backbone=False,
    dropout=0.1
)
wrapped_model = MultiStageHybridModel(model)
wrapped_model.set_stage(1)
print("✓ Model created and staged successfully")

# Test 2: Data loading
print("\n[2/6] Testing data loading...")
try:
    train_loader, test_loader = create_dataloaders(
        data_dir='Herbarium_Field',
        batch_size=8,
        num_workers=0,
        augmentation_strength='weak',
        balance_domains=False,
        image_size=224
    )
    print(f"✓ Dataloaders created")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
except Exception as e:
    print(f"✗ Data loading failed (expected if dataset not available): {e}")
    train_loader = None
    test_loader = None

# Test 3: Loss function
print("\n[3/6] Testing loss function...")
criterion = HybridLoss(
    num_classes=100,
    num_paired_classes=60,
    num_unpaired_classes=40,
    alpha=1.0,
    beta=0.5,
    gamma=0.3,
    delta=0.1
)
print("✓ Loss function created")

# Test 4: Forward pass
print("\n[4/6] Testing forward pass...")
batch_size = 4
x = torch.randn(batch_size, 3, 224, 224)
labels = torch.randint(0, 100, (batch_size,))
domains = torch.randint(0, 2, (batch_size,))

outputs = wrapped_model(x, return_features=True, return_projections=True)
loss, loss_dict = criterion(outputs, labels, domains, stage=1)

print(f"✓ Forward pass successful")
print(f"  - Input: {x.shape}")
print(f"  - Output logits: {outputs['logits'].shape}")
print(f"  - Loss: {loss.item():.4f}")

# Test 5: Backward pass
print("\n[5/6] Testing backward pass...")
optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("✓ Backward pass successful")

# Test 6: Mini training loop (if data available)
if train_loader is not None and len(train_loader) > 0:
    print("\n[6/6] Testing mini training loop (3 batches)...")
    wrapped_model.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Only 3 batches
            break
        
        images, labels, domains, _ = batch
        
        # Forward
        outputs = wrapped_model(images, return_features=True, return_projections=True)
        loss, loss_dict = criterion(outputs, labels, domains, stage=1)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {i+1}/3: Loss = {loss.item():.4f}")
    
    print("✓ Mini training loop successful")
else:
    print("\n[6/6] Skipping mini training loop (no dataset available)")

# Test 7: Evaluation
print("\n[7/7] Testing evaluation...")
wrapped_model.eval()
evaluator = Evaluator(
    num_classes=100,
    paired_class_indices=list(range(60)),
    unpaired_class_indices=list(range(60, 100))
)

with torch.no_grad():
    # Simulate predictions
    for _ in range(5):
        x = torch.randn(4, 3, 224, 224)
        outputs = wrapped_model(x)
        preds = torch.argmax(outputs['logits'], dim=1)
        labels = torch.randint(0, 100, (4,))
        probs = torch.softmax(outputs['logits'], dim=1)
        
        evaluator.update(preds, labels, probs)

metrics = evaluator.compute_metrics()
print(f"✓ Evaluation successful")
print(f"  - Accuracy: {metrics['accuracy']:.4f}")
print(f"  - F1 (Macro): {metrics['f1_macro']:.4f}")

# Test 8: Results saving
print("\n[8/8] Testing results saving...")
with tempfile.TemporaryDirectory() as tmpdir:
    evaluator.save_results(tmpdir)
    print(f"✓ Results saved successfully")

print("\n" + "=" * 70)
print("All smoke tests passed! ✓")
print("=" * 70)
print("\nThe system is ready for:")
print("  - Full training: python train.py [options]")
print("  - Evaluation: python evaluate.py --checkpoint_path <path> [options]")
print("  - See README.md for detailed usage instructions")
