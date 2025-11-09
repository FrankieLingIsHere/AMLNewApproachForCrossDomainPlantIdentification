"""Custom dataset for Herbarium-Field cross-domain plant classification."""

import os
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data.augmentation import get_augmentation, MixUpAugmentation, CutMixAugmentation


class HerbariumFieldDataset(Dataset):
    """
    Dataset for loading herbarium and field images.
    
    Supports:
    - Paired classes (both herbarium and field)
    - Unpaired classes (herbarium only)
    - Domain labels
    - Balanced sampling
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        domain: str = 'both',  # 'herbarium', 'field', or 'both'
        paired_class_file: str = None,
        unpaired_class_file: str = None,
        augmentation_strength: str = 'medium',
        image_size: int = 224,
        is_train: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.domain = domain
        self.image_size = image_size
        self.is_train = is_train
        
        # Load class information
        if paired_class_file is None:
            paired_class_file = os.path.join(data_dir, 'list', 'class_with_pairs.txt')
        if unpaired_class_file is None:
            unpaired_class_file = os.path.join(data_dir, 'list', 'class_without_pairs.txt')
        
        self.paired_classes = self._load_class_list(paired_class_file)
        self.unpaired_classes = self._load_class_list(unpaired_class_file)
        
        # Create class to index mapping
        all_classes = sorted(self.paired_classes + self.unpaired_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load samples
        self.samples = self._load_samples()
        
        # Setup augmentation
        self.herbarium_aug = get_augmentation('herbarium', augmentation_strength, is_train, image_size)
        self.field_aug = get_augmentation('field', augmentation_strength, is_train, image_size)
    
    def _load_class_list(self, filepath: str) -> List[str]:
        """Load class IDs from a text file."""
        with open(filepath, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    
    def _load_samples(self) -> List[Tuple[str, str, int]]:
        """
        Load samples from the dataset.
        
        Returns:
            List of (image_path, class_id, domain_label) tuples
            domain_label: 0 for herbarium, 1 for field/photo
        """
        samples = []
        
        if self.split == 'train':
            train_list = os.path.join(self.data_dir, 'list', 'train.txt')
            with open(train_list, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    
                    rel_path, class_id = parts
                    class_id = class_id.strip()
                    
                    # Determine domain
                    if 'herbarium' in rel_path:
                        current_domain = 'herbarium'
                        domain_label = 0
                    else:
                        current_domain = 'field'
                        domain_label = 1
                    
                    # Filter by domain if specified
                    if self.domain != 'both' and current_domain != self.domain:
                        continue
                    
                    # Check if class is in our class list
                    if class_id not in self.class_to_idx:
                        continue
                    
                    full_path = os.path.join(self.data_dir, rel_path)
                    if os.path.exists(full_path):
                        samples.append((full_path, class_id, domain_label))
        
        elif self.split == 'test':
            test_list = os.path.join(self.data_dir, 'list', 'test.txt')
            test_dir = os.path.join(self.data_dir, 'test')
            
            # Test set only contains field images
            with open(test_list, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    
                    filename, class_id = parts
                    class_id = class_id.strip()
                    
                    if class_id not in self.class_to_idx:
                        continue
                    
                    full_path = os.path.join(test_dir, filename)
                    if os.path.exists(full_path):
                        samples.append((full_path, class_id, 1))  # domain_label=1 for field
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        """
        Get a sample.
        
        Returns:
            Tuple of (image, class_idx, domain_label, class_id)
        """
        img_path, class_id, domain_label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation based on domain
        if domain_label == 0:  # herbarium
            image = self.herbarium_aug(image)
        else:  # field
            image = self.field_aug(image)
        
        # Get class index
        class_idx = self.class_to_idx[class_id]
        
        return image, class_idx, domain_label, class_id
    
    def is_paired_class(self, class_id: str) -> bool:
        """Check if a class is paired (has both herbarium and field images)."""
        return class_id in self.paired_classes
    
    def is_unpaired_class(self, class_id: str) -> bool:
        """Check if a class is unpaired (herbarium only)."""
        return class_id in self.unpaired_classes


class BalancedBatchSampler:
    """
    Sampler that creates balanced batches with equal herbarium and field samples.
    Only works for paired classes.
    """
    
    def __init__(self, dataset: HerbariumFieldDataset, batch_size: int, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Separate indices by domain
        self.herbarium_indices = []
        self.field_indices = []
        
        for idx, (_, class_id, domain_label) in enumerate(dataset.samples):
            # Only include paired classes for balanced sampling
            if not dataset.is_paired_class(class_id):
                continue
            
            if domain_label == 0:
                self.herbarium_indices.append(idx)
            else:
                self.field_indices.append(idx)
        
        # Calculate number of batches
        half_batch = batch_size // 2
        self.num_batches = min(len(self.herbarium_indices), len(self.field_indices)) // half_batch
    
    def __iter__(self):
        # Shuffle indices
        np.random.shuffle(self.herbarium_indices)
        np.random.shuffle(self.field_indices)
        
        half_batch = self.batch_size // 2
        
        for i in range(self.num_batches):
            # Get half batch from each domain
            herb_batch = self.herbarium_indices[i * half_batch:(i + 1) * half_batch]
            field_batch = self.field_indices[i * half_batch:(i + 1) * half_batch]
            
            # Combine and shuffle
            batch = herb_batch + field_batch
            np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        return self.num_batches


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_strength: str = 'medium',
    balance_domains: bool = False,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: Path to Herbarium_Field directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation_strength: 'weak', 'medium', or 'strong'
        balance_domains: Whether to balance domains in batches
        image_size: Image size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = HerbariumFieldDataset(
        data_dir=data_dir,
        split='train',
        domain='both',
        augmentation_strength=augmentation_strength,
        image_size=image_size,
        is_train=True
    )
    
    test_dataset = HerbariumFieldDataset(
        data_dir=data_dir,
        split='test',
        domain='field',  # Test only has field images
        augmentation_strength='weak',
        image_size=image_size,
        is_train=False
    )
    
    # Create dataloaders
    if balance_domains:
        train_sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing HerbariumFieldDataset...")
    
    data_dir = "Herbarium_Field"
    
    if os.path.exists(data_dir):
        # Create dataset
        train_dataset = HerbariumFieldDataset(
            data_dir=data_dir,
            split='train',
            domain='both',
            augmentation_strength='medium'
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Number of classes: {len(train_dataset.class_to_idx)}")
        print(f"Paired classes: {len(train_dataset.paired_classes)}")
        print(f"Unpaired classes: {len(train_dataset.unpaired_classes)}")
        
        # Test sample
        if len(train_dataset) > 0:
            image, class_idx, domain_label, class_id = train_dataset[0]
            print(f"\nSample:")
            print(f"  Image shape: {image.shape}")
            print(f"  Class index: {class_idx}")
            print(f"  Domain label: {domain_label} ({'herbarium' if domain_label == 0 else 'field'})")
            print(f"  Class ID: {class_id}")
        
        # Test dataloader
        train_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=0,
            balance_domains=False
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        images, labels, domains, class_ids = next(iter(train_loader))
        print(f"\nBatch:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Domains: {domains.shape}")
        print(f"  Domain distribution: Herbarium={( domains == 0).sum()}, Field={(domains == 1).sum()}")
    else:
        print(f"Data directory not found: {data_dir}")
