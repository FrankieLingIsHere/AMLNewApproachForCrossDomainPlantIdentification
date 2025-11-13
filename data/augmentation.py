"""Data augmentation strategies for herbarium and field images."""

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np


class HerbariumAugmentation:
    """Strong augmentation for herbarium images to simulate field conditions."""
    
    def __init__(self, strength='medium', image_size=224):
        self.image_size = image_size
        
        if strength == 'weak':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif strength == 'medium':
            self.transform = A.Compose([
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=20, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:  # strong
            self.transform = A.Compose([
                A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.7),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.5),
                A.CoarseDropout(num_holes_range=(6, 12), hole_height_range=(12, 24), hole_width_range=(12, 24), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.transform(image=image)['image']


class FieldAugmentation:
    """Standard augmentation for field images (lighter than herbarium)."""
    
    def __init__(self, strength='medium', image_size=224):
        self.image_size = image_size
        
        if strength == 'weak':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif strength in ['medium', 'strong']:
            self.transform = A.Compose([
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.transform(image=image)['image']


class TestAugmentation:
    """Minimal augmentation for test/validation."""
    
    def __init__(self, image_size=224):
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.transform(image=image)['image']


class MixUpAugmentation:
    """MixUp augmentation for paired classes."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, image1, image2, label1, label2):
        """
        Mix two images and their labels.
        
        Args:
            image1, image2: Images [C, H, W]
            label1, label2: Labels (int)
            
        Returns:
            mixed_image, mixed_label (one-hot or lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        mixed_image = lam * image1 + (1 - lam) * image2
        
        return mixed_image, lam, label1, label2


class CutMixAugmentation:
    """CutMix augmentation for paired classes."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, image1, image2, label1, label2):
        """
        Apply CutMix to two images.
        
        Args:
            image1, image2: Images [C, H, W]
            label1, label2: Labels (int)
            
        Returns:
            mixed_image, lambda, label1, label2
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        _, H, W = image1.shape
        
        # Random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_image = image1.clone()
        mixed_image[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_image, lam, label1, label2


def get_augmentation(domain='herbarium', strength='medium', is_train=True, image_size=224):
    """
    Get appropriate augmentation pipeline.
    
    Args:
        domain: 'herbarium' or 'field'
        strength: 'weak', 'medium', or 'strong'
        is_train: Whether for training (True) or testing (False)
        image_size: Target image size
        
    Returns:
        Augmentation transform
    """
    if not is_train:
        return TestAugmentation(image_size)
    
    if domain == 'herbarium':
        return HerbariumAugmentation(strength, image_size)
    else:  # field
        return FieldAugmentation(strength, image_size)


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")
    
    from PIL import Image
    
    # Create dummy image
    dummy_image = Image.new('RGB', (256, 256), color='red')
    
    # Test herbarium augmentation
    print("\n1. Herbarium augmentation (strong):")
    herb_aug = HerbariumAugmentation(strength='strong')
    aug_img = herb_aug(dummy_image)
    print(f"  Output shape: {aug_img.shape}, dtype: {aug_img.dtype}")
    
    # Test field augmentation
    print("\n2. Field augmentation (medium):")
    field_aug = FieldAugmentation(strength='medium')
    aug_img = field_aug(dummy_image)
    print(f"  Output shape: {aug_img.shape}, dtype: {aug_img.dtype}")
    
    # Test test augmentation
    print("\n3. Test augmentation:")
    test_aug = TestAugmentation()
    aug_img = test_aug(dummy_image)
    print(f"  Output shape: {aug_img.shape}, dtype: {aug_img.dtype}")
    
    # Test MixUp
    print("\n4. MixUp:")
    mixup = MixUpAugmentation(alpha=0.2)
    img1 = torch.randn(3, 224, 224)
    img2 = torch.randn(3, 224, 224)
    mixed, lam, l1, l2 = mixup(img1, img2, 0, 1)
    print(f"  Mixed image shape: {mixed.shape}, lambda: {lam:.3f}")
    
    # Test CutMix
    print("\n5. CutMix:")
    cutmix = CutMixAugmentation(alpha=1.0)
    mixed, lam, l1, l2 = cutmix(img1, img2, 0, 1)
    print(f"  Mixed image shape: {mixed.shape}, lambda: {lam:.3f}")
