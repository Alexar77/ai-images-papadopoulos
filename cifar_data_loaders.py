"""
Unified CIFAR Data Loaders - CIFAR-10 & CIFAR-100
Supports both classification tasks with flexible configuration
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar_transforms(dataset='cifar10', augment=True, image_size=32):
    """
    Get transforms for CIFAR datasets
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        augment: Whether to apply data augmentation
        image_size: Target image size (32 for CNN, 224 for ViT)
    
    Returns:
        train_transform, test_transform
    """
    
    # CIFAR-10 normalization
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    # CIFAR-100 normalization
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, test_transform


def load_cifar10_dataset(
    data_dir='./data',
    batch_size=128,
    val_split=0.1,
    augment=True,
    image_size=32,
    num_workers=0
):
    """
    Load CIFAR-10 dataset
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    train_transform, test_transform = get_cifar_transforms('cifar10', augment, image_size)
    
    # Load full training set
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = 10
    
    print(f"\n{'='*70}")
    print("CIFAR-10 Dataset Loaded")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Data augmentation: {augment}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, num_classes


def load_cifar100_dataset(
    data_dir='./data',
    batch_size=128,
    val_split=0.1,
    augment=True,
    num_workers=0
):
    """
    Load CIFAR-100 dataset
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    train_transform, test_transform = get_cifar_transforms('cifar100', augment, image_size=32)
    
    # Load full training set
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = 100
    
    print(f"\n{'='*70}")
    print("CIFAR-100 Dataset Loaded")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: 32x32")
    print(f"Batch size: {batch_size}")
    print(f"Data augmentation: {augment}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == "__main__":
    print("Testing unified CIFAR data loaders...\n")
    
    # Test CIFAR-10
    print("="*70)
    print("CIFAR-10 Test")
    print("="*70)
    train_loader, val_loader, test_loader, num_classes = load_cifar10_dataset(
        batch_size=64, image_size=32
    )
    images, labels = next(iter(train_loader))
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Sample classes: {[CIFAR10_CLASSES[l] for l in labels[:5]]}\n")
    
    # Test CIFAR-100
    print("="*70)
    print("CIFAR-100 Test")
    print("="*70)
    train_loader, val_loader, test_loader, num_classes = load_cifar100_dataset(
        batch_size=64
    )
    images, labels = next(iter(train_loader))
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Sample labels: {labels[:5].tolist()}")
