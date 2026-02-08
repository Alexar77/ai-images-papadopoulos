"""
Φόρτωση και προετοιμασία του SBD (Semantic Boundaries Dataset)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


def get_sbd_transforms(image_size=256):
    """
    Transformations για SBD segmentation dataset
    
    Args:
        image_size: Μέγεθος εικόνας (256x256 ή 512x512)
    """
    
    # For images
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # For masks (no normalization)
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform


class SBDSegmentation(torch.utils.data.Dataset):
    """Custom Dataset για SBD με transforms"""
    
    def __init__(self, root, image_set='train', download=True, image_size=256):
        try:
            self.dataset = torchvision.datasets.SBDataset(
                root=root,
                image_set=image_set,
                mode='segmentation',
                download=download
            )
        except Exception as e:
            # Some torchvision versions fail on repeated download attempts
            # if extracted folders already exist. Retry without download.
            if download and "already exists" in str(e):
                self.dataset = torchvision.datasets.SBDataset(
                    root=root,
                    image_set=image_set,
                    mode='segmentation',
                    download=False
                )
            else:
                raise
        self.image_transform, self.mask_transform = get_sbd_transforms(image_size)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # Convert mask to long tensor and squeeze
        mask = (mask * 255).long().squeeze(0)
        
        return image, mask


def load_sbd_dataset(data_dir='./data', batch_size=8, num_workers=2, image_size=256):
    """
    Φόρτωση SBD dataset για semantic segmentation
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    # If SBD files are already present, skip download to avoid repeated
    # extraction/move overhead on slower filesystems (e.g. WSL mounted drives).
    has_sbd = all(
        os.path.exists(os.path.join(data_dir, p))
        for p in ['img', 'cls', 'inst', 'train.txt', 'val.txt']
    )
    train_download = not has_sbd

    # Φόρτωση training set
    train_dataset = SBDSegmentation(
        root=data_dir,
        image_set='train',
        download=train_download,
        image_size=image_size
    )
    
    # Φόρτωση validation set
    val_dataset = SBDSegmentation(
        root=data_dir,
        image_set='val',
        download=False,
        image_size=image_size
    )
    
    # DataLoaders
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
    
    num_classes = 21  # Pascal VOC has 20 classes + background
    
    print(f"SBD Dataset loaded!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, num_classes


# Pascal VOC class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]


if __name__ == "__main__":
    print("Testing SBD data loading...")
    train_loader, val_loader, num_classes = load_sbd_dataset(batch_size=4)
    print(f"\n✓ Dataset ready with {num_classes} classes!")
