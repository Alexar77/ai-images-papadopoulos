"""
Unified Oxford Pet Data Loaders
Supports both classification and object detection tasks
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


# Pet class names (37 breeds)
PET_CLASSES = [
    'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
    'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair',
    'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter',
    'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
    'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian',
    'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]


def collate_fn_detection(batch):
    """Custom collate function for object detection"""
    return tuple(zip(*batch))


class PetDetectionDataset(torch.utils.data.Dataset):
    """Oxford Pet Dataset for Object Detection"""
    
    def __init__(self, root, split='trainval', transforms=None):
        self.dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types=['category', 'segmentation'],
            download=True
        )
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, (label, mask) = self.dataset[idx]
        
        # Convert PIL to tensor
        img = transforms.ToTensor()(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        
        # Get bounding box from mask
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        
        boxes = []
        labels = []
        masks = []
        
        for obj_id in obj_ids:
            pos = torch.where(mask == obj_id)
            if len(pos[0]) == 0:
                continue
            
            xmin = torch.min(pos[1]).item()
            xmax = torch.max(pos[1]).item()
            ymin = torch.min(pos[0]).item()
            ymax = torch.max(pos[0]).item()
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label + 1)
            masks.append(mask == obj_id)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target


def load_pet_classification_dataset(
    data_dir='./data',
    batch_size=32,
    image_size=224,
    num_workers=0
):
    """
    Load Oxford Pet for classification (Transfer Learning)
    
    Returns:
        train_loader, test_loader, num_classes
    """
    
    # Transforms for pretrained models (ImageNet normalization)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='trainval',
        target_types='category',
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='test',
        target_types='category',
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = 37
    
    print(f"\n{'='*70}")
    print("Oxford Pet Classification Dataset Loaded")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    return train_loader, test_loader, num_classes


def load_pet_detection_dataset(
    data_dir='./data',
    batch_size=4,
    num_workers=0
):
    """
    Load Oxford Pet for object detection
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = PetDetectionDataset(
        root=data_dir,
        split='trainval',
        transforms=train_transforms
    )
    
    val_dataset = PetDetectionDataset(
        root=data_dir,
        split='test',
        transforms=train_transforms
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True
    )
    
    num_classes = 38  # 37 pet breeds + background
    
    print(f"\n{'='*70}")
    print("Oxford Pet Detection Dataset Loaded")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes} (37 pets + background)")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    print("Testing unified Pet data loaders...\n")
    
    # Test Classification
    print("="*70)
    print("Pet Classification Test")
    print("="*70)
    train_loader, test_loader, num_classes = load_pet_classification_dataset(batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Sample breeds: {[PET_CLASSES[l] for l in labels[:3]]}\n")
    
    # Test Detection
    print("="*70)
    print("Pet Detection Test")
    print("="*70)
    train_loader, val_loader, num_classes = load_pet_detection_dataset(batch_size=2)
    images, targets = next(iter(train_loader))
    print(f"✓ Batch size: {len(images)}")
    print(f"✓ Boxes in first image: {len(targets[0]['boxes'])}")
    print(f"✓ Labels: {targets[0]['labels']}")
