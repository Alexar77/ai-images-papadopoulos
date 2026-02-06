"""
Transfer Learning Models για Oxford Pet Classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_transfer_model(model_name='resnet18', num_classes=37, pretrained=True, freeze_features=True):
    """
    Δημιουργία transfer learning model
    
    Args:
        model_name: 'resnet18', 'resnet50', 'alexnet', 'vgg16', 'efficientnet_b0'
        num_classes: Αριθμός κλάσεων (37 για Oxford Pet)
        pretrained: Χρήση ImageNet weights
        freeze_features: Πάγωμα των feature extraction layers
        
    Returns:
        model, model_info (dict)
    """
    
    print(f"\n{'='*70}")
    print(f"Creating {model_name.upper()} for Transfer Learning")
    print(f"{'='*70}")
    print(f"Pretrained: {pretrained}")
    print(f"Freeze features: {freeze_features}")
    
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        trainable_layer_name = 'fc'
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        trainable_layer_name = 'fc'
        
    elif model_name == 'alexnet':
        model = models.alexnet(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        trainable_layer_name = 'classifier.6'
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        trainable_layer_name = 'classifier.6'
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        trainable_layer_name = 'classifier.1'
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Πάγωμα feature extraction layers
    if freeze_features and pretrained:
        for name, param in model.named_parameters():
            if trainable_layer_name not in name:
                param.requires_grad = False
        print("✓ Feature extraction layers frozen")
    else:
        print("✓ All layers trainable")
    
    # Υπολογισμός trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"{'='*70}")
    
    model_info = {
        'name': model_name,
        'pretrained': pretrained,
        'freeze_features': freeze_features,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    return model, model_info


if __name__ == "__main__":
    # Test models
    for model_name in ['resnet18', 'alexnet', 'vgg16']:
        model, info = get_transfer_model(model_name, num_classes=37, freeze_features=True)
        print(f"\n✓ {model_name}: {info['trainable_params']:,} trainable params\n")
