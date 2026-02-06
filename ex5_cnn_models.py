"""
CNN Models για Comparative Study: VGG & ResNet
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG11_CIFAR(nn.Module):
    """Simplified VGG11 for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(VGG11_CIFAR, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18_CIFAR(nn.Module):
    """ResNet18 adapted for CIFAR-10"""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_CIFAR, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT' if pretrained else None)
        
        # Modify first conv for 32x32 input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class ResNet50_CIFAR(nn.Module):
    """ResNet50 adapted for CIFAR-10"""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet50_CIFAR, self).__init__()
        
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT' if pretrained else None)
        
        # Modify for 32x32 input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        
        # Modify final fc
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def get_cnn_model(model_name='vgg11', num_classes=10, pretrained=False):
    """
    Factory function για CNN models
    
    Args:
        model_name: 'vgg11', 'resnet18', 'resnet50'
        num_classes: Αριθμός κλάσεων
        pretrained: Load pretrained weights
    
    Returns:
        model
    """
    
    print(f"\n{'='*70}")
    print(f"Creating CNN Model: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Number of classes: {num_classes}")
    print(f"Pretrained: {pretrained}")
    
    if model_name == 'vgg11':
        model = VGG11_CIFAR(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = ResNet18_CIFAR(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet50':
        model = ResNet50_CIFAR(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    return model


if __name__ == "__main__":
    # Test models
    print("Testing CNN models...\n")
    
    for model_name in ['vgg11', 'resnet18', 'resnet50']:
        model = get_cnn_model(model_name, num_classes=10)
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        print(f"✓ {model_name}: Input {x.shape} -> Output {out.shape}\n")
