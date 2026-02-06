"""
Vanilla CNN Implementation for CIFAR-100
Exercise #1: Architecture and Hyperparameter Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaCNN(nn.Module):
    """
    Vanilla CNN με 4 συνελικτικά στρώματα για CIFAR-100
    Αρχιτεκτονική: Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> Flatten -> Dense
    """
    
    def __init__(self, num_classes=100):
        super(VanillaCNN, self).__init__()
        
        # Συνελικτικό στρώμα 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Συνελικτικό στρώμα 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # Συνελικτικό στρώμα 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        
        # Συνελικτικό στρώμα 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4 -> 2x2
        
        # Πλήρως συνδεδεμένα στρώματα (Dense layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional blocks
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten and fully connected
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Υπολογισμός συνολικού αριθμού παραμέτρων"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(3, 32, 32)):
    """Εμφάνιση περίληψης του μοντέλου"""
    print("=" * 70)
    print("Vanilla CNN Architecture Summary")
    print("=" * 70)
    print(f"Input size: {input_size}")
    print(f"Number of classes: 100 (CIFAR-100)")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print("=" * 70)
    print("\nLayer Structure:")
    print("-" * 70)
    for name, module in model.named_children():
        print(f"{name:15s}: {module}")
    print("=" * 70)


if __name__ == "__main__":
    # Test του μοντέλου
    model = VanillaCNN(num_classes=100)
    get_model_summary(model)
    
    # Test με τυχαίο input
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"\nTest output shape: {output.shape}")
    print(f"Expected shape: [1, 100]")
