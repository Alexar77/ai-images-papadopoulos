"""
U-Net Architecture για Semantic Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling: ConvTranspose -> Concatenate -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture για Semantic Segmentation
    
    Args:
        in_channels: Input channels (3 για RGB)
        num_classes: Αριθμός κλάσεων segmentation
        base_channels: Base number of channels (default: 64)
    """
    
    def __init__(self, in_channels=3, num_classes=21, base_channels=64):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder (Upsampling)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def count_parameters(self):
        """Υπολογισμός trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_unet_model(num_classes=21, base_channels=64):
    """
    Δημιουργία U-Net model
    
    Args:
        num_classes: Αριθμός κλάσεων
        base_channels: Base channels (32, 64, 128)
    """
    
    model = UNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
    
    print(f"\n{'='*70}")
    print(f"U-Net Model Summary")
    print(f"{'='*70}")
    print(f"Input: 3 channels (RGB)")
    print(f"Output: {num_classes} classes")
    print(f"Base channels: {base_channels}")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"{'='*70}")
    
    return model


if __name__ == "__main__":
    # Test U-Net
    model = get_unet_model(num_classes=21, base_channels=64)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"\nTest output shape: {output.shape}")
    print(f"Expected: [1, 21, 256, 256]")
