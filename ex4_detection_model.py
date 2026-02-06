"""
Faster R-CNN Model για Object Detection
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(num_classes, pretrained=True, backbone='resnet50'):
    """
    Δημιουργία Faster R-CNN model
    
    Args:
        num_classes: Αριθμός κλάσεων (συμπεριλαμβανομένου background)
        pretrained: Load pretrained weights
        backbone: 'resnet50' ή 'mobilenet'
    
    Returns:
        model
    """
    
    print(f"\n{'='*70}")
    print(f"Creating Faster R-CNN ({backbone.upper()}) for Object Detection")
    print(f"{'='*70}")
    print(f"Pretrained: {pretrained}")
    print(f"Number of classes: {num_classes}")
    
    if backbone == 'resnet50':
        # Load pretrained Faster R-CNN with ResNet50 backbone
        model = fasterrcnn_resnet50_fpn(
            weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT' if pretrained else None
        )
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif backbone == 'mobilenet':
        # MobileNet variant (lighter)
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights='FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT' if pretrained else None
        )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}")
    
    return model


if __name__ == "__main__":
    # Test models
    for backbone in ['resnet50', 'mobilenet']:
        model = get_detection_model(num_classes=38, backbone=backbone)
        
        # Test forward pass
        model.eval()
        x = [torch.rand(3, 300, 400)]
        predictions = model(x)
        print(f"\n✓ {backbone}: {len(predictions)} predictions\n")
