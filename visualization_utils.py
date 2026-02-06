"""
Unified Visualization Utilities
All plotting and reporting functions for exercises 1-5
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch


# ============================================================================
# CLASS NAMES (Shared)
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

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

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)


# ============================================================================
# CLASSIFICATION VISUALIZATION (Ex1, Ex2, Ex5)
# ============================================================================

def plot_training_curves(history, save_path=None, title="Training Progress"):
    """
    Plot training curves (loss and accuracy)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0].set_title('Loss', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_acc' in history and history['train_acc']:
        axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    axes[1].set_title('Accuracy', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    
    plt.close()


def plot_comparison_results(results, save_path=None):
    """
    Bar plot comparing different experiments
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r['name'] for r in results]
    accuracies = [r['test_acc'] if 'test_acc' in r else max(r['history']['val_acc']) for r in results]
    
    bars = ax.bar(names, accuracies, color='steelblue', alpha=0.8)
    ax.set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{accuracies[i]:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {save_path}")
    
    plt.close()


def visualize_classification_predictions(
    images, labels, predictions, probabilities,
    class_names=CIFAR10_CLASSES,
    save_path=None,
    title="Classification Results"
):
    """
    Visualize classification predictions
    """
    
    num_samples = len(images)
    rows = 2
    cols = min(5, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 6))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(min(num_samples, rows*cols)):
        ax = axes[idx]
        
        # Denormalize image
        img = images[idx]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # Try different normalization schemes
        if img.max() <= 1.0:
            # CIFAR-10/100 or ImageNet normalization
            if img.shape[-1] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
        
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = class_names[labels[idx]] if labels[idx] < len(class_names) else f"Class {labels[idx]}"
        pred_label = class_names[predictions[idx]] if predictions[idx] < len(class_names) else f"Class {predictions[idx]}"
        confidence = probabilities[idx][predictions[idx]] * 100
        
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        title_text = f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)"
        
        ax.set_title(title_text, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(min(num_samples, rows*cols), rows*cols):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Predictions saved: {save_path}")
    
    plt.close()


# ============================================================================
# SEGMENTATION VISUALIZATION (Ex3)
# ============================================================================

def apply_colormap_to_mask(mask, colormap=VOC_COLORMAP):
    """Apply colormap to segmentation mask"""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colormap)):
        colored_mask[mask == i] = colormap[i]
    return colored_mask


def visualize_segmentation(
    images, masks, predictions,
    save_path=None,
    title="Segmentation Results"
):
    """
    Visualize segmentation results
    """
    
    num_samples = len(images)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Original image
        img = images[idx]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Original Image', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Ground truth mask
        gt_colored = apply_colormap_to_mask(masks[idx])
        axes[idx, 1].imshow(gt_colored)
        axes[idx, 1].set_title('Ground Truth', fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Prediction mask
        pred_colored = apply_colormap_to_mask(predictions[idx])
        axes[idx, 2].imshow(pred_colored)
        axes[idx, 2].set_title('Prediction', fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Segmentation visualization saved: {save_path}")
    
    plt.close()


def plot_segmentation_training_curves(history, save_path=None):
    """Plot training curves for segmentation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel Accuracy
    axes[0, 1].plot(epochs, history['train_pixel_acc'], 'b-', linewidth=2, label='Train')
    axes[0, 1].plot(epochs, history['val_pixel_acc'], 'r-', linewidth=2, label='Val')
    axes[0, 1].set_title('Pixel Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean IoU
    axes[1, 0].plot(epochs, history['train_mean_iou'], 'b-', linewidth=2, label='Train')
    axes[1, 0].plot(epochs, history['val_mean_iou'], 'r-', linewidth=2, label='Val')
    axes[1, 0].set_title('Mean IoU', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch Times
    axes[1, 1].bar(epochs, history['epoch_times'], color='orange', alpha=0.7)
    axes[1, 1].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    
    plt.close()


# ============================================================================
# DETECTION VISUALIZATION (Ex4)
# ============================================================================

def visualize_detections(
    images, predictions,
    class_names=None,
    save_path=None,
    title="Object Detection Results"
):
    """
    Visualize object detection predictions
    """
    
    if class_names is None:
        class_names = ['__background__'] + PET_CLASSES
    
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for idx, (img, pred) in enumerate(zip(images, predictions)):
        ax = axes[idx]
        
        # Convert to displayable format
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        ax.imshow(img)
        
        # Draw boxes
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            text = f"{class_name}: {score:.2f}"
            ax.text(
                x1, y1 - 5,
                text,
                bbox=dict(facecolor='red', alpha=0.7),
                fontsize=8,
                color='white'
            )
        
        ax.set_title(f"Detections: {len(boxes)}")
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detection visualization saved: {save_path}")
    
    plt.close()


def plot_detection_training_curves(history, save_path=None):
    """Plot training curves for detection"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_title('Total Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Classifier + Box Reg
    axes[0, 1].plot(epochs, history['train_loss_classifier'], 'r-', linewidth=2, label='Classifier')
    axes[0, 1].plot(epochs, history['train_loss_box_reg'], 'g-', linewidth=2, label='Box Regression')
    axes[0, 1].set_title('Classification & Box Regression Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # RPN Losses
    axes[1, 0].plot(epochs, history['train_loss_objectness'], 'm-', linewidth=2, label='Objectness')
    axes[1, 0].plot(epochs, history['train_loss_rpn_box_reg'], 'c-', linewidth=2, label='RPN Box Reg')
    axes[1, 0].set_title('RPN Losses', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Epoch Times
    axes[1, 1].bar(epochs, history['epoch_times'], color='orange', alpha=0.7)
    axes[1, 1].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    
    plt.close()


# ============================================================================
# COMPARATIVE VISUALIZATION (Ex5)
# ============================================================================

def plot_comparative_training_curves(results, save_path=None):
    """
    Plot training curves for multiple experiments
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training Loss
    ax = axes[0, 0]
    for result in results:
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], marker='o', label=result['name'], linewidth=2)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Validation Loss
    ax = axes[0, 1]
    for result in results:
        history = result['history']
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], marker='s', label=result['name'], linewidth=2)
    ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Training Accuracy
    ax = axes[1, 0]
    for result in results:
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], marker='o', label=result['name'], linewidth=2)
    ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Validation Accuracy
    ax = axes[1, 1]
    for result in results:
        history = result['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], marker='s', label=result['name'], linewidth=2)
    ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    
    plt.close()


def plot_model_comparison_bars(results, save_path=None):
    """
    Bar plot for model comparison
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    names = [r['name'] for r in results]
    test_accs = [r['metrics']['test_acc'] for r in results]
    best_val_accs = [max(r['history']['val_acc']) for r in results]
    params = [r['metrics']['parameters']['total_millions'] for r in results]
    times = [sum(r['history']['epoch_times']) for r in results]
    
    # Test Accuracy
    ax = axes[0, 0]
    bars = ax.bar(names, test_accs, color='steelblue', alpha=0.8)
    ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{test_accs[i]:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Best Validation Accuracy
    ax = axes[0, 1]
    bars = ax.bar(names, best_val_accs, color='coral', alpha=0.8)
    ax.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{best_val_accs[i]:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Parameters
    ax = axes[1, 0]
    bars = ax.bar(names, params, color='seagreen', alpha=0.8)
    ax.set_title('Model Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameters (Millions)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{params[i]:.1f}M',
                ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Training Time
    ax = axes[1, 1]
    bars = ax.bar(names, [t/60 for t in times], color='orange', alpha=0.8)
    ax.set_title('Total Training Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (minutes)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{times[i]/60:.1f}m',
                ha='center', va='bottom', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Model comparison saved: {save_path}")
    
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_experiment_report(results, save_path, title="Experiment Report"):
    """
    Generate detailed text report
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write("="*80 + "\n\n")
        
        sorted_results = sorted(results, key=lambda x: x.get('test_acc', 0), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"\n{i}. {result['name']}\n")
            f.write("-"*80 + "\n")
            
            if 'config' in result:
                f.write("CONFIGURATION:\n")
                for key, value in result['config'].items():
                    f.write(f"  • {key}: {value}\n")
            
            if 'metrics' in result:
                f.write("\nRESULTS:\n")
                metrics = result['metrics']
                if 'test_acc' in metrics:
                    f.write(f"  • Test Accuracy: {metrics['test_acc']:.2f}%\n")
                if 'best_val_acc' in metrics:
                    f.write(f"  • Best Val Accuracy: {metrics['best_val_acc']:.2f}%\n")
                if 'parameters' in metrics:
                    f.write(f"  • Parameters: {metrics['parameters']['total']:,}\n")
                if 'total_training_time' in metrics:
                    f.write(f"  • Training Time: {metrics['total_training_time']/60:.1f}m\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Ranking by Test Accuracy:\n")
        for i, result in enumerate(sorted_results, 1):
            test_acc = result.get('test_acc', result.get('metrics', {}).get('test_acc', 0))
            f.write(f"{i}. {result['name']}: {test_acc:.2f}%\n")
    
    print(f"✓ Report saved: {save_path}")


# Aliases for backward compatibility
plot_detection_comparison = plot_comparison_results
generate_comparative_report = generate_experiment_report
