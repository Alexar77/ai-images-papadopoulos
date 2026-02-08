"""
Unified Visualization Utilities
All plotting and reporting functions for exercises 1-5
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
import os


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


def _to_result_list(results):
    """Accept list or dict of experiment results and return a list."""
    if isinstance(results, dict):
        return list(results.values())
    return list(results)


def plot_ex1_accuracy_overview(results, save_path=None):
    """Ex1: Main bar chart with test accuracy for all experiments."""
    results_list = _to_result_list(results)
    results_list = sorted(results_list, key=lambda r: r.get('test_acc', 0), reverse=True)

    names = [r['name'] for r in results_list]
    accs = [r.get('test_acc', 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, accs, color='steelblue', alpha=0.9)
    ax.set_title('Ex1 Overview - Test Accuracy by Experiment', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{accs[i]:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 accuracy overview saved: {save_path}")
    plt.close()


def plot_ex1_optimizer_train_accuracy_curves(results, save_path=None):
    """Ex1: Training accuracy curves for optimizer comparison experiments."""
    results_list = _to_result_list(results)
    optimizer_results = [r for r in results_list if r['name'].startswith('Optimizer_')]
    if not optimizer_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in optimizer_results:
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], linewidth=2, label=result['name'])

    ax.set_title('Ex1 Optimizers - Training Accuracy Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 optimizer curves saved: {save_path}")
    plt.close()


def plot_ex1_learning_rate_loss_curves(results, save_path=None):
    """Ex1: Train loss curves for learning rate experiments."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if 'learning_rate' in r.get('config', {})]
    lr_results = [r for r in lr_results if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lr_results = sorted(lr_results, key=lambda r: r['config']['learning_rate'])
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in lr_results:
        history = result['history']
        lr = result['config']['learning_rate']
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], linewidth=2, label=f"lr={lr}")

    ax.set_title('Ex1 Learning Rates - Training Loss Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 learning-rate loss curves saved: {save_path}")
    plt.close()


def plot_ex1_generalization_gap(results, save_path=None):
    """Ex1: Best train-test accuracy gap per experiment."""
    results_list = _to_result_list(results)
    names = []
    gaps = []

    for result in results_list:
        history = result.get('history', {})
        train_acc = max(history.get('train_acc', [0]))
        val_acc = result.get('test_acc', 0)
        names.append(result['name'])
        gaps.append(train_acc - val_acc)

    sort_idx = np.argsort(gaps)[::-1]
    names = [names[i] for i in sort_idx]
    gaps = [gaps[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, gaps, color='indianred', alpha=0.85)
    ax.set_title('Ex1 Generalization Gap (Best Train Acc - Test Acc)', fontweight='bold')
    ax.set_ylabel('Gap (%)')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{gaps[i]:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 generalization-gap plot saved: {save_path}")
    plt.close()


def plot_ex1_time_vs_accuracy(results, save_path=None):
    """Ex1: Training-time vs test-accuracy scatter."""
    results_list = _to_result_list(results)
    names = [r['name'] for r in results_list]
    times = [r.get('total_time', r.get('metrics', {}).get('total_training_time', 0)) for r in results_list]
    accs = [r.get('test_acc', r.get('metrics', {}).get('test_acc', 0)) for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, accs, s=80, c='darkorange', alpha=0.85)

    for i, name in enumerate(names):
        ax.annotate(name, (times[i], accs[i]), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_title('Ex1 Efficiency Tradeoff - Training Time vs Test Accuracy', fontweight='bold')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 time-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex1_learning_rate_vs_accuracy(results, save_path=None):
    """Ex1: Learning rate against final test accuracy."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if 'learning_rate' in r.get('config', {})]
    lr_results = [r for r in lr_results if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lrs = [r['config']['learning_rate'] for r in lr_results]
    accs = [r.get('test_acc', 0) for r in lr_results]
    sort_idx = np.argsort(lrs)
    lrs = np.array(lrs)[sort_idx]
    accs = np.array(accs)[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, accs, marker='o', linewidth=2, color='teal')
    ax.set_xscale('log')
    ax.set_title('Ex1 Learning Rate vs Test Accuracy', fontweight='bold')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex1 learning-rate-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex2_accuracy_overview(results, save_path=None):
    """Ex2: Main bar chart with test accuracy for all transfer-learning experiments."""
    results_list = _to_result_list(results)
    results_list = sorted(results_list, key=lambda r: r.get('test_acc', 0), reverse=True)

    names = [r['name'] for r in results_list]
    accs = [r.get('test_acc', 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, accs, color='royalblue', alpha=0.9)
    ax.set_title('Ex2 Overview - Test Accuracy by Experiment', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{accs[i]:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 accuracy overview saved: {save_path}")
    plt.close()


def plot_ex2_architecture_train_accuracy_curves(results, save_path=None):
    """Ex2: Training accuracy curves for architecture comparison experiments."""
    results_list = _to_result_list(results)
    arch_results = [r for r in results_list if r['name'].startswith('Architecture_')]
    if not arch_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in arch_results:
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], linewidth=2, label=result['name'])

    ax.set_title('Ex2 Architectures - Training Accuracy Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 architecture curves saved: {save_path}")
    plt.close()


def plot_ex2_frozen_vs_finetuned_curves(results, save_path=None):
    """Ex2: Frozen vs fine-tuned training comparison."""
    results_list = _to_result_list(results)
    mode_results = [r for r in results_list if r['name'].startswith('ResNet18_')]
    mode_results = [r for r in mode_results if ('Frozen' in r['name']) or ('FineTuned' in r['name'])]
    if not mode_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for result in mode_results:
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        axes[0].plot(epochs, history['train_acc'], linewidth=2, label=result['name'])
        axes[1].plot(epochs, history['train_loss'], linewidth=2, label=result['name'])

    axes[0].set_title('Train Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title('Training Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle('Ex2 Frozen vs Fine-Tuned (ResNet18)', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 frozen-vs-finetuned curves saved: {save_path}")
    plt.close()


def plot_ex2_learning_rate_train_loss_curves(results, save_path=None):
    """Ex2: Training loss curves for learning-rate experiments."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lr_results = sorted(
        lr_results,
        key=lambda r: r.get('config', {}).get('learning_rate', r.get('hyperparameters', {}).get('learning_rate', 0))
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in lr_results:
        history = result['history']
        lr = result.get('config', {}).get('learning_rate', result.get('hyperparameters', {}).get('learning_rate', 'NA'))
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], linewidth=2, label=f'lr={lr}')

    ax.set_title('Ex2 Learning Rates - Training Loss Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 learning-rate val-loss curves saved: {save_path}")
    plt.close()


def plot_ex2_time_vs_accuracy(results, save_path=None):
    """Ex2: Training-time vs test-accuracy scatter."""
    results_list = _to_result_list(results)
    names = [r['name'] for r in results_list]
    times = [r.get('total_time', r.get('metrics', {}).get('total_training_time', 0)) for r in results_list]
    accs = [r.get('test_acc', r.get('metrics', {}).get('test_acc', 0)) for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, accs, s=80, c='darkorange', alpha=0.85)
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], accs[i]), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_title('Ex2 Efficiency Tradeoff - Training Time vs Test Accuracy', fontweight='bold')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 time-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex2_learning_rate_vs_accuracy(results, save_path=None):
    """Ex2: Learning rate against final test accuracy."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lrs = [r.get('config', {}).get('learning_rate', r.get('hyperparameters', {}).get('learning_rate', np.nan)) for r in lr_results]
    accs = [r.get('test_acc', 0) for r in lr_results]

    valid = [(lr, acc) for lr, acc in zip(lrs, accs) if not np.isnan(lr)]
    if not valid:
        return

    lrs, accs = zip(*valid)
    sort_idx = np.argsort(lrs)
    lrs = np.array(lrs)[sort_idx]
    accs = np.array(accs)[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, accs, marker='o', linewidth=2, color='teal')
    ax.set_xscale('log')
    ax.set_title('Ex2 Learning Rate vs Test Accuracy', fontweight='bold')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex2 learning-rate-vs-accuracy plot saved: {save_path}")
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


def plot_ex3_miou_overview(results, save_path=None):
    """Ex3: Main bar chart with best validation mIoU per experiment."""
    results_list = _to_result_list(results)
    results_list = sorted(results_list, key=lambda r: r.get('val_miou', 0), reverse=True)

    names = [r['name'] for r in results_list]
    mious = [r.get('val_miou', 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, mious, color='teal', alpha=0.9)
    ax.set_title('Ex3 Overview - Best Validation mIoU by Experiment', fontweight='bold')
    ax.set_ylabel('mIoU (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mious[i]:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 mIoU overview saved: {save_path}")
    plt.close()


def plot_ex3_optimizer_miou_curves(results, save_path=None):
    """Ex3: Validation mIoU curves for optimizer comparison experiments."""
    results_list = _to_result_list(results)
    optimizer_results = [r for r in results_list if r['name'].startswith('Optimizer_')]
    if not optimizer_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in optimizer_results:
        history = result['history']
        epochs = range(1, len(history['val_mean_iou']) + 1)
        ax.plot(epochs, np.array(history['val_mean_iou']) * 100.0, linewidth=2, label=result['name'])

    ax.set_title('Ex3 Optimizers - Validation mIoU Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation mIoU (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 optimizer mIoU curves saved: {save_path}")
    plt.close()


def plot_ex3_optimizer_pixel_acc_curves(results, save_path=None):
    """Ex3: Validation pixel-accuracy curves for optimizer comparison experiments."""
    results_list = _to_result_list(results)
    optimizer_results = [r for r in results_list if r['name'].startswith('Optimizer_')]
    if not optimizer_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in optimizer_results:
        history = result['history']
        epochs = range(1, len(history['val_pixel_acc']) + 1)
        ax.plot(epochs, np.array(history['val_pixel_acc']) * 100.0, linewidth=2, label=result['name'])

    ax.set_title('Ex3 Optimizers - Validation Pixel Accuracy Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Pixel Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 optimizer pixel-accuracy curves saved: {save_path}")
    plt.close()


def plot_ex3_lr_val_loss_curves(results, save_path=None):
    """Ex3: Validation loss curves for learning-rate experiments."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lr_results = sorted(
        lr_results,
        key=lambda r: r.get('config', {}).get('learning_rate', r.get('hyperparameters', {}).get('learning_rate', 0))
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in lr_results:
        history = result['history']
        lr = result.get('config', {}).get('learning_rate', result.get('hyperparameters', {}).get('learning_rate', 'NA'))
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], linewidth=2, label=f'lr={lr}')

    ax.set_title('Ex3 Learning Rates - Validation Loss Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 learning-rate val-loss curves saved: {save_path}")
    plt.close()


def plot_ex3_time_vs_miou(results, save_path=None):
    """Ex3: Training-time vs best-mIoU scatter."""
    results_list = _to_result_list(results)
    names = [r['name'] for r in results_list]
    times = [r.get('total_time', r.get('metrics', {}).get('total_training_time', 0)) for r in results_list]
    mious = [r.get('val_miou', r.get('metrics', {}).get('best_val_miou', 0) * 100.0) for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, mious, s=80, c='darkorange', alpha=0.85)
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], mious[i]), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_title('Ex3 Efficiency Tradeoff - Training Time vs Best mIoU', fontweight='bold')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Best Validation mIoU (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 time-vs-mIoU plot saved: {save_path}")
    plt.close()


def plot_ex3_lr_vs_miou(results, save_path=None):
    """Ex3: Learning rate against best validation mIoU."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if r['name'].startswith('LearningRate_') or r['name'].startswith('LR_')]
    if not lr_results:
        return

    lrs = [r.get('config', {}).get('learning_rate', r.get('hyperparameters', {}).get('learning_rate', np.nan)) for r in lr_results]
    mious = [r.get('val_miou', 0) for r in lr_results]

    valid = [(lr, miou) for lr, miou in zip(lrs, mious) if not np.isnan(lr)]
    if not valid:
        return

    lrs, mious = zip(*valid)
    sort_idx = np.argsort(lrs)
    lrs = np.array(lrs)[sort_idx]
    mious = np.array(mious)[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, mious, marker='o', linewidth=2, color='purple')
    ax.set_xscale('log')
    ax.set_title('Ex3 Learning Rate vs Best mIoU', fontweight='bold')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Best Validation mIoU (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex3 learning-rate-vs-mIoU plot saved: {save_path}")
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


def plot_ex4_accuracy_overview(results, save_path=None):
    """Ex4: Main bar chart with detection accuracy (IoU@0.5) per experiment."""
    results_list = _to_result_list(results)
    results_list = sorted(results_list, key=lambda r: r.get('test_acc', 0), reverse=True)

    names = [r['name'] for r in results_list]
    accs = [r.get('test_acc', 0) for r in results_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, accs, color='steelblue', alpha=0.9)
    ax.set_title('Ex4 Overview - Detection Accuracy (IoU@0.5)', fontweight='bold')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{accs[i]:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex4 accuracy overview saved: {save_path}")
    plt.close()


def plot_ex4_total_loss_curves(results, save_path=None):
    """Ex4: Total training loss curves for all experiments."""
    results_list = _to_result_list(results)
    if not results_list:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results_list:
        history = result.get('history', {})
        curve = history.get('train_loss', [])
        if not curve:
            continue
        epochs = range(1, len(curve) + 1)
        ax.plot(epochs, curve, linewidth=2, label=result['name'])

    ax.set_title('Ex4 Total Training Loss Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex4 total-loss curves saved: {save_path}")
    plt.close()


def plot_ex4_best_loss_components(results, save_path=None):
    """Ex4: Loss components for the best experiment by detection accuracy."""
    results_list = _to_result_list(results)
    if not results_list:
        return

    best = max(results_list, key=lambda r: r.get('test_acc', r.get('metrics', {}).get('detection_accuracy', 0)))
    history = best.get('history', {})
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if len(list(epochs)) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss_classifier'], linewidth=2, label='Classifier')
    ax.plot(epochs, history['train_loss_box_reg'], linewidth=2, label='Box Regression')
    ax.plot(epochs, history['train_loss_objectness'], linewidth=2, label='Objectness')
    ax.plot(epochs, history['train_loss_rpn_box_reg'], linewidth=2, label='RPN Box Reg')
    ax.set_title(f"Ex4 Best Run Loss Components - {best['name']}", fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex4 best-run loss components saved: {save_path}")
    plt.close()


def plot_ex4_lr_vs_accuracy(results, save_path=None):
    """Ex4: Learning rate against detection accuracy."""
    results_list = _to_result_list(results)
    lr_results = [r for r in results_list if 'learning_rate' in r.get('config', {})]
    if not lr_results:
        return

    lrs = [r['config']['learning_rate'] for r in lr_results]
    accs = [r.get('test_acc', r.get('metrics', {}).get('detection_accuracy', 0)) for r in lr_results]
    sort_idx = np.argsort(lrs)
    lrs = np.array(lrs)[sort_idx]
    accs = np.array(accs)[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, accs, marker='o', linewidth=2, color='teal')
    ax.set_xscale('log')
    ax.set_title('Ex4 Learning Rate vs Detection Accuracy', fontweight='bold')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex4 learning-rate-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex4_time_vs_accuracy(results, save_path=None):
    """Ex4: Training-time vs detection-accuracy scatter."""
    results_list = _to_result_list(results)
    names = [r['name'] for r in results_list]
    times = [sum(r.get('history', {}).get('epoch_times', [])) for r in results_list]
    accs = [r.get('test_acc', r.get('metrics', {}).get('detection_accuracy', 0)) for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, accs, s=80, c='darkorange', alpha=0.85)
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], accs[i]), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_title('Ex4 Efficiency Tradeoff - Training Time vs Detection Accuracy', fontweight='bold')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex4 time-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex5_accuracy_overview(results, save_path=None):
    """Ex5: Main bar chart with test accuracy for all experiments."""
    results_list = _to_result_list(results)
    results_list = sorted(results_list, key=lambda r: r['metrics']['test_acc'], reverse=True)

    names = [r['name'] for r in results_list]
    accs = [r['metrics']['test_acc'] for r in results_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, accs, color='steelblue', alpha=0.9)
    ax.set_title('Ex5 Overview - Test Accuracy by Experiment', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{accs[i]:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 accuracy overview saved: {save_path}")
    plt.close()


def plot_ex5_family_mean_accuracy(results, save_path=None):
    """Ex5: Grouped family-level mean accuracy (CNN vs ViT)."""
    results_list = _to_result_list(results)
    cnn_accs = [r['metrics']['test_acc'] for r in results_list if r['config']['architecture'] == 'cnn']
    vit_accs = [r['metrics']['test_acc'] for r in results_list if r['config']['architecture'] == 'vit']
    if not cnn_accs or not vit_accs:
        return

    labels = ['CNN', 'ViT']
    means = [np.mean(cnn_accs), np.mean(vit_accs)]
    stds = [np.std(cnn_accs), np.std(vit_accs)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=['seagreen', 'darkorange'], alpha=0.9)
    ax.set_title('Ex5 Family Comparison - Mean Test Accuracy', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{means[i]:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 family mean-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex5_best_cnn_vs_vit_train_loss(results, save_path=None):
    """Ex5: Train-loss curves of best CNN and best ViT runs."""
    results_list = _to_result_list(results)
    cnn_runs = [r for r in results_list if r['config']['architecture'] == 'cnn']
    vit_runs = [r for r in results_list if r['config']['architecture'] == 'vit']
    if not cnn_runs or not vit_runs:
        return

    best_cnn = max(cnn_runs, key=lambda r: r['metrics']['test_acc'])
    best_vit = max(vit_runs, key=lambda r: r['metrics']['test_acc'])

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in [(best_cnn, 'seagreen'), (best_vit, 'darkorange')]:
        curve = run['history'].get('train_loss', [])
        epochs = range(1, len(curve) + 1)
        ax.plot(epochs, curve, linewidth=2.5, label=run['name'], color=color)

    ax.set_title('Ex5 Best CNN vs Best ViT - Training Loss Curves', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 best CNN-vs-ViT loss plot saved: {save_path}")
    plt.close()


def plot_ex5_lr_vs_accuracy_by_family(results, save_path=None):
    """Ex5: Learning rate vs test accuracy, separate curves for CNN and ViT."""
    results_list = _to_result_list(results)
    cnn_runs = [r for r in results_list if r['config']['architecture'] == 'cnn']
    vit_runs = [r for r in results_list if r['config']['architecture'] == 'vit']
    if not cnn_runs and not vit_runs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for runs, label, color in [(cnn_runs, 'CNN', 'seagreen'), (vit_runs, 'ViT', 'darkorange')]:
        if not runs:
            continue
        pairs = sorted([(r['config']['learning_rate'], r['metrics']['test_acc']) for r in runs], key=lambda x: x[0])
        lrs = [p[0] for p in pairs]
        accs = [p[1] for p in pairs]
        ax.plot(lrs, accs, marker='o', linewidth=2, label=label, color=color)

    ax.set_xscale('log')
    ax.set_title('Ex5 Learning Rate vs Test Accuracy (CNN vs ViT)', fontweight='bold')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 LR-vs-accuracy plot saved: {save_path}")
    plt.close()


def plot_ex5_params_vs_accuracy(results, save_path=None):
    """Ex5: Parameters (millions) vs test accuracy tradeoff."""
    results_list = _to_result_list(results)
    if not results_list:
        return

    params_m = [r['metrics']['parameters']['total_millions'] for r in results_list]
    accs = [r['metrics']['test_acc'] for r in results_list]
    names = [r['name'] for r in results_list]
    colors = ['seagreen' if r['config']['architecture'] == 'cnn' else 'darkorange' for r in results_list]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(params_m, accs, s=90, c=colors, alpha=0.85)
    for i, name in enumerate(names):
        ax.annotate(name, (params_m[i], accs[i]), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_title('Ex5 Tradeoff - Parameters vs Test Accuracy', fontweight='bold')
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 params-vs-accuracy plot saved: {save_path}")
    plt.close()


def create_ex5_best_predictions_panel(results, results_root, save_path=None):
    """Ex5: Side-by-side qualitative panel from best CNN and best ViT predictions."""
    results_list = _to_result_list(results)
    cnn_runs = [r for r in results_list if r['config']['architecture'] == 'cnn']
    vit_runs = [r for r in results_list if r['config']['architecture'] == 'vit']
    if not cnn_runs or not vit_runs:
        return

    best_cnn = max(cnn_runs, key=lambda r: r['metrics']['test_acc'])
    best_vit = max(vit_runs, key=lambda r: r['metrics']['test_acc'])

    cnn_img_path = os.path.join(results_root, best_cnn['experiment_name'], 'predictions.png')
    vit_img_path = os.path.join(results_root, best_vit['experiment_name'], 'predictions.png')
    if not (os.path.exists(cnn_img_path) and os.path.exists(vit_img_path)):
        return

    cnn_img = plt.imread(cnn_img_path)
    vit_img = plt.imread(vit_img_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(cnn_img)
    axes[0].set_title(f"Best CNN: {best_cnn['name']}", fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(vit_img)
    axes[1].set_title(f"Best ViT: {best_vit['name']}", fontweight='bold')
    axes[1].axis('off')
    plt.suptitle('Ex5 Qualitative Comparison - Best CNN vs Best ViT', fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ex5 best predictions panel saved: {save_path}")
    plt.close()


# ============================================================================
# COMPARATIVE VISUALIZATION (Ex5)
# ============================================================================

def plot_comparative_training_curves(results, save_path=None):
    """
    Plot training curves for multiple experiments
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    use_validation = any(len(result['history'].get('val_loss', [])) > 0 for result in results)
    
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
    
    # Secondary loss plot (validation if available, otherwise repeat train)
    ax = axes[0, 1]
    for result in results:
        history = result['history']
        if use_validation:
            curve = history.get('val_loss', [])
            title = 'Validation Loss'
        else:
            curve = history.get('train_loss', [])
            title = 'Training Loss (Reference)'
        epochs = range(1, len(curve) + 1)
        ax.plot(epochs, curve, marker='s', label=result['name'], linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    
    # Secondary accuracy plot (validation if available, otherwise repeat train)
    ax = axes[1, 1]
    for result in results:
        history = result['history']
        if use_validation:
            curve = history.get('val_acc', [])
            title = 'Validation Accuracy'
        else:
            curve = history.get('train_acc', [])
            title = 'Training Accuracy (Reference)'
        epochs = range(1, len(curve) + 1)
        ax.plot(epochs, curve, marker='s', label=result['name'], linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    use_validation = any(r['history'].get('val_acc') for r in results)
    best_val_accs = [
        max(r['history']['val_acc']) if r['history'].get('val_acc')
        else max(r['history']['train_acc']) if r['history'].get('train_acc')
        else 0.0
        for r in results
    ]
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
    
    # Secondary accuracy metric
    ax = axes[0, 1]
    bars = ax.bar(names, best_val_accs, color='coral', alpha=0.8)
    ax.set_title('Best Validation Accuracy' if use_validation else 'Best Training Accuracy', fontsize=14, fontweight='bold')
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
