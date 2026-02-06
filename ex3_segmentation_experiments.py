"""
Πειράματα Semantic Segmentation με U-Net
Άσκηση #3: Σημασιολογική Τμηματοποίηση
"""

import torch
import torch.nn as nn
import time
import os
from datetime import datetime

from ex3_sbd_data_loader import load_sbd_dataset
from ex3_unet_model import get_unet_model
from training_utils import SegmentationTrainer, get_optimizer, get_scheduler
from visualization_utils import (
    plot_segmentation_training_curves,
    visualize_segmentation,
    plot_comparison_results,
    generate_experiment_report
)


def run_segmentation_experiment(experiment_name, hyperparameters,
                                train_loader, val_loader, num_classes,
                                num_epochs=30, results_dir='results_segmentation'):
    """
    Εκτέλεση segmentation experiment
    """
    
    print("\n" + "=" * 80)
    print(f"ΕΝΑΡΞΗ ΠΕΙΡΑΜΑΤΟΣ: {experiment_name}")
    print("=" * 80)
    print("\nΥπερ-παράμετροι:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Create directory
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_unet_model(
        num_classes=num_classes,
        base_channels=hyperparameters.get('base_channels', 64)
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = get_optimizer(
        model,
        optimizer_name=hyperparameters.get('optimizer', 'adam'),
        learning_rate=hyperparameters.get('learning_rate', 0.001),
        weight_decay=hyperparameters.get('weight_decay', 1e-4)
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=hyperparameters.get('scheduler', 'step'),
        num_epochs=num_epochs
    )
    
    # Train
    trainer = SegmentationTrainer(model, device=device)
    start_time = time.time()
    history = trainer.train(
        train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler, num_classes=num_classes,
        early_stopping_patience=7
    )
    training_time = time.time() - start_time
    
    # Get sample predictions
    images, predictions, ground_truth = trainer.get_predictions(val_loader, num_samples=4)
    
    # Visualizations
    plot_segmentation_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png')
    )
    
    plot_segmentation_results(
        images, predictions, ground_truth,
        save_path=os.path.join(exp_dir, 'segmentation_results.png'),
        num_samples=4
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
    
    # Results
    results = {
        'val_miou': max(history['val_miou']),
        'val_pixel_acc': max(history['val_pixel_acc']),
        'test_acc': max(history['val_miou']),  # Using val_miou as primary metric
        'val_acc': max(history['val_miou']),   # For compatibility with report generator
        'total_time': training_time,
        'hyperparameters': hyperparameters,
        'history': history
    }
    
    print(f"\nΠείραμα '{experiment_name}' ολοκληρώθηκε!")
    print(f"  Best Validation mIoU: {results['val_miou']:.2f}%")
    print(f"  Best Pixel Accuracy: {results['val_pixel_acc']:.2f}%")
    print(f"  Training Time: {training_time:.2f}s")
    print("=" * 80)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SBD Semantic Segmentation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (5 epochs)')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("\n*** QUICK TEST MODE: 5 epochs ***\n")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results_segmentation', f'experiments_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("SBD SEMANTIC SEGMENTATION - U-NET EXPERIMENTS")
    print("=" * 80)
    
    # Load data
    print("\nΦόρτωση SBD Dataset...")
    train_loader, val_loader, num_classes = load_sbd_dataset(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    all_results = {}
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 1: Σύγκριση U-Net sizes
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 1: ΣΥΓΚΡΙΣΗ U-NET SIZES")
    print("=" * 80)
    
    base_config = {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'scheduler': 'step'
    }
    
    for base_ch in [32, 64]:
        config = base_config.copy()
        config['base_channels'] = base_ch
        
        all_results[f'UNet_base{base_ch}'] = run_segmentation_experiment(
            f'UNet_BaseChannels_{base_ch}',
            config,
            train_loader, val_loader, num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 2: Σύγκριση Optimizers
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 2: ΣΥΓΚΡΙΣΗ OPTIMIZERS")
    print("=" * 80)
    
    for opt in ['adam', 'adamw', 'sgd']:
        config = base_config.copy()
        config['optimizer'] = opt
        config['base_channels'] = 64
        
        all_results[f'Optimizer_{opt}'] = run_segmentation_experiment(
            f'Optimizer_{opt.upper()}',
            config,
            train_loader, val_loader, num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 3: Σύγκριση Learning Rates
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 3: ΣΥΓΚΡΙΣΗ LEARNING RATES")
    print("=" * 80)
    
    for lr in [0.0001, 0.001, 0.01]:
        config = base_config.copy()
        config['learning_rate'] = lr
        config['base_channels'] = 64
        
        all_results[f'LR_{lr}'] = run_segmentation_experiment(
            f'LearningRate_{lr}',
            config,
            train_loader, val_loader, num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΑΝΑΦΟΡΕΣ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΔΗΜΙΟΥΡΓΙΑ ΑΝΑΦΟΡΩΝ")
    print("=" * 80)
    
    # Save results
    save_experiment_results(all_results, save_dir=results_dir)
    generate_experiment_report(
        all_results,
        save_path=os.path.join(results_dir, 'experiment_report.txt')
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print("=" * 80)
    for exp_name, exp_data in sorted(all_results.items(), 
                                     key=lambda x: x[1]['val_miou'], 
                                     reverse=True):
        print(f"{exp_name:30s}: mIoU={exp_data['val_miou']:.2f}%, "
              f"PixAcc={exp_data['val_pixel_acc']:.2f}%")
    
    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΣΗ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 80)
    print(f"\nΑποτελέσματα: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
