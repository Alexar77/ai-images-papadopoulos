"""
Πειράματα Semantic Segmentation με U-Net
Άσκηση #3: Σημασιολογική Τμηματοποίηση
"""

import torch
import torch.nn as nn
import time
import os
import json
import argparse
import shutil
from datetime import datetime

from ex3_sbd_data_loader import load_sbd_dataset
from ex3_unet_model import get_unet_model
from training_utils import SegmentationTrainer, get_optimizer, get_scheduler
from visualization_utils import (
    plot_segmentation_training_curves,
    visualize_segmentation,
    generate_experiment_report,
    plot_ex3_miou_overview,
    plot_ex3_optimizer_miou_curves,
    plot_ex3_optimizer_pixel_acc_curves,
    plot_ex3_lr_val_loss_curves,
    plot_ex3_time_vs_miou,
    plot_ex3_lr_vs_miou
)


def save_experiment_results(all_results, save_dir):
    """Save experiment results to JSON."""
    output_path = os.path.join(save_dir, 'experiment_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved: {output_path}")


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
    # Reduce background dominance: penalize background errors less than foreground.
    bg_weight = hyperparameters.get('background_weight', 0.2)
    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[0] = bg_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = get_optimizer(
        model,
        optimizer_name=hyperparameters.get('optimizer', 'adam'),
        lr=hyperparameters.get('learning_rate', 0.001),
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
        early_stopping_patience=hyperparameters.get('early_stopping_patience', 7)
    )
    training_time = time.time() - start_time
    
    # Get sample predictions
    images, ground_truth, predictions = trainer.get_predictions(val_loader, num_samples=4)
    
    # Visualizations
    plot_segmentation_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png')
    )
    
    visualize_segmentation(
        images, ground_truth, predictions,
        save_path=os.path.join(exp_dir, 'segmentation_results.png'),
        title=f'{experiment_name} - Segmentation Results'
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
    
    # Results
    best_val_miou = max(history['val_mean_iou']) if history.get('val_mean_iou') else 0.0
    best_val_pixel_acc = max(history['val_pixel_acc']) if history.get('val_pixel_acc') else 0.0
    results = {
        'name': experiment_name,
        'config': hyperparameters,
        'metrics': {
            'best_val_acc': best_val_miou * 100.0,
            'test_acc': best_val_miou * 100.0,
            'best_val_miou': best_val_miou,
            'best_val_pixel_acc': best_val_pixel_acc,
            'total_training_time': training_time,
            'parameters': {'total': model.count_parameters()}
        },
        'val_miou': best_val_miou * 100.0,
        'val_pixel_acc': best_val_pixel_acc * 100.0,
        'test_acc': best_val_miou * 100.0,
        'val_acc': best_val_miou * 100.0,
        'total_time': training_time,
        'hyperparameters': hyperparameters,
        'history': history
    }
    
    print(f"\nΠείραμα '{experiment_name}' ολοκληρώθηκε!")
    print(f"  Best Validation mIoU (foreground classes): {results['val_miou']:.2f}%")
    print(f"  Best Pixel Accuracy: {results['val_pixel_acc']:.2f}%")
    print(f"  Training Time: {training_time:.2f}s")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SBD Semantic Segmentation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--results_dir', type=str, default='results_segmentation', help='Results directory')
    parser.add_argument('--quick_test', action='store_true', help='Quick test (3 epochs)')
    parser.add_argument('--background_weight', type=float, default=0.2,
                        help='Loss weight for background class (class 0). Lower => less background bias.')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 3
        print("\n*** QUICK TEST MODE: 3 epochs ***\n")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f'experiments_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("SBD SEMANTIC SEGMENTATION - U-NET EXPERIMENTS")
    print("=" * 80)
    
    # Load data
    print("\nΦόρτωση SBD Dataset...")
    train_loader, val_loader, num_classes = load_sbd_dataset(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    all_results = {}
    
    print("\n[Mode] Assignment-3: 2 hyperparameters x 2 values")
    print("Hyperparameters under study: optimizer (2 values), learning_rate (2 values)")

    base_config = {
        'weight_decay': 1e-4,
        'scheduler': 'step',
        'base_channels': 64,
        'early_stopping_patience': 4,
        'background_weight': args.background_weight
    }

    # Hyperparameter 1: optimizer (fixed lr)
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 1: ΣΥΓΚΡΙΣΗ OPTIMIZERS (2 τιμές)")
    print("=" * 80)
    for opt in ['adamw', 'sgd']:
        config = base_config.copy()
        config['optimizer'] = opt
        config['learning_rate'] = 0.0003

        all_results[f'Optimizer_{opt}'] = run_segmentation_experiment(
            f'Optimizer_{opt.upper()}',
            config,
            train_loader, val_loader, num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )

    # Hyperparameter 2: learning rate (fixed optimizer)
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 2: ΣΥΓΚΡΙΣΗ LEARNING RATES (2 τιμές)")
    print("=" * 80)
    for lr in [0.0003, 0.001]:
        config = base_config.copy()
        config['optimizer'] = 'adamw'
        config['learning_rate'] = lr

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
        list(all_results.values()),
        save_path=os.path.join(results_dir, 'experiment_report.txt')
    )

    # Report-focused Ex3 figures (5-6 key plots)
    plot_ex3_miou_overview(
        all_results,
        save_path=os.path.join(results_dir, 'report_01_miou_overview.png')
    )
    plot_ex3_optimizer_miou_curves(
        all_results,
        save_path=os.path.join(results_dir, 'report_02_optimizer_miou_curves.png')
    )
    plot_ex3_optimizer_pixel_acc_curves(
        all_results,
        save_path=os.path.join(results_dir, 'report_03_optimizer_pixelacc_curves.png')
    )
    plot_ex3_lr_val_loss_curves(
        all_results,
        save_path=os.path.join(results_dir, 'report_04_lr_val_loss_curves.png')
    )
    plot_ex3_time_vs_miou(
        all_results,
        save_path=os.path.join(results_dir, 'report_05_time_vs_miou.png')
    )
    plot_ex3_lr_vs_miou(
        all_results,
        save_path=os.path.join(results_dir, 'report_06_lr_vs_miou.png')
    )

    # Best qualitative panel (Input / GT / Prediction) copied from best run
    best_result = max(all_results.values(), key=lambda x: x['val_miou'])
    best_exp_name = best_result['name']
    src_panel = os.path.join(results_dir, best_exp_name, 'segmentation_results.png')
    dst_panel = os.path.join(results_dir, 'report_07_best_segmentation_panel.png')
    if os.path.exists(src_panel):
        shutil.copyfile(src_panel, dst_panel)
        print(f"✓ Ex3 best segmentation panel saved: {dst_panel}")
    
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
