"""
Άσκηση #4: Object Detection με Faster R-CNN στο Oxford Pet Dataset
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import os
import json
import argparse
from datetime import datetime

from pet_data_loaders import load_pet_detection_dataset
from ex4_detection_model import get_detection_model
from training_utils import DetectionTrainer
from visualization_utils import (
    visualize_detections,
    plot_detection_training_curves,
    plot_comparison_results as plot_detection_comparison,
    generate_experiment_report
)


def run_detection_experiment(
    backbone='resnet50',
    lr=0.005,
    num_epochs=10,
    batch_size=4,
    optimizer_name='sgd',
    scheduler_name='step',
    experiment_name=None,
    quick_test=False
):
    """
    Εκτέλεση ενός detection experiment
    """
    
    if quick_test:
        num_epochs = 3
        print("\n⚡ QUICK TEST MODE - Running only 3 epochs!\n")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"detection_{backbone}_{optimizer_name}_lr{lr}_e{num_epochs}_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"Backbone: {backbone}")
    print(f"Learning Rate: {lr}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load data
    train_loader, val_loader, num_classes = load_pet_detection_dataset(
        batch_size=batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create model
    model = get_detection_model(num_classes=num_classes, backbone=backbone, pretrained=True)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=0.0005)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Scheduler
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
    
    # Train
    trainer = DetectionTrainer(model, device=device)
    history = trainer.train(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=7
    )
    
    # Get sample predictions
    print("\n" + "="*70)
    print("Generating sample predictions...")
    images, predictions = trainer.get_predictions(val_loader, num_samples=4, score_threshold=0.5)
    
    # Create results directory
    results_dir = 'results_detection'
    os.makedirs(results_dir, exist_ok=True)
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save visualizations
    visualize_detections(
        images, predictions,
        save_path=os.path.join(exp_dir, 'detections.png'),
        title=f"Detections - {experiment_name}"
    )
    
    plot_detection_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png')
    )
    
    # Calculate final metrics
    final_loss = history['train_loss'][-1]
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'config': {
            'backbone': backbone,
            'learning_rate': lr,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'device': device
        },
        'metrics': {
            'final_train_loss': final_loss,
            'best_train_loss': min(history['train_loss']),
            'final_classifier_loss': history['train_loss_classifier'][-1],
            'final_box_reg_loss': history['train_loss_box_reg'][-1]
        },
        'history': history
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'history'}, f, indent=4)
    
    print(f"\n✓ Results saved to: {exp_dir}")
    print(f"✓ Final Training Loss: {final_loss:.4f}")
    print("="*70 + "\n")
    
    return results


def run_all_experiments(quick_test=False):
    """Εκτέλεση όλων των experiments"""
    
    all_results = []
    
    print("\n" + "="*70)
    print("OBJECT DETECTION EXPERIMENTS - Oxford Pet Dataset")
    print("Faster R-CNN with different configurations")
    print("="*70 + "\n")
    
    # Experiment 1: Different Learning Rates (ResNet50)
    print("\n📊 SERIES 1: Σύγκριση Learning Rates (ResNet50)")
    print("-" * 70)
    for lr in [0.001, 0.005, 0.01]:
        result = run_detection_experiment(
            backbone='resnet50',
            lr=lr,
            num_epochs=5 if not quick_test else 3,
            batch_size=4,
            optimizer_name='sgd',
            scheduler_name='step',
            experiment_name=f"resnet50_lr{lr}",
            quick_test=quick_test
        )
        all_results.append(result)
    
    # Experiment 2: Different Backbones
    print("\n📊 SERIES 2: Σύγκριση Backbones")
    print("-" * 70)
    for backbone in ['resnet50', 'mobilenet']:
        if backbone == 'resnet50' and any(r['config']['backbone'] == 'resnet50' for r in all_results):
            continue  # Skip if already run
        
        result = run_detection_experiment(
            backbone=backbone,
            lr=0.005,
            num_epochs=5 if not quick_test else 3,
            batch_size=4,
            optimizer_name='sgd',
            scheduler_name='step',
            experiment_name=f"{backbone}_baseline",
            quick_test=quick_test
        )
        all_results.append(result)
    
    # Experiment 3: Different Optimizers (ResNet50)
    print("\n📊 SERIES 3: Σύγκριση Optimizers (ResNet50)")
    print("-" * 70)
    for opt in ['adam', 'adamw']:
        result = run_detection_experiment(
            backbone='resnet50',
            lr=0.001,  # Lower LR for Adam/AdamW
            num_epochs=5 if not quick_test else 3,
            batch_size=4,
            optimizer_name=opt,
            scheduler_name='cosine',
            experiment_name=f"resnet50_{opt}",
            quick_test=quick_test
        )
        all_results.append(result)
    
    # Generate comparison plot
    print("\n" + "="*70)
    print("Generating comparison plots...")
    
    results_for_plot = [
        {
            'name': r['experiment_name'],
            'history': r['history'],
            'final_loss': r['metrics']['final_train_loss']
        }
        for r in all_results
    ]
    
    plot_detection_comparison(
        results_for_plot,
        save_path='results_detection/all_experiments_comparison.png'
    )
    
    # Save summary
    summary = {
        'total_experiments': len(all_results),
        'experiments': [
            {
                'name': r['experiment_name'],
                'config': r['config'],
                'final_loss': r['metrics']['final_train_loss'],
                'best_loss': r['metrics']['best_train_loss']
            }
            for r in all_results
        ]
    }
    
    with open('results_detection/experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENTS SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(all_results)}\n")
    
    # Sort by final loss
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['final_train_loss'])
    
    print("Results (sorted by final training loss):")
    print("-" * 70)
    for i, result in enumerate(sorted_results, 1):
        config = result['config']
        metrics = result['metrics']
        print(f"{i}. {result['experiment_name']}")
        print(f"   Backbone: {config['backbone']} | LR: {config['learning_rate']} | "
              f"Optimizer: {config['optimizer']}")
        print(f"   Final Loss: {metrics['final_train_loss']:.4f} | "
              f"Best Loss: {metrics['best_train_loss']:.4f}")
        print()
    
    print("="*70)
    print(f"✓ All results saved to: results_detection/")
    print(f"✓ Summary saved to: results_detection/experiments_summary.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Experiments')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with 3 epochs')
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'mobilenet'])
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.single:
        # Run single experiment
        run_detection_experiment(
            backbone=args.backbone,
            lr=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            quick_test=args.quick_test
        )
    else:
        # Run all experiments
        run_all_experiments(quick_test=args.quick_test)
