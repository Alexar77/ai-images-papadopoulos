"""
Άσκηση #5: Συγκριτική Αξιολόγηση CNN vs Transformer - CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import json
import argparse
from datetime import datetime

from cifar_data_loaders import load_cifar10_dataset
from ex5_cnn_models import get_cnn_model
from ex5_vit_model import get_vit_model
from training_utils import ClassificationTrainer, count_parameters
from visualization_utils import (
    plot_training_curves,
    plot_comparative_training_curves,
    plot_model_comparison_bars,
    visualize_classification_predictions as visualize_predictions,
    generate_experiment_report as generate_comparative_report,
    CIFAR10_CLASSES
)


def run_comparative_experiment(
    architecture='cnn',
    model_name='vgg11',
    lr=0.001,
    num_epochs=20,
    batch_size=128,
    optimizer_name='adam',
    scheduler_name='cosine',
    image_size=32,
    experiment_name=None,
    quick_test=False,
    results_root='results_comparative'
):
    """
    Εκτέλεση ενός comparative experiment
    """
    
    if quick_test:
        num_epochs = 5
        print("\n⚡ QUICK TEST MODE - Running only 5 epochs!\n")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{architecture}_{model_name}_lr{lr}_e{num_epochs}_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"Architecture: {architecture.upper()}")
    print(f"Model: {model_name}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Learning Rate: {lr}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load data
    train_loader, _, test_loader, num_classes = load_cifar10_dataset(
        batch_size=batch_size,
        image_size=image_size,
        augment=True,
        num_workers=0
    )
    
    # Create model
    if architecture == 'cnn':
        model = get_cnn_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    elif architecture == 'vit':
        model = get_vit_model(variant=model_name, num_classes=num_classes, img_size=image_size)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Count parameters
    param_info = count_parameters(model)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Scheduler
    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    else:
        scheduler = None
    
    # Train
    trainer = ClassificationTrainer(model, device=device)
    history = trainer.train(
        train_loader,
        None,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=7
    )
    
    # Test
    test_loss, test_acc = trainer.test(test_loader, criterion)
    
    # Get sample predictions
    print("\n" + "="*70)
    print("Generating sample predictions...")
    images, labels, predictions, probabilities = trainer.get_predictions(
        test_loader, num_samples=10
    )
    
    # Create results directory
    results_dir = results_root
    os.makedirs(results_dir, exist_ok=True)
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save visualizations
    plot_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png'),
        title=f"Training Curves - {experiment_name}"
    )

    visualize_predictions(
        images, labels, predictions, probabilities,
        class_names=CIFAR10_CLASSES,
        save_path=os.path.join(exp_dir, 'predictions.png'),
        title=f"Predictions - {experiment_name}"
    )
    
    # Save results
    results = {
        'name': experiment_name,
        'test_acc': test_acc,
        'experiment_name': experiment_name,
        'config': {
            'architecture': architecture,
            'model_name': model_name,
            'image_size': image_size,
            'learning_rate': lr,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'device': device
        },
        'metrics': {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'final_train_loss': history['train_loss'][-1],
            'parameters': param_info,
            'total_training_time': sum(history['epoch_times']),
            'inference_time': history['inference_time']
        },
        'history': history
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {k: v for k, v in results.items() if k != 'history'}
        json.dump(json_results, f, indent=4)
    
    print(f"\n✓ Results saved to: {exp_dir}")
    print(f"✓ Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")
    
    return results


def run_all_experiments(quick_test=False, results_root='results_comparative'):
    """Εκτέλεση όλων των comparative experiments"""
    
    all_results = []
    
    print("\n" + "="*70)
    print("COMPARATIVE STUDY: CNN vs TRANSFORMER - CIFAR-10")
    print("="*70 + "\n")
    
    # ============================================================================
    # SERIES 1: CNN ARCHITECTURES
    # ============================================================================
    print("\n📊 SERIES 1: CNN Architectures")
    print("-" * 70)
    
    cnn_configs = [
        {'model_name': 'vgg11', 'lr': 0.001, 'image_size': 32},
        {'model_name': 'resnet18', 'lr': 0.001, 'image_size': 32},
        {'model_name': 'resnet50', 'lr': 0.001, 'image_size': 32},
    ]
    
    for config in cnn_configs:
        result = run_comparative_experiment(
            architecture='cnn',
            model_name=config['model_name'],
            lr=config['lr'],
            num_epochs=20 if not quick_test else 5,
            batch_size=128,
            optimizer_name='adam',
            scheduler_name='cosine',
            image_size=config['image_size'],
            experiment_name=f"cnn_{config['model_name']}_baseline",
            quick_test=quick_test,
            results_root=results_root
        )
        all_results.append(result)
    
    # ============================================================================
    # SERIES 2: VISION TRANSFORMERS
    # ============================================================================
    print("\n📊 SERIES 2: Vision Transformers")
    print("-" * 70)
    
    vit_configs = [
        {'model_name': 'tiny', 'lr': 0.001, 'batch_size': 64},
        {'model_name': 'small', 'lr': 0.0005, 'batch_size': 32},
    ]
    
    for config in vit_configs:
        result = run_comparative_experiment(
            architecture='vit',
            model_name=config['model_name'],
            lr=config['lr'],
            num_epochs=30 if not quick_test else 5,
            batch_size=config['batch_size'],
            optimizer_name='adamw',
            scheduler_name='cosine',
            image_size=224,  # ViT needs larger images
            experiment_name=f"vit_{config['model_name']}_baseline",
            quick_test=quick_test,
            results_root=results_root
        )
        all_results.append(result)
    
    # ============================================================================
    # SERIES 3: HYPERPARAMETER COMPARISON (Best Models)
    # ============================================================================
    print("\n📊 SERIES 3: Hyperparameter Comparison")
    print("-" * 70)
    
    # Best CNN with different LRs
    for lr in [0.0005, 0.002]:
        result = run_comparative_experiment(
            architecture='cnn',
            model_name='resnet18',
            lr=lr,
            num_epochs=20 if not quick_test else 5,
            batch_size=128,
            optimizer_name='adam',
            scheduler_name='cosine',
            image_size=32,
            experiment_name=f"cnn_resnet18_lr{lr}",
            quick_test=quick_test,
            results_root=results_root
        )
        all_results.append(result)
    
    # Best ViT with different optimizers
    result = run_comparative_experiment(
        architecture='vit',
        model_name='tiny',
        lr=0.001,
        num_epochs=30 if not quick_test else 5,
        batch_size=64,
        optimizer_name='adam',  # Try Adam instead of AdamW
        scheduler_name='cosine',
        image_size=224,
        experiment_name=f"vit_tiny_adam",
        quick_test=quick_test,
        results_root=results_root
    )
    all_results.append(result)
    
    # ============================================================================
    # GENERATE COMPARISON VISUALIZATIONS
    # ============================================================================
    print("\n" + "="*70)
    print("Generating comprehensive comparison visualizations...")
    
    # Training curves comparison
    plot_comparative_training_curves(
        results=all_results,
        save_path=os.path.join(results_root, 'all_training_curves.png')
    )
    
    # Model comparison bars
    plot_model_comparison_bars(
        results=all_results,
        save_path=os.path.join(results_root, 'model_comparison.png')
    )
    
    # Generate detailed report
    generate_comparative_report(
        results=all_results,
        save_path=os.path.join(results_root, 'comparative_report.txt')
    )
    
    # Save summary JSON
    summary = {
        'total_experiments': len(all_results),
        'experiments': [
            {
                'name': r['experiment_name'],
                'architecture': r['config']['architecture'],
                'model': r['config']['model_name'],
                'test_acc': r['metrics']['test_acc'],
                'parameters': r['metrics']['parameters']['total'],
                'training_time': r['metrics']['total_training_time']
            }
            for r in all_results
        ]
    }
    
    with open(os.path.join(results_root, 'experiments_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # ============================================================================
    # PRINT FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("EXPERIMENTS SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(all_results)}\n")
    
    # Group by architecture
    cnn_results = [r for r in all_results if r['config']['architecture'] == 'cnn']
    vit_results = [r for r in all_results if r['config']['architecture'] == 'vit']
    
    print("\n📊 CNN MODELS:")
    print("-" * 70)
    cnn_sorted = sorted(cnn_results, key=lambda x: x['metrics']['test_acc'], reverse=True)
    for i, result in enumerate(cnn_sorted, 1):
        print(f"{i}. {result['config']['model_name'].upper()}")
        print(f"   Test Acc: {result['metrics']['test_acc']:.2f}% | "
              f"Params: {result['metrics']['parameters']['total_millions']:.1f}M | "
              f"Time: {result['metrics']['total_training_time']/60:.1f}m")
    
    print("\n📊 TRANSFORMER MODELS:")
    print("-" * 70)
    vit_sorted = sorted(vit_results, key=lambda x: x['metrics']['test_acc'], reverse=True)
    for i, result in enumerate(vit_sorted, 1):
        print(f"{i}. ViT-{result['config']['model_name'].upper()}")
        print(f"   Test Acc: {result['metrics']['test_acc']:.2f}% | "
              f"Params: {result['metrics']['parameters']['total_millions']:.1f}M | "
              f"Time: {result['metrics']['total_training_time']/60:.1f}m")
    
    print("\n📊 OVERALL WINNER:")
    print("-" * 70)
    best_result = max(all_results, key=lambda x: x['metrics']['test_acc'])
    print(f"🏆 {best_result['experiment_name']}")
    print(f"   Architecture: {best_result['config']['architecture'].upper()}")
    print(f"   Model: {best_result['config']['model_name']}")
    print(f"   Test Accuracy: {best_result['metrics']['test_acc']:.2f}%")
    print(f"   Parameters: {best_result['metrics']['parameters']['total_millions']:.1f}M")
    
    print("\n" + "="*70)
    print(f"✓ All results saved to: {results_root}/")
    print(f"✓ Detailed report: {results_root}/comparative_report.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparative Study: CNN vs Transformer')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with 5 epochs')
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment')
    parser.add_argument('--architecture', type=str, default='cnn',
                       choices=['cnn', 'vit'])
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--results_dir', type=str, default='results_comparative')
    
    args = parser.parse_args()
    
    if args.single:
        # Run single experiment
        run_comparative_experiment(
            architecture=args.architecture,
            model_name=args.model,
            lr=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            quick_test=args.quick_test,
            results_root=args.results_dir
        )
    else:
        # Run all experiments
        run_all_experiments(quick_test=args.quick_test, results_root=args.results_dir)
