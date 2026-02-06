"""
Πειράματα Transfer Learning για Oxford Pet Classification
Άσκηση #2: Μεταφορά μάθησης
"""

import torch
import torch.nn as nn
import time
import os
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from pet_data_loaders import load_pet_classification_dataset, PET_CLASSES
from ex2_transfer_learning_models import get_transfer_model
from training_utils import ClassificationTrainer, get_optimizer, get_scheduler
from visualization_utils import (plot_training_curves, plot_comparison_results,
                           visualize_classification_predictions, generate_experiment_report)


def save_experiment_results(all_results, save_dir):
    """Save experiment results to JSON."""
    output_path = os.path.join(save_dir, 'experiment_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved: {output_path}")


def run_transfer_experiment(experiment_name, model_name, hyperparameters,
                           train_loader, val_loader, test_loader, 
                           num_classes, num_epochs=20, results_dir='results_transfer'):
    """
    Εκτέλεση ενός transfer learning πειράματος
    """
    
    print("\n" + "=" * 80)
    print(f"ΕΝΑΡΞΗ ΠΕΙΡΑΜΑΤΟΣ: {experiment_name}")
    print("=" * 80)
    print("\nΥπερ-παράμετροι:")
    print(f"  Model: {model_name}")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Δημιουργία φακέλου
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Δημιουργία μοντέλου
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, model_info = get_transfer_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=hyperparameters.get('pretrained', True),
        freeze_features=hyperparameters.get('freeze_features', True)
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
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
    
    # Training
    trainer = ClassificationTrainer(model, device=device)
    start_time = time.time()
    history = trainer.train(
        train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler,
        early_stopping_patience=5
    )
    training_time = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = trainer.test(test_loader, criterion)
    
    # Visualizations
    plot_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png'),
        title=f'{experiment_name} - Training Curves'
    )

    images, labels, predictions, probabilities = trainer.get_predictions(
        test_loader, num_samples=12
    )
    visualize_classification_predictions(
        images, labels, predictions, probabilities,
        class_names=PET_CLASSES,
        save_path=os.path.join(exp_dir, 'sample_predictions.png'),
        title=f'{experiment_name} - Sample Predictions'
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
    
    # Results
    best_val_acc = max(history['val_acc']) if history.get('val_acc') else 0.0
    results = {
        'name': experiment_name,
        'config': {
            'model_name': model_name,
            **hyperparameters
        },
        'metrics': {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'total_training_time': training_time,
            'parameters': {'total': model_info['total_params']}
        },
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_time': training_time,
        'hyperparameters': hyperparameters,
        'model_info': model_info,
        'history': history
    }
    
    print(f"\nΠείραμα '{experiment_name}' ολοκληρώθηκε!")
    print(f"  Best Validation Accuracy: {results['val_acc']:.2f}%")
    print(f"  Test Accuracy: {results['test_acc']:.2f}%")
    print(f"  Training Time: {training_time:.2f}s")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Oxford Pet Transfer Learning')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--results_dir', type=str, default='results_transfer', help='Results directory')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 5 epochs')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("\n*** QUICK TEST MODE: 5 epochs ***\n")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f'experiments_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("OXFORD PET - TRANSFER LEARNING EXPERIMENTS")
    print("=" * 80)
    
    # Load data
    print("\nΦόρτωση Oxford-IIIT Pet Dataset...")
    train_loader, test_loader, num_classes = load_pet_classification_dataset(
        batch_size=args.batch_size,
        num_workers=2
    )
    train_dataset = train_loader.dataset
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_results = {}
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 1: Σύγκριση Αρχιτεκτονικών
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 1: ΣΥΓΚΡΙΣΗ ΑΡΧΙΤΕΚΤΟΝΙΚΩΝ")
    print("=" * 80)
    
    architectures = ['resnet18', 'resnet50', 'alexnet', 'vgg16']
    base_config = {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'scheduler': 'step',
        'pretrained': True,
        'freeze_features': True
    }
    
    for arch in architectures:
        all_results[f'Arch_{arch}'] = run_transfer_experiment(
            f'Architecture_{arch}',
            arch,
            base_config,
            train_loader, val_loader, test_loader,
            num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 2: Frozen vs Fine-tuned
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 2: FROZEN VS FINE-TUNED")
    print("=" * 80)
    
    for freeze in [True, False]:
        config = base_config.copy()
        config['freeze_features'] = freeze
        mode = 'Frozen' if freeze else 'FineTuned'
        
        all_results[f'Mode_{mode}'] = run_transfer_experiment(
            f'ResNet18_{mode}',
            'resnet18',
            config,
            train_loader, val_loader, test_loader,
            num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 3: Σύγκριση Learning Rates
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 3: ΣΥΓΚΡΙΣΗ LEARNING RATES")
    print("=" * 80)
    
    learning_rates = [0.0001, 0.001, 0.01]
    
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        
        all_results[f'LR_{lr}'] = run_transfer_experiment(
            f'LearningRate_{lr}',
            'resnet18',
            config,
            train_loader, val_loader, test_loader,
            num_classes,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΑΝΑΦΟΡΕΣ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΔΗΜΙΟΥΡΓΙΑ ΑΝΑΦΟΡΩΝ")
    print("=" * 80)
    
    # Architecture comparison
    arch_results = {k: v for k, v in all_results.items() if k.startswith('Arch_')}
    if arch_results:
        plot_comparison_results(
            list(arch_results.values()),
            save_path=os.path.join(results_dir, 'comparison_architectures.png')
        )
    
    # Frozen vs Fine-tuned
    mode_results = {k: v for k, v in all_results.items() if k.startswith('Mode_')}
    if mode_results:
        plot_comparison_results(
            list(mode_results.values()),
            save_path=os.path.join(results_dir, 'comparison_frozen_vs_finetuned.png')
        )
    
    # Learning rates
    lr_results = {k: v for k, v in all_results.items() if k.startswith('LR_')}
    if lr_results:
        plot_comparison_results(
            list(lr_results.values()),
            save_path=os.path.join(results_dir, 'comparison_learning_rates.png')
        )
    
    # Overall comparison
    plot_comparison_results(
        list(all_results.values()),
        save_path=os.path.join(results_dir, 'comparison_all_experiments.png')
    )
    
    # Save results
    save_experiment_results(all_results, save_dir=results_dir)
    generate_experiment_report(
        list(all_results.values()),
        save_path=os.path.join(results_dir, 'experiment_report.txt')
    )
    
    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΣΗ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 80)
    print(f"\nΑποτελέσματα: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
