"""
Πειράματα Transfer Learning για Oxford Pet Classification
Άσκηση #2: Μεταφορά μάθησης
"""

import torch
import torch.nn as nn
import time
import os
from datetime import datetime

from pet_data_loaders import load_pet_classification_dataset, PET_CLASSES
from ex2_transfer_learning_models import get_transfer_model
from training_utils import ClassificationTrainer, get_optimizer, get_scheduler
from visualization_utils import (plot_training_curves, plot_comparison_results,
                           visualize_classification_predictions, generate_experiment_report)


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
        learning_rate=hyperparameters.get('learning_rate', 0.001),
        weight_decay=hyperparameters.get('weight_decay', 1e-4)
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=hyperparameters.get('scheduler', 'step'),
        num_epochs=num_epochs
    )
    
    # Training
    trainer = Trainer(model, device=device)
    start_time = time.time()
    history = trainer.train(
        train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler, verbose=True,
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
    
    # Save model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
    
    # Results
    results = {
        'val_acc': max(history['val_acc']),
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Oxford Pet Transfer Learning')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 5 epochs')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("\n*** QUICK TEST MODE: 5 epochs ***\n")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results_transfer', f'experiments_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("OXFORD PET - TRANSFER LEARNING EXPERIMENTS")
    print("=" * 80)
    
    # Load data
    print("\nΦόρτωση Oxford-IIIT Pet Dataset...")
    train_loader, val_loader, test_loader, num_classes = load_oxford_pet(
        batch_size=args.batch_size,
        num_workers=2
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
            arch_results,
            save_path=os.path.join(results_dir, 'comparison_architectures.png')
        )
    
    # Frozen vs Fine-tuned
    mode_results = {k: v for k, v in all_results.items() if k.startswith('Mode_')}
    if mode_results:
        plot_comparison_results(
            mode_results,
            save_path=os.path.join(results_dir, 'comparison_frozen_vs_finetuned.png')
        )
    
    # Learning rates
    lr_results = {k: v for k, v in all_results.items() if k.startswith('LR_')}
    if lr_results:
        plot_comparison_results(
            lr_results,
            save_path=os.path.join(results_dir, 'comparison_learning_rates.png')
        )
    
    # Overall comparison
    plot_comparison_results(
        all_results,
        save_path=os.path.join(results_dir, 'comparison_all_experiments.png')
    )
    
    # Save results
    save_experiment_results(all_results, save_dir=results_dir)
    generate_experiment_report(
        all_results,
        save_path=os.path.join(results_dir, 'experiment_report.txt')
    )
    
    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΣΗ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 80)
    print(f"\nΑποτελέσματα: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
