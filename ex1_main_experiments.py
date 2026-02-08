"""
Κύριο script για την εκτέλεση πειραμάτων σύγκρισης υπερ-παραμέτρων
Exercise #1: Βελτιστοποίηση αρχιτεκτονικής και υπερ-παραμέτρων
"""

import torch
import time
import os
import argparse
import json
from datetime import datetime

from ex1_vanilla_cnn import VanillaCNN
from cifar_data_loaders import load_cifar100_dataset
from training_utils import ClassificationTrainer, get_criterion, get_optimizer, get_scheduler
from visualization_utils import (plot_training_curves, plot_comparison_results, 
                           visualize_classification_predictions, generate_experiment_report,
                           plot_ex1_accuracy_overview, plot_ex1_optimizer_train_accuracy_curves,
                           plot_ex1_learning_rate_loss_curves, plot_ex1_generalization_gap,
                           plot_ex1_time_vs_accuracy, plot_ex1_learning_rate_vs_accuracy)


def save_experiment_results(all_results, save_dir):
    """Save experiment results to JSON."""
    output_path = os.path.join(save_dir, 'experiment_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved: {output_path}")


def run_experiment(experiment_name, hyperparameters, train_loader, 
                   test_loader, num_epochs=30, results_dir='results'):
    """
    Εκτέλεση ενός πειράματος με συγκεκριμένες υπερ-παραμέτρους
    
    Args:
        experiment_name: Όνομα πειράματος
        hyperparameters: Dictionary με υπερ-παραμέτρους
        train_loader, test_loader: Data loaders
        num_epochs: Αριθμός epochs
        results_dir: Φάκελος αποθήκευσης αποτελεσμάτων
    """
    
    print("\n" + "=" * 80)
    print(f"ΕΝΑΡΞΗ ΠΕΙΡΑΜΑΤΟΣ: {experiment_name}")
    print("=" * 80)
    print("\nΥπερ-παράμετροι:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Create experiment directory
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VanillaCNN(num_classes=100)
    
    # Setup training components
    criterion = get_criterion(hyperparameters.get('criterion', 'crossentropy'))
    optimizer = get_optimizer(
        model,
        optimizer_name=hyperparameters.get('optimizer', 'sgd'),
        lr=hyperparameters.get('learning_rate', 0.01),
        weight_decay=hyperparameters.get('weight_decay', 5e-4)
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=hyperparameters.get('scheduler', 'cosine'),
        num_epochs=num_epochs
    )
    
    # Initialize trainer
    trainer = ClassificationTrainer(model, device=device)
    
    # Train
    start_time = time.time()
    history = trainer.train(
        train_loader, None, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler,
        early_stopping_patience=10
    )
    training_time = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = trainer.test(test_loader, criterion)
    
    # Get sample predictions
    images, labels, predictions, probabilities = trainer.get_predictions(
        test_loader, num_samples=16
    )
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(exp_dir, 'training_curves.png'),
        title=f'{experiment_name} - Training Curves'
    )
    
    # Plot sample predictions
    class_names = [f'Class {i}' for i in range(100)]
    visualize_classification_predictions(
        images, labels, predictions, probabilities,
        class_names=class_names,
        save_path=os.path.join(exp_dir, 'sample_predictions.png')
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
    
    # Prepare results
    best_train_acc = max(history['train_acc']) if history.get('train_acc') else 0.0
    results = {
        'name': experiment_name,
        'config': hyperparameters,
        'metrics': {
            'best_train_acc': best_train_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'total_training_time': training_time,
            'parameters': {'total': model.count_parameters()}
        },
        'train_acc': best_train_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_time': training_time,
        'hyperparameters': hyperparameters,
        'history': history
    }
    
    print(f"\nΠείραμα '{experiment_name}' ολοκληρώθηκε!")
    print(f"  Best Training Accuracy: {results['train_acc']:.2f}%")
    print(f"  Test Accuracy: {results['test_acc']:.2f}%")
    print(f"  Training Time: {training_time:.2f} seconds")
    print("=" * 80)
    
    return results


def main():
    """Κύρια συνάρτηση για την εκτέλεση όλων των πειραμάτων"""
    
    parser = argparse.ArgumentParser(description='CIFAR-100 Vanilla CNN Experiments')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test with few epochs')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("\n*** QUICK TEST MODE: Running with only 5 epochs ***\n")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f'experiments_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("CIFAR-100 VANILLA CNN - HYPERPARAMETER COMPARISON")
    print("=" * 80)
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    print("\nΦόρτωση δεδομένων CIFAR-100...")
    train_loader, _, test_loader, num_classes = load_cifar100_dataset(
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Initialize results dictionary
    all_results = {}
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 1: Σύγκριση Loss Functions
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 1: ΣΥΓΚΡΙΣΗ ΣΥΝΑΡΤΗΣΕΩΝ ΚΟΣΤΟΥΣ")
    print("=" * 80)
    
    base_config = {
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'scheduler': 'cosine'
    }
    
    # CrossEntropyLoss (βασική επιλογή)
    exp_config = base_config.copy()
    exp_config['criterion'] = 'crossentropy'
    all_results['Loss_CrossEntropy'] = run_experiment(
        'Loss_CrossEntropy',
        exp_config,
        train_loader, test_loader,
        num_epochs=args.epochs,
        results_dir=results_dir
    )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 2: Σύγκριση Optimizers
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 2: ΣΥΓΚΡΙΣΗ ΜΕΘΟΔΩΝ ΒΕΛΤΙΣΤΟΠΟΙΗΣΗΣ")
    print("=" * 80)
    
    base_config = {
        'criterion': 'crossentropy',
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'scheduler': 'cosine'
    }
    
    # SGD
    exp_config = base_config.copy()
    exp_config['optimizer'] = 'sgd'
    all_results['Optimizer_SGD'] = run_experiment(
        'Optimizer_SGD',
        exp_config,
        train_loader, test_loader,
        num_epochs=args.epochs,
        results_dir=results_dir
    )
    
    # Adam
    exp_config = base_config.copy()
    exp_config['optimizer'] = 'adam'
    all_results['Optimizer_Adam'] = run_experiment(
        'Optimizer_Adam',
        exp_config,
        train_loader, test_loader,
        num_epochs=args.epochs,
        results_dir=results_dir
    )
    
    # AdamW
    exp_config = base_config.copy()
    exp_config['optimizer'] = 'adamw'
    all_results['Optimizer_AdamW'] = run_experiment(
        'Optimizer_AdamW',
        exp_config,
        train_loader, test_loader,
        num_epochs=args.epochs,
        results_dir=results_dir
    )
    
    # RMSprop
    exp_config = base_config.copy()
    exp_config['optimizer'] = 'rmsprop'
    all_results['Optimizer_RMSprop'] = run_experiment(
        'Optimizer_RMSprop',
        exp_config,
        train_loader, test_loader,
        num_epochs=args.epochs,
        results_dir=results_dir
    )
    
    # ========================================================================
    # ΠΕΙΡΑΜΑ 3: Σύγκριση Learning Rates
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΣΕΙΡΑ ΠΕΙΡΑΜΑΤΩΝ 3: ΣΥΓΚΡΙΣΗ ΡΥΘΜΩΝ ΕΚΠΑΙΔΕΥΣΗΣ")
    print("=" * 80)
    
    base_config = {
        'criterion': 'crossentropy',
        'optimizer': 'sgd',
        'weight_decay': 5e-4,
        'scheduler': 'cosine'
    }
    
    learning_rates = [0.001, 0.01, 0.1]
    
    for lr in learning_rates:
        exp_config = base_config.copy()
        exp_config['learning_rate'] = lr
        all_results[f'LR_{lr}'] = run_experiment(
            f'LearningRate_{lr}',
            exp_config,
            train_loader, test_loader,
            num_epochs=args.epochs,
            results_dir=results_dir
        )
    
    # ========================================================================
    # ΣΥΓΚΡΙΣΗ ΚΑΙ ΑΝΑΦΟΡΑ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
    # ========================================================================
    print("\n" + "=" * 80)
    print("ΔΗΜΙΟΥΡΓΙΑ ΣΥΓΚΡΙΤΙΚΩΝ ΑΝΑΦΟΡΩΝ")
    print("=" * 80)
    
    # Comparison plots for each experiment series
    
    # 1. Loss function comparison
    loss_results = {k: v for k, v in all_results.items() if k.startswith('Loss_')}
    if loss_results:
        plot_comparison_results(
            list(loss_results.values()),
            save_path=os.path.join(results_dir, 'comparison_loss_functions.png')
        )
    
    # 2. Optimizer comparison
    opt_results = {k: v for k, v in all_results.items() if k.startswith('Optimizer_')}
    if opt_results:
        plot_comparison_results(
            list(opt_results.values()),
            save_path=os.path.join(results_dir, 'comparison_optimizers.png')
        )
    
    # 3. Learning rate comparison
    lr_results = {k: v for k, v in all_results.items() if k.startswith('LR_')}
    if lr_results:
        plot_comparison_results(
            list(lr_results.values()),
            save_path=os.path.join(results_dir, 'comparison_learning_rates.png')
        )
    
    # 4. Overall comparison
    plot_comparison_results(
        list(all_results.values()),
        save_path=os.path.join(results_dir, 'comparison_all_experiments.png')
    )

    # 5. Report-focused Ex1 figures (5-6 key plots)
    plot_ex1_accuracy_overview(
        all_results,
        save_path=os.path.join(results_dir, 'report_01_accuracy_overview.png')
    )
    plot_ex1_optimizer_train_accuracy_curves(
        all_results,
        save_path=os.path.join(results_dir, 'report_02_optimizer_curves.png')
    )
    plot_ex1_learning_rate_loss_curves(
        all_results,
        save_path=os.path.join(results_dir, 'report_03_lr_train_loss_curves.png')
    )
    plot_ex1_generalization_gap(
        all_results,
        save_path=os.path.join(results_dir, 'report_04_generalization_gap.png')
    )
    plot_ex1_time_vs_accuracy(
        all_results,
        save_path=os.path.join(results_dir, 'report_05_time_vs_accuracy.png')
    )
    plot_ex1_learning_rate_vs_accuracy(
        all_results,
        save_path=os.path.join(results_dir, 'report_06_lr_vs_accuracy.png')
    )
    
    # Save results to JSON
    save_experiment_results(all_results, save_dir=results_dir)
    
    # Generate text report
    generate_experiment_report(
        list(all_results.values()),
        save_path=os.path.join(results_dir, 'experiment_report.txt')
    )
    
    print("\n" + "=" * 80)
    print("ΟΛΟΚΛΗΡΩΣΗ ΟΛΩΝ ΤΩΝ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 80)
    print(f"\nΌλα τα αποτελέσματα αποθηκεύτηκαν στο: {results_dir}")
    print("\nΠεριεχόμενα:")
    print("  - training_curves.png για κάθε πείραμα")
    print("  - sample_predictions.png για κάθε πείραμα")
    print("  - comparison_*.png για συγκριτικά γραφήματα")
    print("  - experiment_results.json με αριθμητικά αποτελέσματα")
    print("  - experiment_report.txt με πλήρη αναφορά")
    print("=" * 80)


if __name__ == "__main__":
    main()
