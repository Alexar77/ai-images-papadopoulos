"""
Απλό script για γρήγορη εκτέλεση ενός μόνο πειράματος
"""

import torch
from ex1_vanilla_cnn import VanillaCNN, get_model_summary
from ex1_data_loader import load_cifar100
from train_utils import Trainer, get_criterion, get_optimizer, get_scheduler
from visualization import plot_training_curves, plot_sample_predictions
from ex1_data_loader import CIFAR100_CLASSES
import os


def quick_experiment(num_epochs=10):
    """
    Γρήγορο πείραμα με default παραμέτρους
    """
    
    print("=" * 80)
    print("ΓΡΗΓΟΡΟ ΠΕΙΡΑΜΑ - VANILLA CNN ΣΤΟ CIFAR-100")
    print("=" * 80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nΦόρτωση δεδομένων...")
    train_loader, val_loader, test_loader = load_cifar100(batch_size=128)
    
    # Create model
    print("\nΔημιουργία μοντέλου...")
    model = VanillaCNN(num_classes=100)
    get_model_summary(model)
    
    # Training setup
    criterion = get_criterion('cross_entropy')
    optimizer = get_optimizer(model, optimizer_name='sgd', learning_rate=0.01)
    scheduler = get_scheduler(optimizer, scheduler_name='cosine', num_epochs=num_epochs)
    
    # Train
    print(f"\nΕκπαίδευση για {num_epochs} epochs...")
    trainer = Trainer(model, device=device)
    history = trainer.train(
        train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler
    )
    
    # Test
    test_loss, test_acc = trainer.test(test_loader, criterion)
    
    # Visualizations
    print("\nΔημιουργία γραφημάτων...")
    os.makedirs('quick_results', exist_ok=True)
    
    plot_training_curves(
        history,
        save_path='quick_results/training_curves.png'
    )
    
    images, predictions, labels = trainer.get_predictions(test_loader, num_samples=16)
    plot_sample_predictions(
        images, predictions, labels, CIFAR100_CLASSES,
        save_path='quick_results/predictions.png'
    )
    
    # Save model
    torch.save(model.state_dict(), 'quick_results/model.pth')
    
    print("\n" + "=" * 80)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("=" * 80)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nΤα αποτελέσματα αποθηκεύτηκαν στο φάκελο 'quick_results/'")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick CIFAR-100 experiment')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of epochs (default: 10)')
    args = parser.parse_args()
    
    quick_experiment(num_epochs=args.epochs)
