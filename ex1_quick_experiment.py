"""
Απλό script για γρήγορη εκτέλεση ενός μόνο πειράματος
"""

import torch
from ex1_vanilla_cnn import VanillaCNN, get_model_summary
from cifar_data_loaders import load_cifar100_dataset
from training_utils import ClassificationTrainer, get_criterion, get_optimizer, get_scheduler
from visualization_utils import plot_training_curves, visualize_classification_predictions
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
    train_loader, val_loader, test_loader, _ = load_cifar100_dataset(batch_size=128)
    
    # Create model
    print("\nΔημιουργία μοντέλου...")
    model = VanillaCNN(num_classes=100)
    get_model_summary(model)
    
    # Training setup
    criterion = get_criterion('crossentropy')
    optimizer = get_optimizer(model, optimizer_name='sgd', lr=0.01)
    scheduler = get_scheduler(optimizer, scheduler_name='cosine', num_epochs=num_epochs)
    
    # Train
    print(f"\nΕκπαίδευση για {num_epochs} epochs...")
    trainer = ClassificationTrainer(model, device=device)
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
    
    images, labels, predictions, probabilities = trainer.get_predictions(test_loader, num_samples=16)
    class_names = [f"Class {i}" for i in range(100)]
    visualize_classification_predictions(
        images, labels, predictions, probabilities,
        class_names=class_names,
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
