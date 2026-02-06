"""
Script για ανάλυση και σύγκριση αποθηκευμένων αποτελεσμάτων πειραμάτων
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment_results(results_dir):
    """
    Φόρτωση αποτελεσμάτων από JSON αρχείο
    """
    json_path = os.path.join(results_dir, 'experiment_results.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Δε βρέθηκε το αρχείο: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def print_summary_table(results):
    """
    Εμφάνιση πίνακα με σύνοψη αποτελεσμάτων
    """
    print("\n" + "=" * 100)
    print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 100)
    
    # Header
    print(f"{'Experiment':<30} {'Val Acc':<12} {'Test Acc':<12} {'Time (s)':<12} {'Optimizer':<12}")
    print("-" * 100)
    
    # Sort by validation accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('val_acc', 0), reverse=True)
    
    for exp_name, exp_data in sorted_results:
        val_acc = exp_data.get('val_acc', 0)
        test_acc = exp_data.get('test_acc', 0)
        time = exp_data.get('total_time', 0)
        optimizer = exp_data.get('hyperparameters', {}).get('optimizer', 'N/A')
        
        print(f"{exp_name:<30} {val_acc:>10.2f}% {test_acc:>10.2f}% {time:>10.1f}s {optimizer:<12}")
    
    print("=" * 100)
    
    # Best results
    best_exp = max(results.items(), key=lambda x: x[1].get('val_acc', 0))
    print(f"\n🏆 Best Experiment: {best_exp[0]}")
    print(f"   Validation Accuracy: {best_exp[1]['val_acc']:.2f}%")
    print(f"   Test Accuracy: {best_exp[1]['test_acc']:.2f}%")
    print("=" * 100)


def compare_by_category(results):
    """
    Σύγκριση αποτελεσμάτων ανά κατηγορία (Loss, Optimizer, LR)
    """
    categories = {
        'Loss Functions': [k for k in results.keys() if k.startswith('Loss_')],
        'Optimizers': [k for k in results.keys() if k.startswith('Optimizer_')],
        'Learning Rates': [k for k in results.keys() if k.startswith('LR_') or k.startswith('LearningRate_')]
    }
    
    print("\n" + "=" * 100)
    print("ΣΥΓΚΡΙΣΗ ΑΝΑ ΚΑΤΗΓΟΡΙΑ")
    print("=" * 100)
    
    for category, experiments in categories.items():
        if not experiments:
            continue
        
        print(f"\n{category}:")
        print("-" * 100)
        
        for exp_name in sorted(experiments, key=lambda x: results[x].get('val_acc', 0), reverse=True):
            exp_data = results[exp_name]
            val_acc = exp_data.get('val_acc', 0)
            test_acc = exp_data.get('test_acc', 0)
            
            # Clean name
            clean_name = exp_name.replace('Loss_', '').replace('Optimizer_', '').replace('LearningRate_', '').replace('LR_', '')
            
            print(f"  {clean_name:<25} Val: {val_acc:>6.2f}%  Test: {test_acc:>6.2f}%")


def plot_category_comparison(results, save_path='category_comparison.png'):
    """
    Γράφημα σύγκρισης ανά κατηγορία
    """
    categories = {
        'Optimizers': [k for k in results.keys() if k.startswith('Optimizer_')],
        'Learning Rates': [k for k in results.keys() if k.startswith('LR_') or k.startswith('LearningRate_')]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (category, experiments) in enumerate(categories.items()):
        if not experiments or idx >= 2:
            continue
        
        # Clean names and get accuracies
        names = [exp.replace('Optimizer_', '').replace('LearningRate_', '').replace('LR_', '') 
                 for exp in experiments]
        val_accs = [results[exp].get('val_acc', 0) for exp in experiments]
        test_accs = [results[exp].get('test_acc', 0) for exp in experiments]
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[idx].bar(x - width/2, val_accs, width, label='Validation', alpha=0.8)
        axes[idx].bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        
        axes[idx].set_xlabel('Configuration', fontsize=12)
        axes[idx].set_ylabel('Accuracy (%)', fontsize=12)
        axes[idx].set_title(f'{category} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(names, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()


def find_best_configurations(results, top_k=3):
    """
    Εύρεση των καλύτερων ρυθμίσεων
    """
    print("\n" + "=" * 100)
    print(f"TOP {top_k} CONFIGURATIONS")
    print("=" * 100)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('val_acc', 0), reverse=True)
    
    for rank, (exp_name, exp_data) in enumerate(sorted_results[:top_k], 1):
        print(f"\n#{rank}. {exp_name}")
        print("-" * 100)
        print(f"  Validation Accuracy: {exp_data.get('val_acc', 0):.2f}%")
        print(f"  Test Accuracy: {exp_data.get('test_acc', 0):.2f}%")
        print(f"  Training Time: {exp_data.get('total_time', 0):.2f}s")
        
        if 'hyperparameters' in exp_data:
            print("  Hyperparameters:")
            for key, value in exp_data['hyperparameters'].items():
                print(f"    - {key}: {value}")


def main():
    """
    Κύρια συνάρτηση ανάλυσης
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('results_dir', type=str, help='Directory containing experiment results')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    args = parser.parse_args()
    
    print("=" * 100)
    print("ΑΝΑΛΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΠΕΙΡΑΜΑΤΩΝ")
    print("=" * 100)
    print(f"Loading results from: {args.results_dir}")
    
    # Load results
    try:
        results = load_experiment_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nΧρήση: python analyze_results.py <results_directory>")
        print("Παράδειγμα: python analyze_results.py results/experiments_20260204_120000")
        return
    
    print(f"✓ Loaded {len(results)} experiments")
    
    # Print summary table
    print_summary_table(results)
    
    # Compare by category
    compare_by_category(results)
    
    # Find best configurations
    find_best_configurations(results, top_k=3)
    
    # Generate plots if requested
    if args.plot:
        print("\n" + "=" * 100)
        print("ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ")
        print("=" * 100)
        plot_category_comparison(results, 
                                save_path=os.path.join(args.results_dir, 'analysis_comparison.png'))
    
    print("\n" + "=" * 100)
    print("ΑΝΑΛΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ")
    print("=" * 100)


if __name__ == "__main__":
    main()
