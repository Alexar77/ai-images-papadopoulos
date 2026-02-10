"""
Unified Training Utilities
Contains all trainer classes and helper functions for all exercises
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import time
import numpy as np
import copy


# ============================================================================
# OPTIMIZERS & SCHEDULERS (Shared)
# ============================================================================


class EarlyStoppingController:
    """Shared early stopping controller for all trainers."""

    def __init__(self, patience=None, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = None
        self.best_state = None
        self.patience_counter = 0

    def _is_improvement(self, value):
        if self.best_value is None:
            return True
        if self.mode == 'max':
            return value > self.best_value + self.min_delta
        if self.mode == 'min':
            return value < self.best_value - self.min_delta
        raise ValueError(f"Unknown early-stopping mode: {self.mode}")

    def update(self, value, model):
        improved = self._is_improvement(value)
        if improved:
            self.best_value = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.patience_counter = 0
            return improved, False

        self.patience_counter += 1
        should_stop = bool(self.patience and self.patience_counter >= self.patience)
        return improved, should_stop

def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0):
    """Factory function for optimizers"""
    
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='step', num_epochs=10):
    """Factory function for learning rate schedulers"""
    
    if scheduler_name == 'step':
        return StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_criterion(criterion_name='crossentropy'):
    """Factory function for loss functions"""
    
    if criterion_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'mse':
        return nn.MSELoss()
    elif criterion_name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


# ============================================================================
# CLASSIFICATION TRAINER (Ex1, Ex2, Ex5)
# ============================================================================

class ClassificationTrainer:
    """Trainer for image classification tasks"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': None,
            'epoch_times': [],
            'inference_time': None
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, data_loader, criterion):
        """Evaluate on validation/test set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        scheduler=None,
        early_stopping_patience=None
    ):
        """Full training loop"""
        
        print(f"\nΈναρξη εκπαίδευσης για {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 70)

        use_validation = val_loader is not None
        early_stop = EarlyStoppingController(
            patience=early_stopping_patience,
            mode='max' if use_validation else 'min'
        )
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            if use_validation:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
            else:
                val_loss, val_acc = None, None
            
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss if use_validation else train_loss)
                else:
                    scheduler.step()
            
            monitor_value = val_acc if use_validation else train_loss
            _, should_stop = early_stop.update(monitor_value, self.model)
            
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if use_validation:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            if use_validation:
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            if scheduler is not None and hasattr(scheduler, 'get_last_lr'):
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 70)
            
            if should_stop:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
                break

        if early_stop.best_state is not None:
            self.model.load_state_dict(early_stop.best_state)
        
        print(f"\nΕκπαίδευση ολοκληρώθηκε!")
        if use_validation:
            best_val_acc = early_stop.best_value if early_stop.best_value is not None else 0.0
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        else:
            best_train_loss = early_stop.best_value if early_stop.best_value is not None else float('inf')
            print(f"Best Training Loss (early-stop monitor): {best_train_loss:.4f}")
        print("=" * 70)
        
        return self.history
    
    @torch.no_grad()
    def test(self, test_loader, criterion):
        """Final evaluation on test set"""
        print("\nΑξιολόγηση στο Test Set...")
        
        start_time = time.time()
        test_loss, test_acc = self.evaluate(test_loader, criterion)
        inference_time = time.time() - start_time
        
        self.history['test_acc'] = test_acc
        self.history['inference_time'] = inference_time
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Inference Time: {inference_time:.2f}s")
        print("=" * 70)
        
        return test_loss, test_acc
    
    @torch.no_grad()
    def get_predictions(self, data_loader, num_samples=10):
        """Get predictions for visualization"""
        self.model.eval()
        
        all_images = []
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        for images, labels in data_loader:
            images_device = images.to(self.device)
            
            outputs = self.model(images_device)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_images.extend(images.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if len(all_images) >= num_samples:
                break
        
        return (
            all_images[:num_samples],
            all_labels[:num_samples],
            all_predictions[:num_samples],
            all_probabilities[:num_samples]
        )


# ============================================================================
# SEGMENTATION TRAINER (Ex3)
# ============================================================================

class SegmentationTrainer:
    """Trainer for semantic segmentation tasks"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_pixel_acc': [],
            'train_mean_iou': [],
            'val_loss': [],
            'val_pixel_acc': [],
            'val_mean_iou': [],
            'test_pixel_acc': None,
            'test_mean_iou': None,
            'epoch_times': []
        }
    
    def calculate_metrics(
        self,
        preds,
        masks,
        num_classes=21,
        ignore_index=255,
        include_background=False
    ):
        """Calculate pixel accuracy and mean IoU with proper ignore handling."""
        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()

        valid_mask = (masks != ignore_index)
        valid_pixels = valid_mask.sum()
        if valid_pixels == 0:
            return 0.0, 0.0

        # Pixel accuracy on valid (non-void) pixels only
        pixel_acc = (preds[valid_mask] == masks[valid_mask]).sum() / valid_pixels

        # Mean IoU over foreground classes by default (exclude class 0: background)
        ious = []
        class_start = 0 if include_background else 1
        for cls in range(class_start, num_classes):
            pred_cls = (preds == cls) & valid_mask
            mask_cls = (masks == cls) & valid_mask

            intersection = (pred_cls & mask_cls).sum()
            union = (pred_cls | mask_cls).sum()

            if union > 0:
                ious.append(intersection / union)

        mean_iou = float(np.mean(ious)) if ious else 0.0

        return float(pixel_acc), mean_iou
    
    def train_epoch(self, train_loader, criterion, optimizer, num_classes=21):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_pixel_acc = 0
        total_iou = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            pixel_acc, mean_iou = self.calculate_metrics(preds, masks, num_classes)
            total_pixel_acc += pixel_acc
            total_iou += mean_iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'IoU': f'{mean_iou:.4f}'
            })
        
        num_batches = len(train_loader)
        return total_loss / num_batches, total_pixel_acc / num_batches, total_iou / num_batches
    
    @torch.no_grad()
    def evaluate(self, data_loader, criterion, num_classes=21):
        """Evaluate on validation/test set"""
        self.model.eval()
        
        total_loss = 0
        total_pixel_acc = 0
        total_iou = 0
        
        for images, masks in data_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            pixel_acc, mean_iou = self.calculate_metrics(preds, masks, num_classes)
            total_pixel_acc += pixel_acc
            total_iou += mean_iou
        
        num_batches = len(data_loader)
        return total_loss / num_batches, total_pixel_acc / num_batches, total_iou / num_batches
    
    def train(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=10,
        scheduler=None,
        num_classes=21,
        early_stopping_patience=None
    ):
        """Full training loop"""
        
        print(f"\nΈναρξη εκπαίδευσης για {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 70)

        early_stop = EarlyStoppingController(
            patience=early_stopping_patience,
            mode='max'
        )
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss, train_pixel_acc, train_iou = self.train_epoch(
                train_loader, criterion, optimizer, num_classes
            )
            val_loss, val_pixel_acc, val_iou = self.evaluate(
                val_loader, criterion, num_classes
            )
            
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            _, should_stop = early_stop.update(val_iou, self.model)
            
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(train_loss)
            self.history['train_pixel_acc'].append(train_pixel_acc)
            self.history['train_mean_iou'].append(train_iou)
            self.history['val_loss'].append(val_loss)
            self.history['val_pixel_acc'].append(val_pixel_acc)
            self.history['val_mean_iou'].append(val_iou)
            self.history['epoch_times'].append(epoch_time)
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Pixel Acc: {train_pixel_acc:.4f} | mIoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Pixel Acc: {val_pixel_acc:.4f} | mIoU: {val_iou:.4f}")
            if scheduler and hasattr(scheduler, 'get_last_lr'):
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 70)

            if should_stop:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
                break

        if early_stop.best_state is not None:
            self.model.load_state_dict(early_stop.best_state)
        
        print(f"\nΕκπαίδευση ολοκληρώθηκε!")
        best_iou = early_stop.best_value if early_stop.best_value is not None else 0.0
        print(f"Best Validation mIoU: {best_iou:.4f}")
        print("=" * 70)
        
        return self.history
    
    @torch.no_grad()
    def test(self, test_loader, criterion, num_classes=21):
        """Final evaluation on test set"""
        print("\nΑξιολόγηση στο Test Set...")
        
        test_loss, test_pixel_acc, test_iou = self.evaluate(test_loader, criterion, num_classes)
        
        self.history['test_pixel_acc'] = test_pixel_acc
        self.history['test_mean_iou'] = test_iou
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Pixel Accuracy: {test_pixel_acc:.4f}")
        print(f"Test Mean IoU: {test_iou:.4f}")
        print("=" * 70)
        
        return test_loss, test_pixel_acc, test_iou
    
    @torch.no_grad()
    def get_predictions(self, data_loader, num_samples=4):
        """Get predictions for visualization"""
        self.model.eval()
        
        all_images = []
        all_masks = []
        all_predictions = []
        
        for images, masks in data_loader:
            images_device = images.to(self.device)
            
            outputs = self.model(images_device)
            preds = outputs.argmax(dim=1)
            
            all_images.extend(images.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            
            if len(all_images) >= num_samples:
                break
        
        return all_images[:num_samples], all_masks[:num_samples], all_predictions[:num_samples]


# ============================================================================
# DETECTION TRAINER (Ex4)
# ============================================================================

class DetectionTrainer:
    """Trainer for object detection tasks"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_loss_classifier': [],
            'train_loss_box_reg': [],
            'train_loss_objectness': [],
            'train_loss_rpn_box_reg': [],
            'epoch_times': []
        }
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0)).item()
            loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
            loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0)).item()
            loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()
            
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        num_batches = len(train_loader)
        return {
            'total_loss': total_loss / num_batches,
            'loss_classifier': loss_classifier / num_batches,
            'loss_box_reg': loss_box_reg / num_batches,
            'loss_objectness': loss_objectness / num_batches,
            'loss_rpn_box_reg': loss_rpn_box_reg / num_batches
        }
    
    def train(self, train_loader, optimizer, num_epochs=10, scheduler=None, early_stopping_patience=None):
        """Full training loop"""
        
        print(f"\nΈναρξη εκπαίδευσης για {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 70)

        early_stop = EarlyStoppingController(
            patience=early_stopping_patience,
            mode='min'
        )
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            losses = self.train_epoch(train_loader, optimizer)
            
            if scheduler is not None:
                scheduler.step()
            
            _, should_stop = early_stop.update(losses['total_loss'], self.model)
            
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(losses['total_loss'])
            self.history['train_loss_classifier'].append(losses['loss_classifier'])
            self.history['train_loss_box_reg'].append(losses['loss_box_reg'])
            self.history['train_loss_objectness'].append(losses['loss_objectness'])
            self.history['train_loss_rpn_box_reg'].append(losses['loss_rpn_box_reg'])
            self.history['epoch_times'].append(epoch_time)
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
            print(f"Total Loss: {losses['total_loss']:.4f}")
            print(f"  Classifier: {losses['loss_classifier']:.4f} | Box Reg: {losses['loss_box_reg']:.4f}")
            print(f"  Objectness: {losses['loss_objectness']:.4f} | RPN Box: {losses['loss_rpn_box_reg']:.4f}")
            if scheduler and hasattr(scheduler, 'get_last_lr'):
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print("-" * 70)
            
            if should_stop:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
                break

        if early_stop.best_state is not None:
            self.model.load_state_dict(early_stop.best_state)
        
        print(f"\nΕκπαίδευση ολοκληρώθηκε!")
        best_loss = early_stop.best_value if early_stop.best_value is not None else float('inf')
        print(f"Best Training Loss: {best_loss:.4f}")
        print("=" * 70)
        
        return self.history
    
    @torch.no_grad()
    def get_predictions(self, data_loader, num_samples=4, score_threshold=0.5):
        """Get predictions for visualization"""
        self.model.eval()
        
        all_images = []
        all_predictions = []
        
        for images, targets in data_loader:
            images_device = [img.to(self.device) for img in images]
            predictions = self.model(images_device)
            
            for img, pred in zip(images, predictions):
                keep = pred['scores'] > score_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][keep].cpu().numpy(),
                    'labels': pred['labels'][keep].cpu().numpy(),
                    'scores': pred['scores'][keep].cpu().numpy()
                }
                
                all_images.append(img.cpu().numpy())
                all_predictions.append(filtered_pred)
                
                if len(all_images) >= num_samples:
                    break
            
            if len(all_images) >= num_samples:
                break
        
        return all_images[:num_samples], all_predictions[:num_samples]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6
    }


# Alias for backward compatibility
Trainer = ClassificationTrainer
ComparativeTrainer = ClassificationTrainer
