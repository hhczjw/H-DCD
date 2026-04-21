"""
Trainer for H-DCD model
"""
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from losses import H_DCD_Loss, WarmupScheduler

logger = logging.getLogger('H-DCD')


class H_DCD_Trainer:
    """
    Trainer for H-DCD model
    """
    def __init__(self, args, model, device):
        self.args = args
        self.model = model.to(device)
        self.device = device
        self.task_type = args.get('task_type', 'classification')
        
        # Loss function
        self.criterion = H_DCD_Loss(args)
        
        # Optimizer
        optimizer_type = args.get('optimizer', 'adamw')
        if optimizer_type == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:  # adamw
            self.optimizer = AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        
        # Learning rate scheduler
        scheduler_type = args.get('scheduler', 'reduce')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=args.num_epochs,
                eta_min=args.learning_rate * 0.01
            )
        else:  # reduce on plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.task_type == 'regression' else 'max',
                factor=0.5,
                patience=args.get('scheduler_patience', 5),
                verbose=True
            )
        
        # Warmup (optional)
        if args.get('warmup_epochs', 0) > 0:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=args.warmup_epochs,
                base_lr=args.learning_rate * 0.1,
                target_lr=args.learning_rate
            )
        else:
            self.warmup_scheduler = None
        
        # Best model tracking
        self.best_valid_loss = float('inf')
        self.best_valid_metric = -float('inf') if self.task_type == 'classification' else float('inf')
        self.patience_counter = 0
        self.patience = args.get('patience', 10)
        
        # Gradient clipping
        self.grad_clip = args.get('grad_clip', 1.0)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_items = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            outputs = self.model(text, audio, video, return_all=True)
            
            # Compute loss
            loss, loss_dict = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_items:
                    loss_items[k] = 0.0
                loss_items[k] += v
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_loss_items = {k: v / num_batches for k, v in loss_items.items()}
        
        return avg_loss, avg_loss_items
    
    def evaluate(self, dataloader, mode='Valid'):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'{mode}')
            for batch in pbar:
                # Move to device
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                outputs = self.model(text, audio, video, return_all=True)
                
                # Compute loss
                loss, _ = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                if self.task_type == 'classification':
                    preds = torch.argmax(outputs['logits_multi'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:  # regression
                    preds = outputs['logits_multi']
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(all_labels, all_preds)
            f1_weighted = f1_score(all_labels, all_preds, average='weighted')
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            
            results = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            }
            
            logger.info(f"{mode} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                       f"F1-W: {f1_weighted:.4f}, F1-M: {f1_macro:.4f}")
        else:  # regression
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            mae = np.mean(np.abs(all_preds - all_labels))
            corr = np.corrcoef(all_preds, all_labels)[0, 1]
            
            # Binary classification metrics (for sentiment: positive if > 0)
            binary_preds = (all_preds > 0).astype(int)
            binary_labels = (all_labels > 0).astype(int)
            binary_acc = accuracy_score(binary_labels, binary_preds)
            binary_f1 = f1_score(binary_labels, binary_preds)
            
            results = {
                'loss': avg_loss,
                'mae': mae,
                'corr': corr,
                'binary_acc': binary_acc,
                'binary_f1': binary_f1
            }
            
            logger.info(f"{mode} - Loss: {avg_loss:.4f}, MAE: {mae:.4f}, "
                       f"Corr: {corr:.4f}, Binary Acc: {binary_acc:.4f}, F1: {binary_f1:.4f}")
        
        return results
    
    def train(self, train_loader, valid_loader):
        """Full training loop"""
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.args.num_epochs}")
        logger.info(f"Patience: {self.patience}")
        logger.info("=" * 80)
        
        for epoch in range(1, self.args.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Warmup
            if self.warmup_scheduler and epoch <= self.args.get('warmup_epochs', 0):
                self.warmup_scheduler.step()
                logger.info(f"Warmup LR: {self.warmup_scheduler.get_lr():.6f}")
            
            # Train
            train_loss, train_loss_items = self.train_epoch(train_loader, epoch)
            
            # Validate
            valid_results = self.evaluate(valid_loader, mode='Valid')
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if self.task_type == 'classification':
                    self.scheduler.step(valid_results['accuracy'])
                else:
                    self.scheduler.step(valid_results['mae'])
            else:
                self.scheduler.step()
            
            # Check improvement
            if self.task_type == 'classification':
                current_metric = valid_results['accuracy']
                improved = current_metric > self.best_valid_metric
            else:
                current_metric = valid_results['mae']
                improved = current_metric < self.best_valid_metric
            
            if improved:
                self.best_valid_metric = current_metric
                self.best_valid_loss = valid_results['loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_model(os.path.join(self.args.model_save_dir, 'best_model.pth'))
                logger.info(f"✓ Best model saved! Metric: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info("-" * 80)
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best validation metric: {self.best_valid_metric:.4f}")
        logger.info("=" * 80)
    
    def test(self, test_loader):
        """Test the model"""
        # Load best model
        best_model_path = os.path.join(self.args.model_save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            logger.warning("Best model not found, using current model")
        
        # Evaluate
        test_results = self.evaluate(test_loader, mode='Test')
        
        return test_results
    
    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")
