"""
Trainer for H-DCD model

集成 AtCAF 创新点的训练逻辑：
1. 两阶段训练：阶段0(MMILB预训练) → 阶段1(主模型训练)
2. Memory机制：维护正/负样本缓存用于MMILB熵估计
3. 反事实效应损失计算
"""
import os
import time
import logging
import numpy as np
from collections import deque
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
        # 兼容 learning_rate / lr 两种参数名
        _lr = args.get('learning_rate', args.get('lr', 1e-4))
        _wd = args.get('weight_decay', 0.0)
        _num_epochs = args.get('num_epochs', args.get('n_epochs', 100))
        self._num_epochs = _num_epochs
        
        optimizer_type = args.get('optimizer', 'adamw')
        if optimizer_type == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=_lr,
                weight_decay=_wd
            )
        else:  # adamw
            self.optimizer = AdamW(
                model.parameters(),
                lr=_lr,
                weight_decay=_wd
            )
        
        # Learning rate scheduler
        scheduler_type = args.get('scheduler', 'reduce')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=_num_epochs,
                eta_min=_lr * 0.01
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
        _warmup_epochs = args.get('warmup_epochs', 0)
        if _warmup_epochs > 0:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=_warmup_epochs,
                base_lr=_lr * 0.1,
                target_lr=_lr
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
        
        # ================================================================
        # === AtCAF 创新点: Memory 机制和两阶段训练 ===
        # ================================================================
        # MMILB Memory: 缓存正/负样本用于熵估计
        self.mi_memory_size = args.get('mi_memory_size', 10)
        self.mi_warmup_epochs = args.get('mi_warmup_epochs', 5)
        self.use_mutual_info = args.get('use_mutual_info', True)
        self._init_memory()
    
    def _init_memory(self):
        """
        初始化MMILB的Memory缓存
        
        Memory结构: {'tv': {'pos': deque, 'neg': deque}, 
                     'ta': {'pos': deque, 'neg': deque},
                     'va': {'pos': deque, 'neg': deque}}
        使用deque实现固定大小的FIFO缓存
        """
        self.mem = {
            'tv': {'pos': deque(maxlen=self.mi_memory_size), 
                   'neg': deque(maxlen=self.mi_memory_size)},
            'ta': {'pos': deque(maxlen=self.mi_memory_size), 
                   'neg': deque(maxlen=self.mi_memory_size)},
            'va': {'pos': deque(maxlen=self.mi_memory_size), 
                   'neg': deque(maxlen=self.mi_memory_size)},
        }
    
    def _get_mem_for_model(self):
        """
        将deque格式的memory转换为model可用的list格式
        
        Returns:
            mem_dict: 包含list格式的memory字典
        """
        mem_dict = {}
        for key in ['tv', 'ta', 'va']:
            pos_list = list(self.mem[key]['pos'])
            neg_list = list(self.mem[key]['neg'])
            if len(pos_list) > 0 and len(neg_list) > 0:
                mem_dict[key] = {'pos': pos_list, 'neg': neg_list}
            else:
                mem_dict[key] = None
        return mem_dict
    
    def _update_memory(self, pn_dic):
        """
        更新MMILB的Memory缓存
        
        Args:
            pn_dic: 模型输出的正负样本字典 
                    {'tv': {'pos': tensor, 'neg': tensor}, ...}
        """
        if pn_dic is None:
            return
        for key in ['tv', 'ta', 'va']:
            if key in pn_dic and pn_dic[key] is not None:
                pos = pn_dic[key].get('pos', None)
                neg = pn_dic[key].get('neg', None)
                if pos is not None and pos.numel() > 0:
                    self.mem[key]['pos'].append(pos.detach())
                if neg is not None and neg.numel() > 0:
                    self.mem[key]['neg'].append(neg.detach())
    
    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch
        
        两阶段训练逻辑 (来自AtCAF):
        - 阶段0 (epoch <= mi_warmup_epochs): 仅训练MMILB，最大化互信息下界lld
        - 阶段1 (epoch > mi_warmup_epochs): 训练完整模型，包含所有损失
        """
        self.model.train()
        total_loss = 0.0
        loss_items = {}
        
        # 判断当前是否为MMILB预训练阶段
        is_mi_warmup = (self.use_mutual_info and 
                        epoch <= self.mi_warmup_epochs)
        
        if is_mi_warmup:
            logger.info(f"[阶段0] MMILB预训练 (epoch {epoch}/{self.mi_warmup_epochs})")
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 获取memory用于MMILB熵估计
            mem = self._get_mem_for_model() if self.use_mutual_info else None
            
            # Forward (传入labels和mem用于互信息计算)
            outputs = self.model(
                text, audio, video, 
                return_all=True,
                labels=labels,
                mem=mem,
            )
            
            if is_mi_warmup:
                # ============================================================
                # 阶段0: 仅优化MMILB，最大化lld (互信息下界)
                # 损失 = -lld (最大化lld等价于最小化-lld)
                # ============================================================
                mi_out = outputs.get('mi_outputs', {})
                if mi_out and 'lld' in mi_out and isinstance(mi_out['lld'], torch.Tensor):
                    loss = -mi_out['lld']  # 最大化lld
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_dict = {'L_lld_warmup': loss.item()}
            else:
                # ============================================================
                # 阶段1: 完整训练，计算所有损失
                # ============================================================
                loss, loss_dict = self.criterion(outputs, labels)
            
            # 更新MMILB Memory
            mi_out = outputs.get('mi_outputs', {})
            if mi_out and 'pn_dic' in mi_out:
                self._update_memory(mi_out['pn_dic'])
            
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
        logger.info(f"Total epochs: {self._num_epochs}")
        logger.info(f"Patience: {self.patience}")
        logger.info("=" * 80)
        
        _warmup_epochs = self.args.get('warmup_epochs', 0)
        _model_save_dir = self.args.get('model_save_dir', './checkpoints')
        
        for epoch in range(1, self._num_epochs + 1):
            epoch_start_time = time.time()
            
            # Warmup
            if self.warmup_scheduler and epoch <= _warmup_epochs:
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
                self.save_model(os.path.join(_model_save_dir, 'best_model.pth'))
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
        _model_save_dir = self.args.get('model_save_dir', './checkpoints')
        best_model_path = os.path.join(_model_save_dir, 'best_model.pth')
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
