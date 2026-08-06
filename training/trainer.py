import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader

from .losses import create_loss
from .optimizer import create_optimizer
from .metrics import compute_metrics
from .checkpoint import save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualTrainer:
    """
    Robust native trainer isolating structural loops entirely avoiding external dependencies (Lightning/HuggingFace wrappers).
    """

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu", class_weights=None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if class_weights is not None:
             self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        else:
             self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = create_optimizer(self.model, learning_rate=1e-4, weight_decay=1e-3)
        self.best_metric = -1.0
        self.global_step = 0
        
    def _run_epoch(self, dataloader: DataLoader, is_train: bool, return_debug: bool = False) -> Dict[str, Any]:
        """Unified internal tracking iteration separating compute graphs appropriately."""
        if is_train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        all_preds = []
        all_targets = []
        # for participant-level accumulation (validation only)
        pid_logits = {}
        pid_targets = {}
        
        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, targets, pids = batch
                else:
                    inputs, targets = batch
                    pids = None
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                    
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                
                if is_train:
                    loss.backward()
                    
                    if self.global_step < 5:
                        total_norm = 0.0
                        logger.info(f"--- Iteration {self.global_step} Gradient Norms ---")
                        for name, p in self.model.named_parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                                logger.info(f"{name}: {param_norm.item():.4f}")
                        total_norm = total_norm ** 0.5
                        logger.info(f"Total Gradient Norm: {total_norm:.4f}")
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1
                    
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                
                if not is_train and pids is not None:
                    # Accumulate softmax probabilities for participant-level aggregation
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    targets_np = targets.detach().cpu().numpy()
                    pids_np = pids.detach().cpu().numpy()
                    
                    for i, pid in enumerate(pids_np):
                        pid = int(pid)
                        if pid not in pid_logits:
                            pid_logits[pid] = []
                            pid_targets[pid] = targets_np[i]
                        pid_logits[pid].append(probs[i])
                        
        # Normalize bounds safely natively without complex reducers
        avg_loss = total_loss / max(len(dataloader.dataset), 1)
        window_metrics = compute_metrics(all_targets, all_preds, avg_loss)
        
        if not is_train and pid_logits:
            import numpy as np
            part_targets = []
            part_preds = []
            for pid, probs_list in pid_logits.items():
                mean_probs = np.mean(probs_list, axis=0)
                pred_label = np.argmax(mean_probs)
                part_preds.append(pred_label)
                part_targets.append(pid_targets[pid])
                
            part_metrics = compute_metrics(part_targets, part_preds, avg_loss)
            part_metrics['window_accuracy'] = window_metrics['accuracy'] # Diagnostic
            
            if return_debug:
                part_metrics['pid_logits'] = pid_logits
                part_metrics['pid_targets'] = pid_targets
                
            return part_metrics
            
        return window_metrics

    def train(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Orchestrates backward pass mapping per mini-batch sequence."""
        return self._run_epoch(dataloader, is_train=True)
        
    def validate(self, dataloader: DataLoader, return_debug: bool = False) -> Dict[str, Any]:
        """Freezes tensors and asserts generalization constraints."""
        return self._run_epoch(dataloader, is_train=False, return_debug=return_debug)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, metric_for_best: str = 'f1'):
        """
        Runs consecutive epochs natively triggering Checkpointing triggers autonomously.
        """
        logger.info(f"Target Sequence Training Triggered ({epochs} Iterations) -> Mapped on {self.device}")
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train(train_loader)
            val_metrics = self.validate(val_loader)
            
            lr_current = self.optimizer.param_groups[0]['lr']
            
            # Explicit unified logging format
            logger.info(
                f"Epoch [{epoch}/{epochs}] LR: {lr_current:.2e}"
            )
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}"
            )
            logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}"
            )
            
            # Additional reporting on validation dataset distributions
            cm = val_metrics['cm']
            counts = val_metrics['counts']
            logger.info("Validation Confusion Matrix:")
            logger.info(f"Predicted:\n  Class 0 : {counts['pred_0']}\n  Class 1 : {counts['pred_1']}")
            logger.info(f"Ground Truth:\n  Class 0 : {counts['true_0']}\n  Class 1 : {counts['true_1']}")
            logger.info(f"Matrix:\n{cm}")
            
            # Threshold testing for serialization
            current_metric = train_metrics['accuracy'] if metric_for_best == 'accuracy' else val_metrics['f1']
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                
            save_checkpoint(self.model, epoch, self.best_metric, is_best)
            
            if is_best:
                logger.info(f"Epoch {epoch}: Structural weights persisted. Checkpoint Saved -> (Anchor: {self.best_metric:.4f})")

