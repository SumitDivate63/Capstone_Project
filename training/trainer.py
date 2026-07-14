import torch
import torch.nn as nn
from typing import Dict
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

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = create_loss().to(self.device)
        self.optimizer = create_optimizer(self.model)
        self.best_f1 = -1.0
        
    def _run_epoch(self, dataloader: DataLoader, is_train: bool) -> Dict[str, float]:
        """Unified internal tracking iteration separating compute graphs appropriately."""
        if is_train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.set_grad_enabled(is_train):
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                    
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                
        # Normalize bounds safely natively without complex reducers
        avg_loss = total_loss / max(len(dataloader.dataset), 1)
        return compute_metrics(all_targets, all_preds, avg_loss)

    def train(self, dataloader: DataLoader) -> Dict[str, float]:
        """Orchestrates backward pass mapping per mini-batch sequence."""
        return self._run_epoch(dataloader, is_train=True)
        
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Freezes tensors and asserts generalization constraints."""
        return self._run_epoch(dataloader, is_train=False)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
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
                f"Epoch [{epoch}/{epochs}] "
                f"LR: {lr_current:.2e} | "
                f"Tr. Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
                f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}"
            )
            
            # Threshold testing for serialization
            is_best = val_metrics['f1'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['f1']
                
            save_checkpoint(self.model, epoch, self.best_f1, is_best)
            
            if is_best:
                logger.info(f"Epoch {epoch}: Structural weights persisted. Checkpoint Saved -> (F1 Metric Anchor: {self.best_f1:.4f})")
