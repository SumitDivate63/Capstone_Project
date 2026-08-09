"""
Audio Agent Trainer for DAIC-WOZ Depression Detection.

Mirrors the VisualTrainer pattern extended with:
- Participant-level aggregation via softmax probability averaging (window→participant)
- ReduceLROnPlateau scheduler
- Early stopping
- GPU memory reporting
- Extended metrics (macro F1, per-class F1, full confusion matrix)
- Dedicated log file (no conflict with text trainer or visual trainer)
- Fold-specific checkpoint directory (outputs/checkpoints/audio/foldN/)

Training loop is window-level (multiple windows per participant).
Evaluation aggregates windows → participant predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from training.checkpoint import save_checkpoint
from training.optimizer import create_optimizer
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_extended_metrics(
    y_true: List[int],
    y_pred: List[int],
    loss: float
) -> Dict[str, Any]:
    """
    Compute full metric suite including macro F1 and per-class F1.
    Primary selection metric: macro F1 (not binary F1, not accuracy).
    """
    if len(y_true) == 0:
        return {
            "loss": float(loss), "accuracy": 0.0,
            "macro_f1": 0.0, "binary_f1": 0.0,
            "precision": 0.0, "recall": 0.0,
            "per_class_f1": [0.0, 0.0],
            "cm": [[0, 0], [0, 0]],
            "counts": {"true_0": 0, "true_1": 0, "pred_0": 0, "pred_1": 0}
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1]).tolist()

    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1,
        "cm": cm,
        "counts": {
            "true_0": y_true.count(0),
            "true_1": y_true.count(1),
            "pred_0": y_pred.count(0),
            "pred_1": y_pred.count(1),
        }
    }


class AudioTrainer:
    """
    Trainer for the AudioModel. Window-level training, participant-level validation.

    Key differences from VisualTrainer:
    - Uses ReduceLROnPlateau scheduler
    - Has early stopping
    - Uses macro F1 as primary selection metric
    - Reports GPU memory usage each epoch
    - Writes to isolated log/checkpoint directories
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = create_optimizer(self.model, learning_rate=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

        self.best_macro_f1 = -1.0
        self.best_accuracy = -1.0
        self.best_epoch    = -1
        self.global_step   = 0

        logger.info(f"AudioTrainer ready on {self.device}")

    def _gpu_memory_str(self) -> str:
        if self.device.type == "cuda":
            alloc  = torch.cuda.memory_allocated(self.device) / 1024**2
            reserv = torch.cuda.memory_reserved(self.device) / 1024**2
            return f"GPU: {alloc:.0f}MB alloc / {reserv:.0f}MB reserved"
        return "GPU: N/A (CPU mode)"

    def _run_epoch(
        self,
        dataloader: DataLoader,
        is_train: bool,
        return_debug: bool = False
    ) -> Dict[str, Any]:
        """Window-level forward/backward. Participant-level aggregation during validation."""
        self.model.train() if is_train else self.model.eval()

        total_loss     = 0.0
        all_preds      = []
        all_targets    = []
        pid_probs      = {}   # pid → list of softmax prob arrays
        pid_targets    = {}   # pid → true label

        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, targets, pids = batch
                else:
                    inputs, targets = batch
                    pids = None

                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                logits = self.model(inputs)
                loss   = self.criterion(logits, targets)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

                # Participant-level accumulation for validation
                if not is_train and pids is not None:
                    probs      = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    targets_np = targets.detach().cpu().numpy()
                    pids_np    = pids.detach().cpu().numpy()
                    for i, pid in enumerate(pids_np):
                        pid = int(pid)
                        if pid not in pid_probs:
                            pid_probs[pid]   = []
                            pid_targets[pid] = int(targets_np[i])
                        pid_probs[pid].append(probs[i])

        avg_loss     = total_loss / max(len(dataloader.dataset), 1)
        window_metrics = compute_extended_metrics(all_targets, all_preds, avg_loss)

        if not is_train and pid_probs:
            part_targets = []
            part_preds   = []
            part_probs_c0 = []
            part_probs_c1 = []

            for pid, probs_list in pid_probs.items():
                mean_probs = np.mean(probs_list, axis=0)
                pred_label = int(np.argmax(mean_probs))
                part_preds.append(pred_label)
                part_targets.append(pid_targets[pid])
                part_probs_c0.append(float(mean_probs[0]))
                part_probs_c1.append(float(mean_probs[1]))

            part_metrics = compute_extended_metrics(part_targets, part_preds, avg_loss)
            part_metrics["window_accuracy"] = window_metrics["accuracy"]  # diagnostic

            if return_debug:
                part_metrics["pid_probs"]       = pid_probs
                part_metrics["pid_targets"]     = pid_targets
                part_metrics["pid_list"]        = list(pid_probs.keys())
                part_metrics["part_targets"]    = part_targets
                part_metrics["part_preds"]      = part_preds
                part_metrics["part_probs_c0"]   = part_probs_c0
                part_metrics["part_probs_c1"]   = part_probs_c1

            return part_metrics

        return window_metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        return self._run_epoch(dataloader, is_train=True)

    def validate(self, dataloader: DataLoader, return_debug: bool = False) -> Dict[str, Any]:
        return self._run_epoch(dataloader, is_train=False, return_debug=return_debug)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        fold: int = 1,
        seed: int = 42,
        save_dir: str = "outputs/checkpoints/audio/fold1",
        log_fn=None,
        early_stopping_patience: int = 5,
        start_epoch: int = 1,
    ) -> None:
        """
        Full training loop with:
        - Epoch-level logging to dedicated log file
        - ReduceLROnPlateau scheduler stepping on val macro F1
        - Early stopping based on macro F1
        - Checkpoint saving (best + last)
        - GPU memory reporting
        """
        logger.info(f"AudioTrainer.fit: {epochs} epochs on {self.device}, fold={fold}")
        no_improve_count = 0

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate(val_loader)

            lr_current   = self.optimizer.param_groups[0]["lr"]
            val_macro_f1 = val_metrics["macro_f1"]
            gpu_str      = self._gpu_memory_str()

            # Step scheduler
            self.scheduler.step(val_macro_f1)

            is_best = val_macro_f1 > self.best_macro_f1
            if is_best:
                self.best_macro_f1 = val_macro_f1
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch    = epoch
                no_improve_count   = 0
            else:
                no_improve_count += 1

            # Build log line
            log_line = (
                f"[Audio Fold {fold}] Epoch [{epoch}/{epochs}] LR: {lr_current:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
                f"MacroF1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"MacroF1: {val_metrics['macro_f1']:.4f} "
                f"Prec: {val_metrics['precision']:.4f} Rec: {val_metrics['recall']:.4f} | "
                f"CM: {np.array(val_metrics['cm']).tolist()} | {gpu_str}"
            )
            logger.info(log_line)
            if log_fn:
                log_fn(log_line)

            if is_best:
                logger.info(
                    f"  ✓ New best MacroF1={self.best_macro_f1:.4f} at epoch {epoch}. Checkpoint saved."
                )

            scheduler_state = self.scheduler.state_dict()
            save_checkpoint(
                model=self.model,
                epoch=epoch,
                best_f1=self.best_macro_f1,
                is_best=is_best,
                optimizer=self.optimizer,
                scheduler_state=scheduler_state,
                best_accuracy=self.best_accuracy,
                fold=fold,
                seed=seed,
                save_dir=save_dir,
            )

            # Early stopping
            if no_improve_count >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {no_improve_count} epochs without improvement. "
                    f"Best MacroF1={self.best_macro_f1:.4f} at epoch {self.best_epoch}."
                )
                break

        logger.info(
            f"Training complete. Best MacroF1={self.best_macro_f1:.4f} "
            f"at epoch {self.best_epoch}."
        )
