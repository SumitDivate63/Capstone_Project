"""
Text Agent Trainer for DAIC-WOZ Depression Detection.

Unlike the Audio/Visual trainers, Text is PARTICIPANT-LEVEL:
- One sequence (512 tokens) per participant — no windowing
- No participant-level aggregation needed
- Each DataLoader batch already contains one or several participants
- Predictions are directly participant-level

Includes:
- ReduceLROnPlateau scheduler
- Early stopping
- Macro F1 as primary selection metric
- GPU memory reporting
- Fusion export: per-participant probabilities, embeddings
- Dedicated log file (no conflict with audio/visual)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from training.checkpoint import save_checkpoint
from training.optimizer import create_optimizer
from utils.logger import get_logger


logger = get_logger(__name__)


def compute_text_metrics(
    y_true: List[int],
    y_pred: List[int],
    loss: float
) -> Dict[str, Any]:
    """
    Full metric suite for participant-level text evaluation.
    Primary metric: macro F1.
    """

    if len(y_true) == 0:
        return {
            "loss": float(loss),
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "binary_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_f1": [0.0, 0.0],
            "cm": [[0, 0], [0, 0]],
            "counts": {
                "true_0": 0,
                "true_1": 0,
                "pred_0": 0,
                "pred_1": 0
            }
        }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
        labels=[0, 1]
    ).tolist()

    return {
        "loss": float(loss),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "binary_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                zero_division=0
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "per_class_f1": per_class_f1,

        "cm": cm,

        "counts": {
            "true_0": y_true.count(0),
            "true_1": y_true.count(1),
            "pred_0": y_pred.count(0),
            "pred_1": y_pred.count(1),
        }
    }


class TextTrainer:
    """
    Trainer for the TextModel.

    Text is participant-level — each item in the DataLoader IS one participant.

    No window aggregation is required because every participant produces
    exactly one text sequence and therefore one participant-level prediction.

    Key differences from AudioTrainer:
    - forward() receives:
        (token_ids, attention_mask, targets[, pids])
    - No temporal window aggregation
    - Predictions are directly participant-level
    - pid -> probability information can optionally be returned for fusion
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
    ):
        self.device = torch.device(
            device
            if torch.cuda.is_available() and device == "cuda"
            else "cpu"
        )

        self.model = model.to(self.device)

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # ---------------------------------------------------------
        # LR Scheduler
        #
        # IMPORTANT:
        # Do NOT add verbose=False here.
        # The installed PyTorch version does not support that argument.
        # ---------------------------------------------------------
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

        # ---------------------------------------------------------
        # Best-model tracking
        # ---------------------------------------------------------
        self.best_macro_f1 = -1.0
        self.best_accuracy = -1.0
        self.best_epoch = -1

        self.global_step = 0

        logger.info(
            f"TextTrainer ready on {self.device}"
        )

    # =============================================================
    # GPU MEMORY
    # =============================================================

    def _gpu_memory_str(self) -> str:

        if self.device.type == "cuda":

            alloc = (
                torch.cuda.memory_allocated(self.device)
                / 1024**2
            )

            reserv = (
                torch.cuda.memory_reserved(self.device)
                / 1024**2
            )

            return (
                f"GPU: {alloc:.0f}MB alloc / "
                f"{reserv:.0f}MB reserved"
            )

        return "GPU: N/A (CPU mode)"

    # =============================================================
    # EPOCH
    # =============================================================

    def _run_epoch(
        self,
        dataloader: DataLoader,
        is_train: bool,
        return_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Participant-level forward/backward pass.

        Each batch item represents one participant.
        """

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0

        all_preds = []
        all_targets = []
        all_pids = []
        all_probs = []
        all_embeddings = []

        # ---------------------------------------------------------
        # Main batch loop
        # ---------------------------------------------------------

        with torch.set_grad_enabled(is_train):

            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 4:
                    token_ids, attention_mask, targets, pids = batch
                elif len(batch) == 3:
                    token_ids, attention_mask, targets = batch
                    pids = None
                else:
                    raise ValueError(
                        f"Unexpected batch size {len(batch)}. Expected 3 or 4 elements."
                    )

                token_ids = token_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                logits = self.model(token_ids, attention_mask)
                loss = self.criterion(logits, targets)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.global_step += 1

                batch_size = token_ids.size(0)
                total_loss += loss.item() * batch_size

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

                if return_debug:
                    # Extract 256-D pooled embeddings for multimodal fusion
                    with torch.no_grad():
                        emb = self.model.get_embedding(token_ids, attention_mask).cpu().numpy()
                        all_embeddings.extend(emb.tolist())

                if pids is not None:
                    if isinstance(pids, torch.Tensor):
                        all_pids.extend(pids.cpu().numpy().tolist())
                    else:
                        all_pids.extend([int(p) for p in pids])

                assert len(all_targets) == len(all_preds), (
                    f"Target/prediction count mismatch after batch {batch_idx}: "
                    f"{len(all_targets)} targets vs {len(all_preds)} preds"
                )

        # =========================================================
        # Epoch metrics
        # =========================================================

        dataset_size = max(len(dataloader.dataset), 1)
        avg_loss = total_loss / dataset_size

        metrics = compute_text_metrics(all_targets, all_preds, avg_loss)

        # =========================================================
        # Debug / fusion information
        # =========================================================

        if return_debug and all_pids:
            metrics["pid_list"] = [int(p) for p in all_pids]
            metrics["part_targets"] = all_targets
            metrics["part_preds"] = all_preds
            metrics["part_probs_c0"] = [float(p[0]) for p in all_probs]
            metrics["part_probs_c1"] = [float(p[1]) for p in all_probs]
            metrics["embeddings"] = all_embeddings

        return metrics

    # =============================================================
    # TRAIN
    # =============================================================

    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> Dict[str, Any]:

        return self._run_epoch(
            dataloader,
            is_train=True
        )

    # =============================================================
    # VALIDATION
    # =============================================================

    def validate(
        self,
        dataloader: DataLoader,
        return_debug: bool = False
    ) -> Dict[str, Any]:

        return self._run_epoch(
            dataloader,
            is_train=False,
            return_debug=return_debug
        )

    # =============================================================
    # FIT
    # =============================================================

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        fold: int = 1,
        seed: int = 42,
        save_dir: str = "outputs/checkpoints/text/fold1",
        log_fn=None,
        early_stopping_patience: int = 5,
        start_epoch: int = 1,
    ) -> None:

        logger.info(
            f"TextTrainer.fit: {epochs} epochs "
            f"on {self.device}, fold={fold}"
        )

        no_improve_count = 0

        # ---------------------------------------------------------
        # Epoch loop
        # ---------------------------------------------------------

        for epoch in range(
            start_epoch,
            epochs + 1
        ):

            # -----------------------------------------------------
            # Training
            # -----------------------------------------------------

            train_metrics = self.train_epoch(
                train_loader
            )

            # -----------------------------------------------------
            # Validation
            # -----------------------------------------------------

            val_metrics = self.validate(
                val_loader
            )

            # -----------------------------------------------------
            # Current LR
            # -----------------------------------------------------

            lr_current = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # -----------------------------------------------------
            # Validation Macro F1
            # -----------------------------------------------------

            val_macro_f1 = (
                val_metrics["macro_f1"]
            )

            # -----------------------------------------------------
            # GPU memory
            # -----------------------------------------------------

            gpu_str = self._gpu_memory_str()

            # -----------------------------------------------------
            # Scheduler
            # -----------------------------------------------------

            self.scheduler.step(
                val_macro_f1
            )

            # -----------------------------------------------------
            # Best model
            # -----------------------------------------------------

            is_best = (
                val_macro_f1
                > self.best_macro_f1
            )

            if is_best:

                self.best_macro_f1 = (
                    val_macro_f1
                )

                self.best_accuracy = (
                    val_metrics["accuracy"]
                )

                self.best_epoch = epoch

                no_improve_count = 0

            else:

                no_improve_count += 1

            # -----------------------------------------------------
            # Logging
            # -----------------------------------------------------

            log_line = (
                f"[Text Fold {fold}] "
                f"Epoch [{epoch}/{epochs}] "
                f"LR: {lr_current:.2e} | "
                f"Train Loss: "
                f"{train_metrics['loss']:.4f} "
                f"Acc: "
                f"{train_metrics['accuracy']:.4f} "
                f"MacroF1: "
                f"{train_metrics['macro_f1']:.4f} | "
                f"Val Loss: "
                f"{val_metrics['loss']:.4f} "
                f"Acc: "
                f"{val_metrics['accuracy']:.4f} "
                f"MacroF1: "
                f"{val_metrics['macro_f1']:.4f} "
                f"Prec: "
                f"{val_metrics['precision']:.4f} "
                f"Rec: "
                f"{val_metrics['recall']:.4f} | "
                f"CM: "
                f"{np.array(val_metrics['cm']).tolist()} | "
                f"{gpu_str}"
            )

            logger.info(
                log_line
            )

            if log_fn:

                log_fn(
                    log_line
                )

            # -----------------------------------------------------
            # Best checkpoint message
            # -----------------------------------------------------

            if is_best:

                logger.info(
                    f"  ✓ New best MacroF1="
                    f"{self.best_macro_f1:.4f} "
                    f"at epoch {epoch}. "
                    f"Checkpoint saved."
                )

            # -----------------------------------------------------
            # Scheduler state
            # -----------------------------------------------------

            scheduler_state = (
                self.scheduler.state_dict()
            )

            # -----------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Early stopping
            # -----------------------------------------------------

            if (
                no_improve_count
                >= early_stopping_patience
            ):

                logger.info(
                    f"Early stopping triggered "
                    f"after {no_improve_count} "
                    f"epochs without improvement. "
                    f"Best MacroF1="
                    f"{self.best_macro_f1:.4f} "
                    f"at epoch "
                    f"{self.best_epoch}."
                )

                break

        # =========================================================
        # Training complete
        # =========================================================

        logger.info(
            f"Training complete. "
            f"Best MacroF1="
            f"{self.best_macro_f1:.4f} "
            f"at epoch "
            f"{self.best_epoch}."
        )
