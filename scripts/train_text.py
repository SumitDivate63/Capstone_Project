"""
Text Agent 5-Fold Cross-Validation Training Script.

Usage (on Ubuntu server):
    python -m scripts.train_text

Key design decisions:
- Uses SAME fold assignments as Visual Agent (identical SKF seed=42)
- Fold assignments shared with audio script from outputs/checkpoints/visual/fold_assignments.json
- Text is PARTICIPANT-LEVEL: one 512-token sequence per participant, no windowing
  → No window aggregation needed; each batch item IS one participant prediction
- Vocabulary fitted on TRAINING participants of each fold (prevents leakage)
- Isolated outputs: checkpoints/text/, metrics/text/, logs/text/, predictions/text/
- Resume support: detects incomplete folds from metrics files
- Class weights computed per-fold from TRAINING participants only
- Primary metric: Macro F1
- Early stopping: patience=5 epochs
- Max epochs: 20
- Batch size: 8 (participant-level, conservative)
- num_workers: 0 (4-core CPU, two simultaneous processes)

Output files per fold:
    outputs/checkpoints/text/foldN/best_model.pt
    outputs/checkpoints/text/foldN/last_model.pt
    outputs/metrics/text/fold_N_metrics.json
    outputs/predictions/text/fold_N_predictions.json

Final summary:
    outputs/metrics/text/cross_validation_results.json
"""

import os
import json
import random
import time
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets.daic_dataset import DAICDataset
from preprocessing.text.pipeline import TextPreprocessingPipeline, TextPreprocessingConfig
from models.text_agent.text_model import TextModel
from training.text_trainer import TextTrainer
from training.checkpoint import load_checkpoint
from utils.seed import seed_everything

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SEED          = 42
MAX_EPOCHS    = 20
BATCH_SIZE    = 8      # participant-level batches
EARLY_STOP    = 5
NUM_WORKERS   = 0
K_FOLDS       = 5
LR            = 1e-4
WEIGHT_DECAY  = 1e-3

MAX_SEQ_LEN   = 512
TEXT_D_MODEL  = 256
VOCAB_SIZE    = 30000   # from TextPreprocessingConfig

# ──────────────────────────────────────────────────────────────
# Logging setup — ISOLATED to text log file only
# ──────────────────────────────────────────────────────────────
LOG_DIR  = Path("outputs/logs/text")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "text_5fold_training.txt"

def setup_text_logger() -> logging.Logger:
    """Returns a logger that writes ONLY to the text log file (not project.log)."""
    tlog = logging.getLogger("text_training")
    tlog.setLevel(logging.INFO)
    if not tlog.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(str(LOG_FILE))
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        tlog.addHandler(fh)
        tlog.addHandler(ch)
    return tlog


# ──────────────────────────────────────────────────────────────
# Dataset wrapper — PARTICIPANT-LEVEL (no windowing)
# ──────────────────────────────────────────────────────────────
class ProcessedTextDataset(Dataset):
    """
    Wraps preprocessed participant text sequences.
    Yields (token_ids, attention_mask, label, participant_id).
    Each item is ONE participant — no windowing.
    """
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["token_ids"],      # LongTensor (512,)
            item["attention_mask"], # LongTensor (512,)
            item["label"],          # int
            item["pid"],            # int
        )


# ──────────────────────────────────────────────────────────────
# Participant extraction
# ──────────────────────────────────────────────────────────────
def extract_text_participants(
    dataset: DAICDataset,
    pipeline: TextPreprocessingPipeline,
    is_train: bool,
    tlog: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Processes all participants through the text pipeline.
    If is_train=True, first fits the vocabulary.
    Returns list of {pid, label, token_ids, attention_mask}.

    IMPORTANT: Therapist lines (Ellie) are filtered in TextPreprocessor
    (keep_ellie=False is the default) — only participant speech is used.
    """
    if is_train:
        tlog.info("Fitting vocabulary on training participants...")
        train_texts = []
        for i in range(len(dataset)):
            try:
                train_texts.append(dataset[i]["text"])
            except Exception as e:
                tlog.warning(f"  Skipping participant at index {i} during vocab fit: {e}")
        pipeline.fit(train_texts)
        tlog.info(f"Vocabulary built. Size: {len(pipeline.vocabulary.word2id)}")

    pt_data = []
    skipped = []

    for i in range(len(dataset)):
        try:
            pt    = dataset[i]
            pid   = pt["participant_id"]
            label = pt["labels"]["phq8_binary"]

            output = pipeline.transform(pid, pt["text"])
            pt_data.append({
                "pid":            pid,
                "label":          label,
                "token_ids":      output["token_ids"],       # LongTensor (512,)
                "attention_mask": output["attention_mask"],  # LongTensor (512,)
            })
        except Exception as e:
            try:
                pid = dataset.metadata_df.iloc[i]["participant_id"]
            except Exception:
                pid = f"index_{i}"
            tlog.warning(f"  Participant {pid}: text preprocessing failed — {e}")
            skipped.append(pid)

    tlog.info(f"Text extraction complete: {len(pt_data)} loaded, {len(skipped)} skipped.")
    return pt_data


# ──────────────────────────────────────────────────────────────
# Fold assignment — identical to visual training
# ──────────────────────────────────────────────────────────────
def get_or_create_fold_assignments(
    all_pts: List[Dict],
    tlog: logging.Logger,
) -> Tuple[Dict, List]:
    """
    Loads fold assignments from visual checkpoint dir if available.
    Otherwise reconstructs deterministically (identical to visual training).
    """
    fold_file = Path("outputs/checkpoints/visual/fold_assignments.json")

    if fold_file.exists():
        tlog.info(f"Loading existing fold assignments from {fold_file}")
        with open(fold_file) as f:
            assignments = json.load(f)
        fold_list = []
        for fold_key in sorted(assignments.keys()):
            fold_info  = assignments[fold_key]
            train_pids = set(fold_info["train"])
            val_pids   = set(fold_info["val"])
            train_idx  = [i for i, pt in enumerate(all_pts) if pt["pid"] in train_pids]
            val_idx    = [i for i, pt in enumerate(all_pts) if pt["pid"] in val_pids]
            fold_list.append((train_idx, val_idx))
        return assignments, fold_list

    tlog.warning(
        "fold_assignments.json NOT found. Reconstructing deterministically "
        "with StratifiedKFold(n_splits=5, shuffle=True, random_state=42). "
        "Identical to visual training if metadata.csv is unchanged."
    )
    pids   = [pt["pid"]   for pt in all_pts]
    labels = [pt["label"] for pt in all_pts]
    skf    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_list   = list(skf.split(pids, labels))
    assignments = {}
    for fold_num, (train_idx, val_idx) in enumerate(fold_list, 1):
        assignments[f"Fold_{fold_num}"] = {
            "train": [pids[i] for i in train_idx],
            "val":   [pids[i] for i in val_idx],
        }

    fold_file.parent.mkdir(parents=True, exist_ok=True)
    with open(fold_file, "w") as f:
        json.dump(assignments, f, indent=4)
    tlog.info(f"Fold assignments saved to {fold_file}")

    return assignments, fold_list


# ──────────────────────────────────────────────────────────────
# Main CV loop
# ──────────────────────────────────────────────────────────────
def run_text_kfold_cv():
    seed_everything(SEED)
    tlog = setup_text_logger()
    start_time = time.time()

    cv_results_file = Path("outputs/metrics/text/cross_validation_results.json")
    if cv_results_file.exists():
        tlog.info("Text cross-validation already completed. "
                  "Remove cross_validation_results.json to restart.")
        return

    tlog.info("=" * 60)
    tlog.info("TEXT AGENT — 5-FOLD CROSS-VALIDATION TRAINING")
    tlog.info("=" * 60)
    tlog.info(f"Batch size   : {BATCH_SIZE} (participant-level — no windowing)")
    tlog.info(f"Max epochs   : {MAX_EPOCHS}")
    tlog.info(f"Early stop   : {EARLY_STOP}")
    tlog.info(f"LR           : {LR}")
    tlog.info(f"Max seq len  : {MAX_SEQ_LEN}")
    tlog.info(f"D_model      : {TEXT_D_MODEL}")
    tlog.info(f"Vocab size   : {VOCAB_SIZE}")
    tlog.info(f"Num workers  : {NUM_WORKERS}")
    tlog.info(f"Log file     : {LOG_FILE}")
    tlog.info("Therapist lines (Ellie): EXCLUDED (keep_ellie=False)")

    # ── Load data ──────────────────────────────────────────────
    tlog.info("Loading DAIC-WOZ datasets (text only)...")
    train_dataset = DAICDataset(split="train", load_visual=False, load_audio=False, load_text=True)
    dev_dataset   = DAICDataset(split="dev",   load_visual=False, load_audio=False, load_text=True)

    # Text pipeline is fit per-fold (vocabulary from training participants only)
    # We first load raw data for all participants, then process per fold

    tlog.info("Loading raw participant text data...")
    all_raw_train = []
    for i in range(len(train_dataset)):
        try:
            pt = train_dataset[i]
            all_raw_train.append({
                "pid":     pt["participant_id"],
                "label":   pt["labels"]["phq8_binary"],
                "text":    pt["text"],
            })
        except Exception as e:
            tlog.warning(f"  Train participant index {i} failed: {e}")

    all_raw_dev = []
    for i in range(len(dev_dataset)):
        try:
            pt = dev_dataset[i]
            all_raw_dev.append({
                "pid":   pt["participant_id"],
                "label": pt["labels"]["phq8_binary"],
                "text":  pt["text"],
            })
        except Exception as e:
            tlog.warning(f"  Dev participant index {i} failed: {e}")

    all_raw = all_raw_train + all_raw_dev
    tlog.info(f"Total raw participants loaded: {len(all_raw)}")

    # ── Get fold assignments ───────────────────────────────────
    # Build a lightweight list for fold assignment (pip, label only)
    all_pts_summary = [{"pid": r["pid"], "label": r["label"]} for r in all_raw]
    assignments, fold_list = get_or_create_fold_assignments(all_pts_summary, tlog)

    # ── Output directories ─────────────────────────────────────
    metrics_dir = Path("outputs/metrics/text")
    ckpt_base   = Path("outputs/checkpoints/text")
    pred_dir    = Path("outputs/predictions/text")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for f in range(1, K_FOLDS + 1):
        mf = metrics_dir / f"fold_{f}_metrics.json"
        if mf.exists():
            with open(mf) as fp:
                fold_metrics.append(json.load(fp))

    if fold_metrics:
        tlog.info(f"Detected {len(fold_metrics)} completed fold(s). Resuming...")

    # ── K-Fold loop ────────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(fold_list, 1):
        if fold <= len(fold_metrics):
            tlog.info(f"Skipping Fold {fold}/{K_FOLDS} — already completed.")
            continue

        tlog.info(f"\n{'='*50}")
        tlog.info(f"  TEXT FOLD {fold}/{K_FOLDS}")
        tlog.info(f"{'='*50}")

        # ── Build per-fold pipeline (vocabulary from training only) ──
        config   = TextPreprocessingConfig(
            max_sequence_length=MAX_SEQ_LEN,
            lowercase=True,
            keep_ellie=False,   # Exclude therapist speech
            tokenizer="whitespace",
            vocab_size=VOCAB_SIZE,
        )
        pipeline = TextPreprocessingPipeline(config)

        # Fit vocabulary on training participants only
        train_raw_text_data = [all_raw[i]["text"] for i in train_idx]
        tlog.info(f"  Fitting fold {fold} vocabulary on {len(train_raw_text_data)} training participants...")
        pipeline.fit(train_raw_text_data)
        actual_vocab_size = len(pipeline.vocabulary.word2id)
        tlog.info(f"  Vocabulary size: {actual_vocab_size}")

        # Transform all participants with fold-specific pipeline
        def transform_participants(indices, label_name=""):
            data = []
            skipped = []
            for idx in indices:
                raw = all_raw[idx]
                pid   = raw["pid"]
                label = raw["label"]
                try:
                    output = pipeline.transform(pid, raw["text"])
                    data.append({
                        "pid":            pid,
                        "label":          label,
                        "token_ids":      output["token_ids"],
                        "attention_mask": output["attention_mask"],
                    })
                except Exception as e:
                    tlog.warning(f"    Participant {pid} ({label_name}) transform failed: {e}")
                    skipped.append(pid)
            return data, skipped

        tlog.info(f"  Transforming training participants...")
        train_pt_data, train_skip = transform_participants(train_idx, "train")
        tlog.info(f"  Transforming validation participants...")
        val_pt_data,   val_skip   = transform_participants(val_idx, "val")

        train_pids = [d["pid"] for d in train_pt_data]
        val_pids   = [d["pid"] for d in val_pt_data]

        tlog.info(f"  Train participants ({len(train_pids)}): {train_pids}")
        tlog.info(f"  Val   participants ({len(val_pids)}):   {val_pids}")

        # Verify no participant leakage
        overlap = set(train_pids) & set(val_pids)
        if overlap:
            raise RuntimeError(f"PARTICIPANT LEAKAGE in fold {fold}: {overlap}")

        # ── Class weights ──────────────────────────────────────
        part_0_count = sum(1 for d in train_pt_data if d["label"] == 0)
        part_1_count = sum(1 for d in train_pt_data if d["label"] == 1)

        tlog.info(f"  Fold {fold} class distribution (training only):")
        tlog.info(f"    Healthy participants   : {part_0_count}")
        tlog.info(f"    Depressed participants : {part_1_count}")

        class_weights_tensor = None
        if part_0_count > 0 and part_1_count > 0:
            counts  = np.array([part_0_count, part_1_count], dtype=np.float32)
            weights = 1.0 / counts
            weights = weights / weights.sum() * 2.0
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
            tlog.info(f"    Class weights          : [{weights[0]:.4f}, {weights[1]:.4f}]")
        else:
            tlog.warning("    Only one class present — class weights disabled.")

        # ── DataLoaders ────────────────────────────────────────
        train_loader = DataLoader(
            ProcessedTextDataset(train_pt_data),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            ProcessedTextDataset(val_pt_data),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        tlog.info(f"  Train participants in loader: {len(train_pt_data)}")
        tlog.info(f"  Val   participants in loader: {len(val_pt_data)}")

        # ── Model & Trainer ────────────────────────────────────
        model   = TextModel(
            vocab_size=actual_vocab_size,
            d_model=TEXT_D_MODEL,
            max_seq_len=MAX_SEQ_LEN,
        )
        trainer = TextTrainer(
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            class_weights=class_weights_tensor,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # ── Resume support ─────────────────────────────────────
        fold_dir    = ckpt_base / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path   = fold_dir / "last_model.pt"
        start_epoch = 1

        if ckpt_path.exists():
            tlog.info(f"  Resuming Fold {fold} from {ckpt_path}...")
            try:
                state = torch.load(str(ckpt_path), map_location=trainer.device)
                # Note: vocab_size may differ if pipeline changed — catch mismatch
                model.load_state_dict(state["model_state_dict"])
                if state.get("optimizer_state_dict"):
                    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
                if state.get("scheduler_state_dict"):
                    trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
                trainer.best_macro_f1 = state.get("best_f1", -1.0)
                trainer.best_accuracy = state.get("best_accuracy", -1.0)
                start_epoch = state.get("epoch", 0) + 1
                tlog.info(f"  Resumed from epoch {start_epoch - 1}.")
            except Exception as e:
                tlog.warning(f"  Checkpoint load failed ({e}). Starting from scratch.")
                start_epoch = 1

        # ── Training ───────────────────────────────────────────
        if start_epoch <= MAX_EPOCHS:
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=MAX_EPOCHS,
                fold=fold,
                seed=SEED,
                save_dir=str(fold_dir),
                log_fn=lambda msg: None,
                early_stopping_patience=EARLY_STOP,
                start_epoch=start_epoch,
            )
        else:
            tlog.info(f"  Fold {fold} already finished {MAX_EPOCHS} epochs.")

        # ── Final evaluation with best model ───────────────────
        tlog.info(f"  Running final validation for Fold {fold} (best model)...")
        best_model_path = fold_dir / "best_model.pt"
        best_epoch_saved = trainer.best_epoch
        if best_model_path.exists():
            try:
                state = torch.load(str(best_model_path), map_location=trainer.device)
                model.load_state_dict(state["model_state_dict"])
                best_epoch_saved = state.get("epoch", trainer.best_epoch)
                tlog.info(f"  Best model loaded from epoch {best_epoch_saved}.")
            except Exception as e:
                tlog.warning(f"  Could not load best model: {e}. Using last model.")

        val_metrics = trainer.validate(val_loader, return_debug=True)

        # Verify participant count
        if "pid_list" in val_metrics:
            n_pred = len(val_metrics["pid_list"])
            n_val  = len(val_pt_data)
            if n_pred != n_val:
                tlog.warning(
                    f"  Participant count mismatch: {n_pred} predictions vs {n_val} val participants!"
                )
            else:
                tlog.info(f"  ✓ Participant predictions: {n_pred} == {n_val} val participants.")

        # ── Save predictions for fusion ────────────────────────
        fusion_predictions = []
        if "pid_list" in val_metrics and val_metrics["pid_list"]:
            has_emb = "embeddings" in val_metrics and len(val_metrics["embeddings"]) == len(val_metrics["pid_list"])
            for i, pid in enumerate(val_metrics["pid_list"]):
                pred_item = {
                    "participant_id":     int(pid),
                    "true_label":         int(val_metrics["part_targets"][i]),
                    "prediction":         int(val_metrics["part_preds"][i]),
                    "probability_class_0": float(val_metrics["part_probs_c0"][i]),
                    "probability_class_1": float(val_metrics["part_probs_c1"][i]),
                    "fold": fold,
                }
                if has_emb:
                    pred_item["embedding"] = [float(v) for v in val_metrics["embeddings"][i]]
                fusion_predictions.append(pred_item)

        pred_file = pred_dir / f"fold_{fold}_predictions.json"
        with open(pred_file, "w") as pf:
            json.dump(fusion_predictions, pf, indent=4)
        tlog.info(f"  Predictions saved → {pred_file}")

        # ── Save fold metrics ──────────────────────────────────
        metrics_to_save = {
            "fold": fold,
            "accuracy":           float(val_metrics["accuracy"]),
            "macro_f1":           float(val_metrics["macro_f1"]),
            "binary_f1":          float(val_metrics["binary_f1"]),
            "precision":          float(val_metrics["precision"]),
            "recall":             float(val_metrics["recall"]),
            "per_class_f1":       val_metrics["per_class_f1"],
            "confusion_matrix":   np.array(val_metrics["cm"]).tolist(),
            "val_loss":           float(val_metrics["loss"]),
            "best_epoch":         best_epoch_saved,
            "max_epochs":         MAX_EPOCHS,
            "vocab_size_used":    actual_vocab_size,
            "class_distribution": {
                "train_healthy":   int(part_0_count),
                "train_depressed": int(part_1_count),
            },
            "train_pids": train_pids,
            "val_pids":   val_pids,
        }

        metric_file = metrics_dir / f"fold_{fold}_metrics.json"
        with open(metric_file, "w") as mf:
            json.dump(metrics_to_save, mf, indent=4)
        tlog.info(f"  Fold {fold} metrics saved → {metric_file}")
        tlog.info(
            f"  Fold {fold} Results: Acc={metrics_to_save['accuracy']:.4f} "
            f"MacroF1={metrics_to_save['macro_f1']:.4f} "
            f"P={metrics_to_save['precision']:.4f} "
            f"R={metrics_to_save['recall']:.4f}"
        )

        fold_metrics.append(metrics_to_save)

    # ── Cross-validation summary ───────────────────────────────
    accs  = [m["accuracy"]  for m in fold_metrics]
    f1s   = [m["macro_f1"]  for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs  = [m["recall"]    for m in fold_metrics]

    cv_summary = {
        "mean_accuracy":   float(np.mean(accs)),
        "std_accuracy":    float(np.std(accs)),
        "mean_macro_f1":   float(np.mean(f1s)),
        "std_macro_f1":    float(np.std(f1s)),
        "mean_precision":  float(np.mean(precs)),
        "std_precision":   float(np.std(precs)),
        "mean_recall":     float(np.mean(recs)),
        "std_recall":      float(np.std(recs)),
        "per_fold_metrics": fold_metrics,
    }

    try:
        agg_cm = np.zeros((2, 2))
        for m in fold_metrics:
            agg_cm += np.array(m["confusion_matrix"])
        cv_summary["aggregate_confusion_matrix"] = agg_cm.tolist()
    except Exception:
        pass

    cv_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_results_file, "w") as f:
        json.dump(cv_summary, f, indent=4)

    total_min = (time.time() - start_time) / 60
    tlog.info("\n" + "=" * 60)
    tlog.info("TEXT AGENT — FINAL CROSS-VALIDATION RESULTS")
    tlog.info("=" * 60)
    tlog.info(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    tlog.info(f"Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    tlog.info(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    tlog.info(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    tlog.info(f"Results  : {cv_results_file}")
    tlog.info(f"Total time: {total_min:.1f} minutes")


if __name__ == "__main__":
    run_text_kfold_cv()
