"""
Audio Agent 5-Fold Cross-Validation Training Script.

Usage (on Ubuntu server):
    python -m scripts.train_audio

Key design decisions:
- Uses SAME fold assignments as Visual Agent (StratifiedKFold, seed=42, all 134 participants)
- Fold assignments loaded from outputs/checkpoints/visual/fold_assignments.json if it exists,
  otherwise reconstructed deterministically with the same seed (guaranteed identical result)
- Window-level training, participant-level validation (softmax probability averaging)
- Isolated outputs: checkpoints/audio/, metrics/audio/, logs/audio/, predictions/audio/
- Resume support: detects incomplete folds from metrics files
- Class weights computed per-fold from TRAINING participants only
- Primary metric: Macro F1 (not accuracy, not binary F1)
- Early stopping: patience=5 epochs
- Max epochs: 30
- Batch size: 16 (conservative for VRAM; simultaneous with text agent)
- num_workers: 0 (4-core CPU, two simultaneous processes)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

Output files per fold:
    outputs/checkpoints/audio/foldN/best_model.pt
    outputs/checkpoints/audio/foldN/last_model.pt
    outputs/metrics/audio/fold_N_metrics.json
    outputs/predictions/audio/fold_N_predictions.json

Final summary:
    outputs/metrics/audio/cross_validation_results.json
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
from preprocessing.audio.pipeline import AudioPreprocessingPipeline, AudioPreprocessingConfig
from models.audio_agent.audio_model import AudioModel
from training.audio_trainer import AudioTrainer
from training.checkpoint import load_checkpoint
from utils.seed import seed_everything

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SEED          = 42
MAX_EPOCHS    = 30
BATCH_SIZE    = 16
EARLY_STOP    = 5
NUM_WORKERS   = 0    # keep 0: CPU has 4 cores, 2 processes run simultaneously
K_FOLDS       = 5
LR            = 1e-4
WEIGHT_DECAY  = 1e-3

WINDOW_SIZE   = 150
STRIDE        = 30

AUDIO_D_MODEL = 128  # smaller than visual (256) to share VRAM safely

# ──────────────────────────────────────────────────────────────
# Logging setup — ISOLATED to audio log file only
# ──────────────────────────────────────────────────────────────
LOG_DIR = Path("outputs/logs/audio")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "audio_5fold_training.txt"

def setup_audio_logger() -> logging.Logger:
    """Returns a logger that writes ONLY to the audio log file (not project.log)."""
    audio_logger = logging.getLogger("audio_training")
    audio_logger.setLevel(logging.INFO)
    if not audio_logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(str(LOG_FILE))
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        audio_logger.addHandler(fh)
        audio_logger.addHandler(ch)
    return audio_logger


# ──────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────
class ProcessedAudioDataset(Dataset):
    """
    Wraps pre-processed audio window tensors.
    Yields (window_tensor, label, participant_id).
    """
    def __init__(self, data: List[Tuple[torch.Tensor, int, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ──────────────────────────────────────────────────────────────
# Participant extraction
# ──────────────────────────────────────────────────────────────
def extract_audio_participants(
    dataset: DAICDataset,
    pipeline: AudioPreprocessingPipeline,
    is_train: bool,
    alog: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Processes all participants in dataset through the audio pipeline.
    If is_train=True, first fits the pipeline normalizer.
    Returns list of {pid, label, windows}.
    """
    if is_train:
        alog.info("Fitting audio normalizer on training participants...")
        train_audios = []
        for i in range(len(dataset)):
            try:
                train_audios.append(dataset[i]["audio"])
            except Exception as e:
                alog.warning(f"  Skipping participant at index {i} during fit: {e}")
        pipeline.fit(train_audios)
        alog.info("Audio normalizer fitted.")

    pt_data   = []
    skipped   = []

    for i in range(len(dataset)):
        try:
            pt     = dataset[i]
            pid    = pt["participant_id"]
            label  = pt["labels"]["phq8_binary"]
            tensor = pipeline.transform(pid, pt["audio"])
            w_count = tensor.size(0)

            if w_count == 0:
                alog.warning(f"  Participant {pid}: 0 windows generated — skipping.")
                skipped.append(pid)
                continue

            windows = [tensor[w] for w in range(w_count)]
            pt_data.append({"pid": pid, "label": label, "windows": windows})

        except Exception as e:
            try:
                pid = dataset.metadata_df.iloc[i]["participant_id"]
            except Exception:
                pid = f"index_{i}"
            alog.warning(f"  Participant {pid}: preprocessing failed — {e}")
            skipped.append(pid)

    alog.info(f"Audio extraction complete: {len(pt_data)} loaded, {len(skipped)} skipped.")
    return pt_data


# ──────────────────────────────────────────────────────────────
# Fold assignment — identical to visual training
# ──────────────────────────────────────────────────────────────
def get_or_create_fold_assignments(
    all_pts: List[Dict],
    alog: logging.Logger,
) -> Tuple[Dict, List]:
    """
    Loads fold assignments from the visual checkpoint directory if available.
    Otherwise reconstructs deterministically with seed=42 (guaranteed identical to visual).
    """
    fold_file = Path("outputs/checkpoints/visual/fold_assignments.json")

    if fold_file.exists():
        alog.info(f"Loading existing fold assignments from {fold_file}")
        with open(fold_file) as f:
            assignments = json.load(f)
        # Reconstruct ordered fold list from assignments
        fold_list = []
        for fold_key in sorted(assignments.keys()):  # Fold_1, Fold_2, ...
            fold_info = assignments[fold_key]
            train_pids = set(fold_info["train"])
            val_pids   = set(fold_info["val"])
            pid_list   = [pt["pid"] for pt in all_pts]
            train_idx  = [i for i, pt in enumerate(all_pts) if pt["pid"] in train_pids]
            val_idx    = [i for i, pt in enumerate(all_pts) if pt["pid"] in val_pids]
            fold_list.append((train_idx, val_idx))
        return assignments, fold_list

    # Reconstruct deterministically — same as train_visual.py
    alog.warning(
        "fold_assignments.json NOT found. Reconstructing deterministically "
        "with StratifiedKFold(n_splits=5, shuffle=True, random_state=42). "
        "This is mathematically identical to visual training if metadata.csv is unchanged."
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

    # Save for future use by text agent and fusion
    fold_file.parent.mkdir(parents=True, exist_ok=True)
    with open(fold_file, "w") as f:
        json.dump(assignments, f, indent=4)
    alog.info(f"Fold assignments saved to {fold_file}")

    return assignments, fold_list


# ──────────────────────────────────────────────────────────────
# Main CV loop
# ──────────────────────────────────────────────────────────────
def run_audio_kfold_cv():
    seed_everything(SEED)
    alog = setup_audio_logger()
    start_time = time.time()

    # Check if all folds already complete
    cv_results_file = Path("outputs/metrics/audio/cross_validation_results.json")
    if cv_results_file.exists():
        alog.info("Audio cross-validation already completed. "
                  "Remove cross_validation_results.json to restart.")
        return

    alog.info("=" * 60)
    alog.info("AUDIO AGENT — 5-FOLD CROSS-VALIDATION TRAINING")
    alog.info("=" * 60)
    alog.info(f"Batch size   : {BATCH_SIZE}")
    alog.info(f"Max epochs   : {MAX_EPOCHS}")
    alog.info(f"Early stop   : {EARLY_STOP}")
    alog.info(f"LR           : {LR}")
    alog.info(f"Window size  : {WINDOW_SIZE}, stride: {STRIDE}")
    alog.info(f"D_model      : {AUDIO_D_MODEL}")
    alog.info(f"Num workers  : {NUM_WORKERS}")
    alog.info(f"Log file     : {LOG_FILE}")

    # ── Load data ──────────────────────────────────────────────
    alog.info("Loading DAIC-WOZ datasets (audio only)...")
    train_dataset = DAICDataset(split="train", load_visual=False, load_audio=True, load_text=False)
    dev_dataset   = DAICDataset(split="dev",   load_visual=False, load_audio=True, load_text=False)

    config   = AudioPreprocessingConfig(window_size=WINDOW_SIZE, stride=STRIDE)
    pipeline = AudioPreprocessingPipeline(config)

    alog.info("Extracting and preprocessing audio participants (train split)...")
    train_pts = extract_audio_participants(train_dataset, pipeline, is_train=True,  alog=alog)
    alog.info("Extracting and preprocessing audio participants (dev split)...")
    dev_pts   = extract_audio_participants(dev_dataset,   pipeline, is_train=False, alog=alog)

    all_pts = train_pts + dev_pts
    alog.info(f"Total participants available for CV: {len(all_pts)}")

    # ── Determine input feature dimension ──────────────────────
    # F_dim is dynamic — read from the first participant's window tensor
    sample_win = all_pts[0]["windows"][0]
    input_dim  = sample_win.shape[-1]
    alog.info(f"Detected audio feature dimension: {input_dim}")

    # ── Get fold assignments ───────────────────────────────────
    assignments, fold_list = get_or_create_fold_assignments(all_pts, alog)

    # ── Check for already-completed folds ─────────────────────
    metrics_dir  = Path("outputs/metrics/audio")
    ckpt_base    = Path("outputs/checkpoints/audio")
    pred_dir     = Path("outputs/predictions/audio")
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
        alog.info(f"Detected {len(fold_metrics)} completed fold(s). Resuming remaining...")

    # ── K-Fold loop ────────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(fold_list, 1):
        if fold <= len(fold_metrics):
            alog.info(f"Skipping Fold {fold}/{K_FOLDS} — already completed.")
            continue

        alog.info(f"\n{'='*50}")
        alog.info(f"  AUDIO FOLD {fold}/{K_FOLDS}")
        alog.info(f"{'='*50}")

        train_pids = [all_pts[i]["pid"] for i in train_idx]
        val_pids   = [all_pts[i]["pid"] for i in val_idx]
        alog.info(f"Train participants ({len(train_pids)}): {train_pids}")
        alog.info(f"Val   participants ({len(val_pids)}):   {val_pids}")

        # ── Flatten to window-level data ───────────────────────
        train_data = []
        val_data   = []
        part_0_count = part_1_count = 0

        for idx in train_idx:
            pt = all_pts[idx]
            if pt["label"] == 0:
                part_0_count += 1
            else:
                part_1_count += 1
            for w in pt["windows"]:
                train_data.append((w, pt["label"], pt["pid"]))

        for idx in val_idx:
            pt = all_pts[idx]
            for w in pt["windows"]:
                val_data.append((w, pt["label"], pt["pid"]))

        # ── Class weights (training participants only) ─────────
        alog.info(f"Fold {fold} class distribution:")
        alog.info(f"  Healthy participants   : {part_0_count}")
        alog.info(f"  Depressed participants : {part_1_count}")

        class_weights_tensor = None
        if part_0_count > 0 and part_1_count > 0:
            counts  = np.array([part_0_count, part_1_count], dtype=np.float32)
            weights = 1.0 / counts
            weights = weights / weights.sum() * 2.0
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
            alog.info(f"  Class weights          : [{weights[0]:.4f}, {weights[1]:.4f}]")
        else:
            alog.warning("  Only one class present — class weights disabled.")

        # ── DataLoaders ────────────────────────────────────────
        train_loader = DataLoader(
            ProcessedAudioDataset(train_data),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            ProcessedAudioDataset(val_data),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        alog.info(f"Train windows: {len(train_data)} | Val windows: {len(val_data)}")

        # ── Model & Trainer ────────────────────────────────────
        model   = AudioModel(input_dim=input_dim, d_model=AUDIO_D_MODEL)
        trainer = AudioTrainer(
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            class_weights=class_weights_tensor,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # ── Resume support ─────────────────────────────────────
        fold_dir       = ckpt_base / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path      = fold_dir / "last_model.pt"
        start_epoch    = 1

        if ckpt_path.exists():
            alog.info(f"  Resuming Fold {fold} from {ckpt_path}...")
            try:
                state = torch.load(str(ckpt_path), map_location=trainer.device)
                model.load_state_dict(state["model_state_dict"])
                if state.get("optimizer_state_dict"):
                    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
                if state.get("scheduler_state_dict"):
                    trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
                trainer.best_macro_f1 = state.get("best_f1", -1.0)
                trainer.best_accuracy = state.get("best_accuracy", -1.0)
                start_epoch = state.get("epoch", 0) + 1
                alog.info(f"  Resumed from epoch {start_epoch - 1}. Best MacroF1={trainer.best_macro_f1:.4f}")
            except Exception as e:
                alog.warning(f"  Checkpoint load failed ({e}). Starting from scratch.")
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
                log_fn=lambda msg: None,  # already handled by AudioTrainer's logger
                early_stopping_patience=EARLY_STOP,
                start_epoch=start_epoch,
            )
        else:
            alog.info(f"  Fold {fold} already finished {MAX_EPOCHS} epochs.")

        # ── Final evaluation with best model ───────────────────
        alog.info(f"  Running final validation for Fold {fold} (best model)...")
        best_model_path = fold_dir / "best_model.pt"
        if best_model_path.exists():
            try:
                state = torch.load(str(best_model_path), map_location=trainer.device)
                model.load_state_dict(state["model_state_dict"])
                best_epoch_saved = state.get("epoch", -1)
                alog.info(f"  Best model loaded from epoch {best_epoch_saved}.")
            except Exception as e:
                alog.warning(f"  Could not load best model: {e}. Using last model.")
                best_epoch_saved = trainer.best_epoch
        else:
            best_epoch_saved = trainer.best_epoch

        val_metrics = trainer.validate(val_loader, return_debug=True)

        # ── Save predictions for fusion ────────────────────────
        fusion_predictions = []
        if "pid_list" in val_metrics and val_metrics["pid_list"]:
            for i, pid in enumerate(val_metrics["pid_list"]):
                fusion_predictions.append({
                    "participant_id": int(pid),
                    "true_label":     int(val_metrics["part_targets"][i]),
                    "prediction":     int(val_metrics["part_preds"][i]),
                    "probability_class_0": float(val_metrics["part_probs_c0"][i]),
                    "probability_class_1": float(val_metrics["part_probs_c1"][i]),
                    "fold": fold,
                })

        pred_file = pred_dir / f"fold_{fold}_predictions.json"
        with open(pred_file, "w") as pf:
            json.dump(fusion_predictions, pf, indent=4)
        alog.info(f"  Predictions saved → {pred_file}")

        # ── Save fold metrics ──────────────────────────────────
        metrics_to_save = {
            "fold": fold,
            "accuracy":          float(val_metrics["accuracy"]),
            "macro_f1":          float(val_metrics["macro_f1"]),
            "binary_f1":         float(val_metrics["binary_f1"]),
            "precision":         float(val_metrics["precision"]),
            "recall":            float(val_metrics["recall"]),
            "per_class_f1":      val_metrics["per_class_f1"],
            "confusion_matrix":  np.array(val_metrics["cm"]).tolist(),
            "val_loss":          float(val_metrics["loss"]),
            "best_epoch":        best_epoch_saved,
            "max_epochs":        MAX_EPOCHS,
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
        alog.info(f"  Fold {fold} metrics saved → {metric_file}")
        alog.info(
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
    alog.info("\n" + "=" * 60)
    alog.info("AUDIO AGENT — FINAL CROSS-VALIDATION RESULTS")
    alog.info("=" * 60)
    alog.info(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    alog.info(f"Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    alog.info(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    alog.info(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    alog.info(f"Results  : {cv_results_file}")
    alog.info(f"Total time: {total_min:.1f} minutes")


if __name__ == "__main__":
    run_audio_kfold_cv()
