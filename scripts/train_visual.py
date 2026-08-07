import os
import json
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import VisualPreprocessingPipeline, VisualPreprocessingConfig
from models.visual.visual_model import VisualModel
from training.trainer import VisualTrainer
from utils.logger import get_logger

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProcessedVisualDataset(Dataset):
    """
    Zero-copy wrapper translating fully transformed window tensors cleanly into standard iterable shapes.
    Yields (window_tensor, label, participant_id).
    """
    def __init__(self, data: List[Tuple[torch.Tensor, int, int]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def extract_participant_sequences(
    dataset: DAICDataset, 
    pipeline: VisualPreprocessingPipeline, 
    is_train: bool
) -> List[Dict[str, Any]]:
    """Traverses DAIC records converting visual modalities into windowed batches efficiently."""
    pt_data = []
    
    if is_train:
        train_visuals = []
        for i in range(len(dataset)):
            try:
                # Need to check 409 handling explicitly
                if dataset[i]['participant_id'] == 409:
                    continue
                train_visuals.append(dataset[i]["visual"])
            except Exception:
                pass
        pipeline.fit(train_visuals)

    skipped_ids = []
    
    # ---------------------------------------------------------
    # PART 1: 409 HANDLING ABLATION FLAG
    # ---------------------------------------------------------
    # Scaffold config flag for experiments: Experiment A (include, default) vs B (exclude 409)
    exclude_ids = [] # e.g. [409]
    
    if 409 not in exclude_ids:
        logger.info("Participant 409 is retained using the official dataset annotation (label source: official phq8_binary, not threshold-derived).")
    
    for i in range(len(dataset)):
        pt = dataset[i]
        pid = pt["participant_id"]
        
        if pid in exclude_ids:
            logger.info(f"Skipping participant {pid} due to exclude_ids config.")
            skipped_ids.append(pid)
            continue
            
        label = pt["labels"]["phq8_binary"]
        
        try:
            tensor = pipeline.transform(pid, pt["visual"])
            w_count = tensor.size(0)
            if w_count == 0:
                skipped_ids.append(pid)
                continue
                
            windows = [tensor[w] for w in range(w_count)]
            
            pt_data.append({
                "pid": pid,
                "label": label,
                "windows": windows
            })
        except Exception as e:
            skipped_ids.append(pid)
            
    logger.info(f"Loaded {len(pt_data)} participants successfully. Skipped {len(skipped_ids)} participants.")
    return pt_data

def run_kfold_cv():
    set_seed(42)
    start_time = time.time()
    
    logger.info("Accessing primary Data bindings...")
    # Load and merge train + dev for CV
    train_dataset = DAICDataset(split="train", load_audio=False, load_text=False)
    dev_dataset = DAICDataset(split="dev", load_audio=False, load_text=False)
    
    config = VisualPreprocessingConfig(window_size=150, stride=75)
    pipeline = VisualPreprocessingPipeline(config)
    
    logger.info("Pushing continuous training sequences...")
    train_pts = extract_participant_sequences(train_dataset, pipeline, is_train=True)
    
    logger.info("Validating independent continuous test matrices...")
    # We do NOT run pipeline.fit on dev, so is_train=False
    dev_pts = extract_participant_sequences(dev_dataset, pipeline, is_train=False)
    
    # Merge for K-Fold
    all_pts = train_pts + dev_pts
    
    pids = [pt["pid"] for pt in all_pts]
    labels = [pt["label"] for pt in all_pts]
    
    # K-Fold Stratified
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    out_dir = Path("outputs/checkpoints/visual")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_log = out_dir / "fold_assignments.json"
    fold_assignments = {}

    cv_results_file = out_dir / "cross_validation_results.json"
    if cv_results_file.exists():
        logger.info("Cross-validation already completed. Remove cross_validation_results.json to restart.")
        return

    # Check for completed folds
    for f in range(1, k_folds + 1):
        metric_file = out_dir / f"fold{f}" / f"fold_{f}_metrics.json"
        if metric_file.exists():
            with open(metric_file, "r") as mf:
                fold_metrics.append(json.load(mf))
                
    if len(fold_metrics) > 0:
        logger.info(f"Detected {len(fold_metrics)} completed folds. Resuming remaining folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(pids, labels), 1):
        if fold <= len(fold_metrics):
            logger.info(f"Skipping Fold {fold}/{k_folds} - already completed.")
            continue
            
        logger.info(f"\n{'='*40}\nStarting Fold {fold}/{k_folds}\n{'='*40}")
        
        train_pids = [pids[i] for i in train_idx]
        val_pids = [pids[i] for i in val_idx]
        fold_assignments[f"Fold_{fold}"] = {"train": train_pids, "val": val_pids}
        
        # Flatten to Windows
        train_data = []
        val_data = []
        
        # ---------------------------------------------------------
        # PART 2: DYNAMIC PARTICIPANT-LEVEL CLASS COUNTS & WEIGHTS
        # ---------------------------------------------------------
        part_0_count = 0
        part_1_count = 0
        
        for idx in train_idx:
            pt = all_pts[idx]
            
            # Participant-level count for weights
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

        # Class imbalance logic dynamically per fold (Participant-level counts)
        if part_0_count > 0 and part_1_count > 0:
            counts = np.array([part_0_count, part_1_count])
            weights = 1.0 / counts
            weights = weights / weights.sum() * 2.0
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            class_weights_tensor = None
            
        logger.info(f"\nFold {fold}")
        logger.info(f"Healthy participants   : {part_0_count}")
        logger.info(f"Depressed participants : {part_1_count}")
        
        weights_str = f"[{weights[0]:.4f}, {weights[1]:.4f}]" if class_weights_tensor is not None else "None"
        logger.info(f"Class weights          : {weights_str}")
        
        train_loader = DataLoader(ProcessedVisualDataset(train_data), batch_size=32, shuffle=True)
        dev_loader = DataLoader(ProcessedVisualDataset(val_data), batch_size=32, shuffle=False)
        
        model = VisualModel()
        trainer = VisualTrainer(model, device="cuda" if torch.cuda.is_available() else "cpu", class_weights=class_weights_tensor)
        
        # Resume epoch if last_model.pt exists for this fold
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "last_model.pt"
        
        start_epoch = 1
        max_epochs = 10
        if checkpoint_path.exists():
            logger.info(f"Resuming Fold {fold} from checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and trainer.scheduler:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            trainer.best_metric = checkpoint.get('best_f1', -1.0)
            trainer.best_accuracy = checkpoint.get('best_accuracy', -1.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            
        if start_epoch <= max_epochs:
            trainer.fit(train_loader, dev_loader, epochs=max_epochs, metric_for_best="f1", fold=fold, seed=42, start_epoch=start_epoch)
        else:
            logger.info(f"Fold {fold} training already finished {max_epochs} epochs.")
        
        # Last validation eval for logging
        logger.info(f"Running final validation for fold {fold}...")
        val_metrics = trainer.validate(dev_loader)
        
        metrics_to_save = {
            "accuracy": float(val_metrics['accuracy']),
            "precision": float(val_metrics['precision']),
            "recall": float(val_metrics['recall']),
            "macro_f1": float(val_metrics['f1']),
            "per_class_f1": val_metrics.get('per_class_f1', []), # assuming missing if not returned
            "confusion_matrix": val_metrics.get('cm_list', val_metrics.get('cm', [])), # may need to map to list
            "epochs": max_epochs,
            "best_epoch": -1 # we might not know exactly unless we track it
        }
        
        # Convert np types to pure python types
        if 'cm' in val_metrics:
            metrics_to_save['confusion_matrix'] = np.array(val_metrics['cm']).tolist()
            
        # load best model to get true final performance
        best_model_path = fold_dir / "best_model.pt"
        if best_model_path.exists():
            best_ckpt = torch.load(best_model_path)
            metrics_to_save['best_epoch'] = best_ckpt.get('epoch', -1)
            
        fold_metrics.append(metrics_to_save)
        
        metric_file = fold_dir / f"fold_{fold}_metrics.json"
        with open(metric_file, "w") as mf:
            json.dump(metrics_to_save, mf, indent=4)
        
    # Write assignments
    with open(folds_log, "w") as f:
        json.dump(fold_assignments, f, indent=4)
        
    logger.info(f"\nFold assignments logged to {folds_log}")

    # Summary Statistics
    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['macro_f1'] for m in fold_metrics]
    precs = [m['precision'] for m in fold_metrics]
    recs = [m['recall'] for m in fold_metrics]
    
    cv_summary = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "mean_precision": float(np.mean(precs)),
        "std_precision": float(np.std(precs)),
        "mean_recall": float(np.mean(recs)),
        "std_recall": float(np.std(recs)),
        "per_fold_metrics": fold_metrics
    }
    
    # Try to aggregate confusion matrix
    try:
        agg_cm = np.zeros((2, 2))
        for m in fold_metrics:
            agg_cm += np.array(m['confusion_matrix'])
        cv_summary["aggregate_confusion_matrix"] = agg_cm.tolist()
    except Exception:
        pass
        
    with open(cv_results_file, "w") as f:
        json.dump(cv_summary, f, indent=4)
        
    logger.info(f"Cross-validation results saved to {cv_results_file}")
    
    logger.info("\n==================================")
    logger.info("FINAL K-FOLD CROSS-VALIDATION RESULTS")
    logger.info("==================================")
    logger.info(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    logger.info(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    
    total_time = (time.time() - start_time) / 60
    logger.info(f"Total Execution Time: {total_time:.2f} minutes")

if __name__ == "__main__":
    run_kfold_cv()
