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
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    folds_log = out_dir / "fold_assignments.json"
    fold_assignments = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(pids, labels), 1):
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
        
        trainer.fit(train_loader, dev_loader, epochs=10, metric_for_best="f1")
        
        # Last validation eval for logging
        logger.info(f"Running final validation for fold {fold}...")
        val_metrics = trainer.validate(dev_loader)
        fold_metrics.append(val_metrics)
        
    # Write assignments
    with open(folds_log, "w") as f:
        json.dump(fold_assignments, f, indent=4)
        
    logger.info(f"\nFold assignments logged to {folds_log}")

    # Summary Statistics
    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    precs = [m['precision'] for m in fold_metrics]
    recs = [m['recall'] for m in fold_metrics]
    
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
