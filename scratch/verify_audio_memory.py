import os, gc, sys
import json
import logging
import torch
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Ensure paths correctly resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets.daic_dataset import DAICDataset
from preprocessing.audio.pipeline import AudioPreprocessingPipeline, AudioPreprocessingConfig
from models.audio_agent.audio_model import AudioModel
from training.audio_trainer import AudioTrainer

def get_mem_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0.0

def run_mini_verification():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mini_verify")
    
    logger.info(f"Start mem: {get_mem_mb():.0f} MB")
    
    # 1. Load lightweight metadata
    logger.info("Loading lightweight dataset for metadata...")
    meta_dataset = DAICDataset(split="train", load_visual=False, load_audio=False, load_text=False)
    
    # Take first 8 participants
    pids = []
    labels = []
    for i in range(8):
        try:
            pt = meta_dataset[i]
            pids.append(pt["participant_id"])
            labels.append(pt["labels"]["phq8_binary"])
        except:
            break
            
    logger.info(f"Selected {len(pids)} participants: {pids}")
    
    # 2. Fold setup
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    fold_list = list(skf.split(pids, labels))
    
    # Needs a dataset mapping to load actual audio data on demand
    audio_dataset = DAICDataset(split="train", load_visual=False, load_audio=True, load_text=False)
    pid_to_idx = {int(row["participant_id"]): i for i, row in audio_dataset.metadata_df.iterrows()}
    
    for fold, (train_idx, val_idx) in enumerate(fold_list, 1):
        logger.info(f"\n{'='*50}\n  FOLD {fold} MINIMAL TEST\n{'='*50}")
        
        train_pids = [pids[i] for i in train_idx]
        val_pids = [pids[i] for i in val_idx]
        logger.info(f"Train pids ({len(train_pids)}): {train_pids}")
        logger.info(f"Val pids ({len(val_pids)}): {val_pids}")
        assert not set(train_pids).intersection(set(val_pids)), "Overlap between train and val pids!"
        
        config = AudioPreprocessingConfig(window_size=150, stride=30)
        pipeline = AudioPreprocessingPipeline(config)
        
        # Fit scaler
        logger.info(f"Mem before fit: {get_mem_mb():.0f} MB")
        train_audios = []
        for pid in train_pids:
            idx = pid_to_idx.get(pid)
            if idx is not None:
                train_audios.append(audio_dataset[idx]["audio"])
        logger.info(f"Fitting scaler on {len(train_audios)} participants...")
        pipeline.fit(train_audios)
        del train_audios
        logger.info(f"Mem after fit: {get_mem_mb():.0f} MB")
        
        # Transform train
        train_data = []
        for pid in train_pids:
            idx = pid_to_idx.get(pid)
            if idx is not None:
                pt = audio_dataset[idx]
                label = pt["labels"]["phq8_binary"]
                tensor = pipeline.transform(pid, pt["audio"])
                for w in range(tensor.size(0)):
                    train_data.append((tensor[w], label, pid))
                
        logger.info(f"Train windows generated: {len(train_data)}")
        
        # Transform val
        val_data = []
        for pid in val_pids:
            idx = pid_to_idx.get(pid)
            if idx is not None:
                pt = audio_dataset[idx]
                label = pt["labels"]["phq8_binary"]
                tensor = pipeline.transform(pid, pt["audio"])
                for w in range(tensor.size(0)):
                    val_data.append((tensor[w], label, pid))
                    
        logger.info(f"Val windows generated: {len(val_data)}")
                
        logger.info(f"Mem after window gen: {get_mem_mb():.0f} MB")
        
        # Dummy data loaders & train
        from torch.utils.data import DataLoader, Dataset
        class DummyDset(Dataset):
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i]
            
        train_loader = DataLoader(DummyDset(train_data), batch_size=4)
        val_loader = DataLoader(DummyDset(val_data), batch_size=4)
        
        model = AudioModel(input_dim=train_data[0][0].shape[-1], d_model=32)
        
        # Using CPU to avoid missing packages issues in the verification if CUDA is not clean
        trainer = AudioTrainer(model=model, device="cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Mem before train: {get_mem_mb():.0f} MB")
        trainer.fit(
            train_loader, 
            val_loader, 
            epochs=1, 
            fold=fold, 
            save_dir=f"outputs/checkpoints/audio_mini/fold{fold}", 
            log_fn=lambda x: None
        )
        logger.info(f"Mem after train: {get_mem_mb():.0f} MB")
        
        # Cleanup
        del train_data
        del val_data
        del train_loader
        del val_loader
        del trainer
        del model
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"Mem after cleanup: {get_mem_mb():.0f} MB")

if __name__ == '__main__':
    run_mini_verification()
