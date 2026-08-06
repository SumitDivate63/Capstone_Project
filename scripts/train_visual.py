import torch
import time
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import VisualPreprocessingPipeline, VisualPreprocessingConfig
from models.visual.visual_model import VisualModel
from training.trainer import VisualTrainer
from utils.logger import get_logger

logger = get_logger(__name__)

class ProcessedVisualDataset(Dataset):
    """Zero-copy wrapper translating fully transformed window tensors cleanly into standard iterable shapes."""
    def __init__(self, data: List[Tuple[torch.Tensor, int]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def extract_valid_sequences(
    dataset: DAICDataset, 
    pipeline: VisualPreprocessingPipeline, 
    is_train: bool
) -> List[Tuple[torch.Tensor, int]]:
    """Traverses DAIC records converting visual modalities directly into windowed batches efficiently."""
    results = []
    
    if is_train:
        train_visuals = []
        for i in range(len(dataset)):
            try:
                train_visuals.append(dataset[i]["visual"])
            except Exception:
                pass
        pipeline.fit(train_visuals)

    participants_loaded = 0
    participants_skipped = 0
    windows_generated = 0
    healthy_windows = 0
    depressed_windows = 0

    skipped_ids = []
    healthy_participants = 0
    depressed_participants = 0
    seq_lengths = []
    corrupted_csvs = 0
    missing_csvs = 0
    invalid_features = 0
    empty_participants = 0
    
    for i in range(len(dataset)):
        pt = dataset[i]
        pid = pt["participant_id"]
        label = pt["labels"]["phq8_binary"]
        
        try:
            tensor = pipeline.transform(pid, pt["visual"])
            w_count = tensor.size(0)
            if w_count == 0:
                print(f"Participant {pid} skipped\nReason:\nNot enough frames for window generation\n\nRows affected:\n0\n\nColumns:\n0\n\nAction:\nSkipped safely")
                participants_skipped += 1
                skipped_ids.append(pid)
                empty_participants += 1
                continue
                
            windows_generated += w_count
            if label == 0:
                healthy_windows += w_count
                healthy_participants += 1
            else:
                depressed_windows += w_count
                depressed_participants += 1
                
            seq_lengths.append(tensor.size(1))
                
            for w in range(w_count):
                results.append((tensor[w], label))
            participants_loaded += 1
        except Exception as e:
            # We assume errors like corrupted data raised from pipeline are handled here
            err_msg = str(e).lower()
            if "invalid openface" in err_msg or "corrupted" in err_msg:
                invalid_features += 1
                corrupted_csvs += 1
            elif "not found" in err_msg or "missing" in err_msg:
                missing_csvs += 1
            
            participants_skipped += 1
            skipped_ids.append(pid)
            
    print("\n==============================")
    print("DATASET SUMMARY")
    print("==============================")
    print(f"Participants Found\n{len(dataset)}\n")
    print(f"Participants Loaded\n{participants_loaded}\n")
    print(f"Participants Skipped\n{participants_skipped}\n")
    print(f"Skipped IDs\n{skipped_ids}\n")
    print(f"Healthy Participants\n{healthy_participants}\n")
    print(f"Depressed Participants\n{depressed_participants}\n")
    print(f"Total Windows\n{windows_generated}\n")
    print(f"Healthy Windows\n{healthy_windows}\n")
    print(f"Depressed Windows\n{depressed_windows}\n")
    
    avg_windows = windows_generated / participants_loaded if participants_loaded > 0 else 0
    print(f"Average Windows per Participant\n{avg_windows:.2f}\n")
    print(f"Corrupted CSV Files\n{corrupted_csvs}\n")
    print(f"Missing CSV Files\n{missing_csvs}\n")
    print(f"Invalid Feature Files\n{invalid_features}\n")
    print(f"Empty Participants\n{empty_participants}\n")
    
    max_seq = max(seq_lengths) if seq_lengths else 0
    min_seq = min(seq_lengths) if seq_lengths else 0
    print(f"Maximum Sequence Length\n{max_seq}\n")
    print(f"Minimum Sequence Length\n{min_seq}\n")
    
    print(f"Window Size\n{pipeline.config.window_size}\n")
    feature_dim = pipeline.feature_columns
    print(f"Feature Dimension\n{len(feature_dim) if feature_dim else 393}\n")
    print("==============================")
            
    return results


def main():
    start_time = time.time()
    
    logger.info("Accessing primary Data bindings...")
    train_dataset = DAICDataset(split="train", load_audio=False, load_text=False)
    dev_dataset = DAICDataset(split="dev", load_audio=False, load_text=False)
    
    config = VisualPreprocessingConfig(window_size=150, stride=75)
    pipeline = VisualPreprocessingPipeline(config)
    
    logger.info("Pushing continuous training sequences...")
    train_data = extract_valid_sequences(train_dataset, pipeline, is_train=True)
    
    logger.info("Validating independent continuous test matrices...")
    dev_data = extract_valid_sequences(dev_dataset, pipeline, is_train=False)
    
    # Native PyTorch structural batching
    train_loader = DataLoader(ProcessedVisualDataset(train_data), batch_size=32, shuffle=True)
    dev_loader = DataLoader(ProcessedVisualDataset(dev_data), batch_size=32, shuffle=False)
    
    model = VisualModel()
    trainer = VisualTrainer(model)
    
    trainer.fit(train_loader, dev_loader, epochs=10)
    
    duration = (time.time() - start_time) / 60
    logger.info(f"Target execution completed!")
    logger.info(f"Best Validation F1 Outcome: {trainer.best_f1:.4f}")
    logger.info(f"Overall Training Execution Time: {duration:.2f} minutes")

if __name__ == "__main__":
    main()
