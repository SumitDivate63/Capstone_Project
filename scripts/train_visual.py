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

    for i in range(len(dataset)):
        try:
            pt = dataset[i]
            tensor = pipeline.transform(pt["participant_id"], pt["visual"])
            label = pt["labels"]["phq8_binary"]
            
            w_count = tensor.size(0)
            if w_count == 0:
                print(f"Participant {pt['participant_id']} skipped:\nReason: Not enough frames for window generation.")
                participants_skipped += 1
                continue
                
            windows_generated += w_count
            if label == 0:
                healthy_windows += w_count
            else:
                depressed_windows += w_count
                
            # Expand each sequence window block natively linking tracking label
            for w in range(w_count):
                results.append((tensor[w], label))
            participants_loaded += 1
        except Exception as e:
            # We assume errors like corrupted data raised from pipeline are handled here
            # but our cleaned pipeline already printed specific format, we just count skipped here
            logger.warning(f"Participant bounds skipped strictly during matrix translation [ID: {dataset[i]['participant_id']}] -> {e}")
            participants_skipped += 1
            
    print("Participants loaded:", participants_loaded)
    print("Participants skipped:", participants_skipped)
    print("Windows generated:", windows_generated)
    print("Healthy windows:", healthy_windows)
    print("Depressed windows:", depressed_windows)
            
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
