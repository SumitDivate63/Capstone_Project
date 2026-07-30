import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import VisualPreprocessingPipeline, VisualPreprocessingConfig
from models.visual.visual_model import VisualModel
from training.trainer import VisualTrainer

class ProcessedVisualDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def extract_valid_sequences(dataset, pipeline, is_train, max_pts):
    results = []
    visuals = []
    
    pts_to_process = []
    for i in range(min(max_pts, len(dataset))):
        try:
            pt = dataset[i]
            pts_to_process.append(pt)
            if is_train:
                visuals.append(pt["visual"])
        except Exception as e:
            print(f"Skipping participant {e}")
            pass
            
    if is_train and visuals:
        pipeline.fit(visuals)
        
    for pt in pts_to_process:
        try:
            tensor = pipeline.transform(pt["participant_id"], pt["visual"])
            label = pt["labels"]["phq8_binary"]
            for w in range(tensor.size(0)):
                results.append((tensor[w], label))
                if len(results) <= 5: # Print first few labels
                    print(f"Participant ID: {pt['participant_id']}, Window Index: {w}, Assigned Label: {label}")
        except Exception:
            pass
            
    return results

def test_visual_training():
    print("Testing visual training pipeline end-to-end (Diagnostic)...")
    
    # 1. Spin up Datasets efficiently bounding false targets
    train_dataset = DAICDataset(split="train", load_audio=False, load_text=False)
    dev_dataset = DAICDataset(split="dev", load_audio=False, load_text=False)
    
    print(f"Total Train Participants in dataset: {len(train_dataset)}")
    
    config = VisualPreprocessingConfig(window_size=150, stride=150)
    pipeline = VisualPreprocessingPipeline(config)
    
    # 2. Hard cutoff sizes protecting fast validation metrics loops (10 Train, 2 Dev bounds)
    train_data = extract_valid_sequences(train_dataset, pipeline, is_train=True, max_pts=10)
    dev_data = extract_valid_sequences(dev_dataset, pipeline, is_train=False, max_pts=2)
    
    print(f"Total Train Windows: {len(train_data)}")
    print(f"Total Dev Windows: {len(dev_data)}")
    
    train_labels = [item[1] for item in train_data]
    class_0 = train_labels.count(0)
    class_1 = train_labels.count(1)
    class_ratio = class_0 / class_1 if class_1 > 0 else 0
    print(f"Class 0 windows: {class_0}, Class 1 windows: {class_1}")
    print(f"Class ratio (0:1): {class_ratio:.2f}")
    
    if class_0 > 0 and class_1 > 0:
        class_counts = np.array([class_0, class_1])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 2 
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights_tensor = None
    print(f"Computed Class Weights: {class_weights_tensor}")
    
    train_loader = DataLoader(ProcessedVisualDataset(train_data), batch_size=8, shuffle=True)
    dev_loader = DataLoader(ProcessedVisualDataset(dev_data), batch_size=8, shuffle=False)
    
    first_batch = next(iter(train_loader))
    inputs, labels = first_batch
    print("\n--- FIRST TRAINING BATCH INSPECTION ---")
    print(f"Batch size: {labels.size(0)}")
    print(f"Number of class 0: {(labels==0).sum().item()}")
    print(f"Number of class 1: {(labels==1).sum().item()}")
    print(f"Batch tensor shape: {inputs.shape}")
    print(f"Total number of Training Batches: {len(train_loader)}\n")

    model = VisualModel()
    
    trainer = VisualTrainer(model, device="cuda" if torch.cuda.is_available() else "cpu", class_weights=class_weights_tensor)
    
    # 3. Target many validation epochs bounding parameters.
    trainer.fit(train_loader, dev_loader, epochs=15, metric_for_best='accuracy')
    
    # 4. Strict structural checks natively assessing memory constraints
    assert len(train_data) > 0, "Failed to load validation dataset matrices."
    assert len(dev_data) > 0, "Failed isolating continuous tensors correctly."
    
    print("\nVISUAL TRAINING PIPELINE TEST PASSED")

if __name__ == "__main__":
    test_visual_training()
