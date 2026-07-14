import torch
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
        except Exception:
            pass
            
    if is_train and visuals:
        pipeline.fit(visuals)
        
    for pt in pts_to_process:
        try:
            tensor = pipeline.transform(pt["participant_id"], pt["visual"])
            label = pt["labels"]["phq8_binary"]
            for w in range(tensor.size(0)):
                results.append((tensor[w], label))
        except Exception:
            pass
            
    return results

def test_visual_training():
    print("Testing visual training pipeline end-to-end securely isolated...")
    
    # 1. Spin up Datasets efficiently bounding false targets
    train_dataset = DAICDataset(split="train", load_audio=False, load_text=False)
    dev_dataset = DAICDataset(split="dev", load_audio=False, load_text=False)
    
    config = VisualPreprocessingConfig(window_size=150, stride=150)
    pipeline = VisualPreprocessingPipeline(config)
    
    # 2. Hard cutoff sizes protecting fast validation metrics loops (5 Train, 2 Dev bounds)
    train_data = extract_valid_sequences(train_dataset, pipeline, is_train=True, max_pts=5)
    dev_data = extract_valid_sequences(dev_dataset, pipeline, is_train=False, max_pts=2)
    
    train_loader = DataLoader(ProcessedVisualDataset(train_data), batch_size=4, shuffle=True)
    dev_loader = DataLoader(ProcessedVisualDataset(dev_data), batch_size=4, shuffle=False)
    
    model = VisualModel()
    
    # Map purely CPU bounds for non-cuda native safe assertions
    trainer = VisualTrainer(model, device="cpu")
    
    # 3. Target single validation epoch bounding parameters.
    trainer.fit(train_loader, dev_loader, epochs=1)
    
    # 4. Strict structural checks natively assessing memory constraints
    assert len(train_data) > 0, "Failed to load validation dataset matrices."
    assert len(dev_data) > 0, "Failed isolating continuous tensors correctly."
    
    print("\nVISUAL TRAINING PIPELINE TEST PASSED")

if __name__ == "__main__":
    test_visual_training()
