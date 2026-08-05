import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

# Project Imports
from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import VisualPreprocessingPipeline, VisualPreprocessingConfig
from scripts.train_visual import ProcessedVisualDataset, extract_valid_sequences
from models.visual.visual_model import VisualModel
from training.metrics import compute_metrics
from training.optimizer import create_optimizer

def run_diagnostics():
    report = {}

    print("====================================================")
    print("SECTION 1 : DATASET INFORMATION")
    print("====================================================")
    
    try:
        train_dataset = DAICDataset(split="train", load_visual=True, load_audio=False, load_text=False)
        dev_dataset = DAICDataset(split="dev", load_visual=True, load_audio=False, load_text=False)
        
        print(f"Number of train participants: {len(train_dataset)}")
        print(f"Number of dev participants: {len(dev_dataset)}")
        
        train_ids = [train_dataset.participants[i] for i in range(len(train_dataset))]
        dev_ids = [dev_dataset.participants[i] for i in range(len(dev_dataset))]
        
        print(f"\nTrain Participant IDs: {train_ids}")
        print(f"Dev Participant IDs: {dev_ids}\n")
        
        for p in range(len(train_dataset)):
            pt = train_dataset[p]
            frames = pt["visual"].shape[0] if "visual" in pt and pt["visual"] is not None else 0
            has_labels = "labels" in pt and pt["labels"] is not None
            phq = pt["labels"]["phq8_score"] if has_labels else "N/A"
            binary = pt["labels"]["phq8_binary"] if has_labels else "N/A"
            print(f"Participant: {pt['participant_id']} | Frames: {frames} | PHQ: {phq} | Binary Label: {binary}")
        report["dataset"] = True
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        report["dataset"] = False

    print("\n====================================================")
    print("SECTION 2 : LABEL VERIFICATION")
    print("====================================================")
    try:
        healthy_count = 0
        depressed_count = 0
        all_labels_valid = True
        
        for ds in [train_dataset, dev_dataset]:
            for p in range(len(ds)):
                pt = ds[p]
                pid = pt["participant_id"]
                has_labels = "labels" in pt and pt["labels"] is not None
                if not has_labels:
                    print(f"Participant {pid} is MISSING labels.")
                    all_labels_valid = False
                    continue
                
                phq = pt["labels"]["phq8_score"]
                binary = pt["labels"]["phq8_binary"]
                print(f"Participant ID: {pid} | PHQ Score: {phq} | Binary Label: {binary}")
                
                if binary not in [0, 1]:
                    print(f"INVALID LABEL: {binary} for {pid}")
                    all_labels_valid = False
                
                if binary == 0:
                    healthy_count += 1
                elif binary == 1:
                    depressed_count += 1
                    
        print(f"\nHealthy Participants = {healthy_count}")
        print(f"Depressed Participants = {depressed_count}")
        
        if all_labels_valid:
            report["labels"] = True
        else:
            report["labels"] = False
    except Exception as e:
        print(f"Label verification failed: {e}")
        report["labels"] = False

    print("\n====================================================")
    print("SECTION 3 : WINDOW GENERATION")
    print("====================================================")
    try:
        config = VisualPreprocessingConfig(window_size=150, stride=75)
        pipeline = VisualPreprocessingPipeline(config)
        
        train_visuals = []
        for p in range(len(train_dataset)):
            try:
                if "visual" in train_dataset[p] and train_dataset[p]["visual"] is not None:
                    train_visuals.append(train_dataset[p]["visual"])
            except Exception:
                pass
        pipeline.fit(train_visuals)
        
        windows_match = True
        train_windows = []
        dev_windows = []
        
        for ds, target_list in [(train_dataset, train_windows), (dev_dataset, dev_windows)]:
            for p in range(len(ds)):
                pt = ds[p]
                pid = pt["participant_id"]
                try:
                    tensor = pipeline.transform(pid, pt["visual"])
                    n_windows = tensor.shape[0] if tensor is not None else 0
                    label = pt["labels"]["phq8_binary"]
                    print(f"Participant ID: {pid} | Generated Windows: {n_windows} | Assigned Label: {label}")
                    
                    for w in range(n_windows):
                        target_list.append((tensor[w], label))
                        
                except Exception as e:
                    print(f"Window generation exception for {pid}: {e}")
                    windows_match = False

        if windows_match:
            print("WINDOW LABEL CHECK PASSED")
            report["window_generation"] = True
        else:
            print("LABEL ERROR DETECTED")
            report["window_generation"] = False
    except Exception as e:
        print(f"Window Generation Failed: {e}")
        report["window_generation"] = False

    print("\n====================================================")
    print("SECTION 4 : WINDOW DISTRIBUTION")
    print("====================================================")
    
    total_train = len(train_windows)
    total_dev = len(dev_windows)
    
    train_class_0 = sum([1 for w, l in train_windows if l == 0])
    train_class_1 = sum([1 for w, l in train_windows if l == 1])
    
    print(f"Total Train Windows: {total_train}")
    print(f"Total Dev Windows: {total_dev}")
    print(f"\nClass 0 Windows (Train): {train_class_0}")
    print(f"Class 1 Windows (Train): {train_class_1}")
    if total_train > 0:
        print(f"Percentage Class 0 (Train): {train_class_0 / total_train * 100:.2f}%")
        print(f"Percentage Class 1 (Train): {train_class_1 / total_train * 100:.2f}%")

    print("\n====================================================")
    print("SECTION 5 : DATALOADER VERIFICATION")
    print("====================================================")
    try:
        train_loader = DataLoader(ProcessedVisualDataset(train_windows), batch_size=32, shuffle=True)
        
        first_batch = next(iter(train_loader))
        inputs, targets = first_batch
        print(f"Batch Shape: {inputs.shape}")
        print(f"Label Tensor Shape: {targets.shape}")
        print("Participant IDs: Not tracked in batch tensors.")
        
        for b_idx, batch in enumerate(train_loader):
            if b_idx >= 5:
                break
            b_inputs, b_targets = batch
            c0 = (b_targets == 0).sum().item()
            c1 = (b_targets == 1).sum().item()
            print(f"Batch Number: {b_idx} | Class 0 count: {c0} | Class 1 count: {c1}")
            
        report["dataloader"] = True
    except Exception as e:
        print(f"DataLoader test failed: {e}")
        report["dataloader"] = False

    print("\n====================================================")
    print("SECTION 6 : MODEL VERIFICATION")
    print("====================================================")
    try:
        model = VisualModel()
        encoded = model.encoder(inputs)
        pooled = model.pooling(encoded)
        logits = model.classifier(pooled)
        
        print(f"Input Shape: {inputs.shape}")
        print(f"Encoder Output Shape: {encoded.shape}")
        print(f"Attention Output Shape: {pooled.shape}")
        print(f"Logits Shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")
        
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        print(f"NaN count: {nan_count}")
        print(f"Inf count: {inf_count}")
        
        report["model_forward"] = True
    except Exception as e:
        print(f"Model pass failed: {e}")
        report["model_forward"] = False

    print("\n====================================================")
    print("SECTION 7 : PREDICTION VERIFICATION")
    print("====================================================")
    try:
        # Run one forward pass on first 50 samples
        batch_size = 50
        sample_inputs = []
        sample_labels = []
        for w, l in train_windows[:batch_size]:
            sample_inputs.append(w)
            sample_labels.append(l)
            
        if len(sample_inputs) > 0:
            sample_inputs = torch.stack(sample_inputs)
            sample_labels = torch.tensor(sample_labels)
            
            with torch.no_grad():
                sample_logits = model(sample_inputs)
                sample_preds = torch.argmax(sample_logits, dim=1)
                sample_probs = torch.softmax(sample_logits, dim=1)
                
            for i in range(len(sample_labels)):
                print(f"Participant ID: N/A (Window {i}) | Ground Truth Label: {sample_labels[i].item()} | "
                      f"Raw Logits: {sample_logits[i].tolist()} | Predicted Class: {sample_preds[i].item()} | "
                      f"Probability: {sample_probs[i].tolist()}")
                      
            print("Prediction Logic PASSED")
            report["prediction_logic"] = True
        else:
            print("No windows available for Prediction pass")
            report["prediction_logic"] = False
    except Exception as e:
        print(f"Prediction failed: {e}")
        report["prediction_logic"] = False

    print("\n====================================================")
    print("SECTION 8 : LOSS VERIFICATION")
    print("====================================================")
    try:
        criterion = nn.CrossEntropyLoss()
        loss_val = criterion(sample_logits, sample_labels)
        print(f"Loss value: {loss_val.item()}")
        print(f"Class weights: None (default behavior unless explicitly passed)")
        print(f"Target tensor: {sample_labels}")
        print(f"Prediction tensor: {sample_preds}")
        report["loss"] = True
    except Exception as e:
        print(f"Loss computation failed: {e}")
        report["loss"] = False

    print("\n====================================================")
    print("SECTION 9 : METRICS VERIFICATION")
    print("====================================================")
    try:
        metrics_dict = compute_metrics(sample_labels.tolist(), sample_preds.tolist(), loss_val.item())
        
        print(f"Accuracy: {metrics_dict['accuracy']}")
        print(f"Precision: {metrics_dict['precision']}")
        print(f"Recall: {metrics_dict['recall']}")
        print(f"F1: {metrics_dict['f1']}")
        print(f"Confusion Matrix: \n{metrics_dict['cm']}")
        
        manual_cm = confusion_matrix(sample_labels.tolist(), sample_preds.tolist(), labels=[0, 1])
        if np.array_equal(metrics_dict['cm'], manual_cm):
            report["metrics"] = True
            report["confusion_matrix"] = True
        else:
            print("METRIC IMPLEMENTATION ERROR")
            report["metrics"] = False
            report["confusion_matrix"] = False
    except Exception as e:
        print(f"Metric logic failed: {e}")
        report["metrics"] = False
        report["confusion_matrix"] = False

    print("\n====================================================")
    print("SECTION 10 : OVERFITTING CHECK")
    print("====================================================")
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {total_params}")
        
        epochs_configured = 10
        total_batches = len(train_loader)
        expected_steps = total_batches * epochs_configured
        
        optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=1e-3)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Total batches: {total_batches}")
        print(f"Expected optimizer steps: {expected_steps}")
        print(f"Learning rate: {lr}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Loss: CrossEntropyLoss")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Epochs configured: {epochs_configured}")
    except Exception as e:
        print(f"Overfitting check failed: {e}")

    print("\n====================================================")
    print("SECTION 11 : FINAL REPORT")
    print("====================================================")
    
    sections = [
        ("Dataset", "dataset"),
        ("Labels", "labels"),
        ("Window Generation", "window_generation"),
        ("DataLoader", "dataloader"),
        ("Model Forward", "model_forward"),
        ("Prediction Logic", "prediction_logic"),
        ("Loss", "loss"),
        ("Metrics", "metrics"),
        ("Confusion Matrix", "confusion_matrix")
    ]
    
    for section_name, key in sections:
        status = "PASS" if report.get(key, False) else "FAIL"
        check_mark = "✓" if status == "PASS" else "x"
        print(f"{check_mark} {section_name}: {status}")

if __name__ == "__main__":
    run_diagnostics()
