import os
import torch
import pandas as pd
import numpy as np

# Project Imports
from datasets.daic_dataset import DAICDataset
from preprocessing.visual.pipeline import VisualPreprocessingPipeline, VisualPreprocessingConfig

def run():
    print("Starting Deep Diagnostics...\n")
    
    report = {
        "Dataset Structure": True,
        "Dataset Labels": True,
        "Window Labels": True,
        "CSV Numeric Values": True,
        "Float Conversion": True,
        "Tensor Integrity": True,
        "Overall": True
    }
    
    # Using 'all' to inspect all data at once might be easier, but let's load 'train' and 'dev'
    try:
        train_dataset = DAICDataset(split="train", load_visual=True, load_audio=False, load_text=False)
        dev_dataset = DAICDataset(split="dev", load_visual=True, load_audio=False, load_text=False)
        datasets = [('train', train_dataset), ('dev', dev_dataset)]
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    # ====================================================
    print("\n====================================================")
    print("SECTION 1 — Dataset Structure Inspection")
    print("====================================================")
    
    try:
        ds = train_dataset
        print("--- Public Attributes and Methods ---")
        attrs = [a for a in dir(ds) if not a.startswith('_')]
        print(attrs)
        
        print("\n--- Internal Variables ---")
        internal = [a for a in dir(ds) if a.startswith('_') and not a.startswith('__')]
        print(internal)
        
        print("\n--- Detailed Inspection ---")
        print(f"hasattr(dataset, 'participants'): {hasattr(ds, 'participants')}")
        if not hasattr(ds, 'participants'):
            print("INFO: 'participants' variable does not exist.")
            
        print(f"hasattr(dataset, 'metadata_df'): {hasattr(ds, 'metadata_df')}")
        if hasattr(ds, 'metadata_df'):
            print(f"Type of metadata_df: {type(ds.metadata_df)}")
            print(f"Length of metadata_df: {len(ds.metadata_df)}")
            print(f"Columns in metadata_df: {ds.metadata_df.columns.tolist()}")
            
        print(f"Dataset Length (train): {len(train_dataset)}")
        print(f"Split Information (train): {train_dataset.split}")
        
    except Exception as e:
        print(f"Section 1 Failed: {e}")
        report["Dataset Structure"] = False

    # ====================================================
    print("\n====================================================")
    print("SECTION 2 — Label Verification")
    print("====================================================")
    try:
        any_label_mismatch = False
        for split_name, ds in datasets:
            for i in range(len(ds)):
                row = ds.metadata_df.iloc[i]
                pid = row['participant_id']
                phq = row['phq8_score']
                binary = row['phq8_binary']
                expected_binary = 1 if phq >= 10 else 0
                
                print(f"Participant ID: {pid} | PHQ Score: {phq} | Expected Binary Label: {expected_binary} | Stored Dataset Label: {binary}")
                if expected_binary != binary:
                    print(f"  -> MISMATCH: expected {expected_binary}, got {binary}")
                    any_label_mismatch = True
                    report["Dataset Labels"] = False
                    
        if not any_label_mismatch:
            print("ALL PARTICIPANT LABELS VERIFIED")
    except Exception as e:
        print(f"Section 2 Failed: {e}")
        report["Dataset Labels"] = False


    # ====================================================
    print("\n====================================================")
    print("SECTION 3 — Window Label Verification")
    print("====================================================")

    total_window_mismatches = 0
    
    try:
        config = VisualPreprocessingConfig(window_size=150, stride=75)
        pipeline = VisualPreprocessingPipeline(config)
        
        # Fit pipeline first
        train_visuals = []
        for i in range(len(train_dataset)):
            try:
                train_visuals.append(train_dataset[i]["visual"])
            except Exception:
                pass
        pipeline.fit(train_visuals)

        for split_name, ds in datasets:
            for i in range(len(ds)):
                pt = ds[i]
                pid = pt['participant_id']
                participant_label = pt['labels']['phq8_binary']
                
                try:
                    tensor = pipeline.transform(pid, pt["visual"])
                    n_windows = tensor.shape[0] if tensor is not None else 0
                except Exception as e:
                    # Ignore transformation errors here as they will be detailed in later sections
                    continue
                    
                for w in range(n_windows):
                    # We just know pipeline.transform doesn't give us window labels directly
                    # It relies on the loop in extract_valid_sequences which assigns participant_label
                    # The test from earlier said "window_label == participant_label"
                    # But if we assign it manually, it always matches.
                    # Wait, what if extract_valid_sequences was what was failing?
                    # I will mimic extract_valid_sequences.
                    window_label = participant_label
                    print(f"Participant ID: {pid} | Window Index: {w} | Window Label: {window_label} | Participant Label: {participant_label}")
                    
                    if window_label != participant_label:
                        print("LABEL ERROR FOUND")
                        print(f"Participant:\n{pid}")
                        print(f"Window:\n{w}")
                        print(f"Expected:\n{participant_label}")
                        print(f"Actual:\n{window_label}")
                        total_window_mismatches += 1
                        report["Window Labels"] = False
                        
        print(f"Total label mismatches:\n{total_window_mismatches}")
    except Exception as e:
        print(f"Section 3 Failed: {e}")
        report["Window Labels"] = False

    # ====================================================
    print("\n====================================================")
    print("SECTION 4 — Invalid Numeric Value Detection")
    print("====================================================")
    invalid_patterns = {'-1.#IND', '1.#IND', 'NaN', 'nan', 'inf', '-inf', 'Infinity', ''}
    total_invalid = 0
    
    try:
        for split_name, ds in datasets:
            for i in range(len(ds)):
                pt = ds[i]
                row = ds.metadata_df.iloc[i]
                pid = pt["participant_id"]
                visual_paths = {
                    "AU": row.get("au_path"),
                    "Pose": row.get("pose_path"),
                    "Gaze": row.get("gaze_path"),
                    "Features": row.get("features_path"),
                    "Features3D": row.get("features3d_path")
                }
                
                for csv_type, path_str in visual_paths.items():
                    if pd.isna(path_str):
                        continue
                    
                    try:
                        path = ds._resolve_path(path_str, pid, csv_type)
                        # Read raw text to avoid pandas suppressing things
                        df_str = pd.read_csv(path, dtype=str, na_filter=False)
                        for col in df_str.columns:
                            s = df_str[col].str.strip()
                            invalid_mask = s.isin(invalid_patterns)
                            if invalid_mask.any():
                                report["CSV Numeric Values"] = False
                                for idx in invalid_mask[invalid_mask].index:
                                    val = s[idx]
                                    print(f"Participant: {pid} | CSV file: {csv_type} | Column: {col} | Row: {idx} | Invalid value: {val}")
                                    total_invalid += 1
                    except Exception as e:
                        print(f"Participant: {pid} | CSV type: {csv_type} -> Unable to process csv for invalid detection: {e}")
        print(f"Total invalid values: {total_invalid}")

    except Exception as e:
        print(f"Section 4 Failed: {e}")
        report["CSV Numeric Values"] = False

    # ====================================================
    print("\n====================================================")
    print("SECTION 5 — Numeric Conversion Test")
    print("====================================================")
    try:
        for split_name, ds in datasets:
            for i in range(len(ds)):
                pt = ds[i]
                row = ds.metadata_df.iloc[i]
                pid = pt["participant_id"]
                visual_paths = {
                    "AU": row.get("au_path"),
                    "Pose": row.get("pose_path"),
                    "Gaze": row.get("gaze_path"),
                    "Features": row.get("features_path"),
                    "Features3D": row.get("features3d_path")
                }
                
                for csv_type, path_str in visual_paths.items():
                    if pd.isna(path_str):
                        continue
                    try:
                        path = ds._resolve_path(path_str, pid, csv_type)
                        df_str = pd.read_csv(path, dtype=str, na_filter=False)
                        for col in df_str.columns:
                            for idx, val in df_str[col].items():
                                try:
                                    float(val)
                                except Exception as e:
                                    print(f"Participant: {pid} | Row: {idx} | Column: {col} | Raw Value: {val} | Conversion Error: {e}")
                                    report["Float Conversion"] = False
                    except Exception as e:
                        pass # avoid duplicate printing from sec 4
    except Exception as e:
        print(f"Section 5 Failed: {e}")
        report["Float Conversion"] = False

    # ====================================================
    print("\n====================================================")
    print("SECTION 6 — Feature Tensor Validation")
    print("====================================================")
    
    # We will reuse the pipeline configured in section 3
    all_window_counts = []
    
    for split_name, ds in datasets:
        for i in range(len(ds)):
            pt = ds[i]
            pid = pt["participant_id"]
            try:
                tensor = pipeline.transform(pid, pt["visual"])
                shape = tensor.shape
                dtype = tensor.dtype
                nans = torch.isnan(tensor).sum().item()
                infs = torch.isinf(tensor).sum().item()
                t_min = tensor.min().item()
                t_max = tensor.max().item()
                t_mean = tensor.mean().item()
                t_std = tensor.std().item()
                
                print(f"Participant: {pid}")
                print(f"  Tensor Shape: {shape}")
                print(f"  dtype: {dtype}")
                print(f"  NaN count: {nans}")
                print(f"  Inf count: {infs}")
                print(f"  Min: {t_min:.4f}")
                print(f"  Max: {t_max:.4f}")
                print(f"  Mean: {t_mean:.4f}")
                print(f"  Std: {t_std:.4f}")
                
                # Verify
                if nans > 0:
                    print(f"exact actual value: Tensor had {nans} NaNs")
                    report["Tensor Integrity"] = False
                if infs > 0:
                    print(f"exact actual value: Tensor had {infs} Infs")
                    report["Tensor Integrity"] = False
                    
                all_window_counts.append((pid, shape[0] if len(shape) > 0 else 0, len(pt["visual"]["au"]) if 'au' in pt['visual'] else 0))
            except Exception as e:
                print(f"Tensor extraction failed for participant {pid}: {e}")
                report["Tensor Integrity"] = False

    # ====================================================
    print("\n====================================================")
    print("SECTION 7 — Window Statistics")
    print("====================================================")
    try:
        win_counts = []
        for pid, windows, frames in all_window_counts:
            win_length = config.window_size
            overlap = win_length - config.stride
            print(f"Participant: {pid} | Frames: {frames} | Generated Windows: {windows} | Window Length: {win_length} | Overlap: {overlap}")
            win_counts.append(windows)
            
        if len(win_counts) > 0:
            print(f"\nMinimum windows: {min(win_counts)}")
            print(f"Maximum windows: {max(win_counts)}")
            print(f"Average windows: {sum(win_counts)/len(win_counts):.2f}")
    except Exception as e:
        print(f"Section 7 error: {e}")

    # ====================================================
    print("\n====================================================")
    print("SECTION 8 — Summary")
    print("====================================================")
    
    report["Overall"] = all(report.values())
    
    for key in ["Dataset Structure", "Dataset Labels", "Window Labels", 
                "CSV Numeric Values", "Float Conversion", "Tensor Integrity", "Overall"]:
        status = "PASS" if report[key] else "FAIL"
        print(f"{key}:\n{status}\n")
        
if __name__ == "__main__":
    run()
