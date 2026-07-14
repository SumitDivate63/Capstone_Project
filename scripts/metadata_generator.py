"""
Metadata Generator for the DAIC-WOZ dataset.

This script scans the dataset and generates a comprehensive metadata CSV
and a summary JSON file detailing missing files and dataset statistics.
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
from tqdm import tqdm

# Add the project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.path_config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Required file suffixes for each modality
REQUIRED_FILES = {
    "audio_path": "_AUDIO.wav",
    "transcript_path": "_TRANSCRIPT.csv",
    "covarep_path": "_COVAREP.csv",
    "formant_path": "_FORMANT.csv",
    "au_path": "_CLNF_AUs.txt",
    "pose_path": "_CLNF_pose.txt",
    "gaze_path": "_CLNF_gaze.txt",
    "features_path": "_CLNF_features.txt",
    "features3d_path": "_CLNF_features3D.txt",
    "hog_path": "_CLNF_hog.bin",
}

def generate_metadata() -> None:
    """Scan the dataset and generate metadata records."""
    paths = PathConfig()
    paths.create_directories()

    dataset_root = paths.dataset_root
    logger.info(f"Dataset root configured as: {dataset_root}")

    train_csv = dataset_root / "train_split_Depression_AVEC2017.csv"
    dev_csv = dataset_root / "dev_split_Depression_AVEC2017.csv"

    splits = []
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        train_df['split'] = 'train'
        splits.append(train_df)
    else:
        logger.warning(f"Train targets CSV not found: {train_csv}")

    if dev_csv.exists():
        dev_df = pd.read_csv(dev_csv)
        dev_df['split'] = 'dev'
        splits.append(dev_df)
    else:
        logger.warning(f"Dev targets CSV not found: {dev_csv}")

    if not splits:
        logger.error("No label CSVs found. Cannot generate metadata.")
        return

    combined_df = pd.concat(splits, ignore_index=True)
    
    # Strip any whitespace from column names just in case
    combined_df.columns = combined_df.columns.str.strip()

    required_columns = ["Participant_ID", "PHQ8_Binary", "PHQ8_Score", "Gender", "split"]
    missing_cols = [col for col in required_columns if col not in combined_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in labels: {missing_cols}")
        logger.error(f"Available columns: {list(combined_df.columns)}")
        return

    participants_in_split_files = len(combined_df)
    logger.info(f"Participants in split files: {participants_in_split_files}")

    # Build a set of all actually existing participant folders
    existing_folders = set()
    if dataset_root.exists():
        for d in dataset_root.iterdir():
            if d.is_dir() and d.name.endswith("_P"):
                existing_folders.add(d.name)
                
    participant_folders_found = len(existing_folders)
    logger.info(f"Participant folders found: {participant_folders_found}")

    metadata_records = []
    
    missing_stats = {
        "audio": 0,
        "transcript": 0,
        "covarep": 0,
        "formant": 0,
        "au": 0,
        "pose": 0,
        "gaze": 0,
        "features": 0,
        "features3d": 0,
        "hog": 0
    }

    participants_processed = 0
    participants_skipped_no_folder = 0
    participants_with_missing_modalities = 0
    
    train_subjects = 0
    dev_subjects = 0

    for _, row in tqdm(combined_df.iterrows(), total=participants_in_split_files, desc="Processing Participants"):
        p_id = str(int(row["Participant_ID"]))
        split = row["split"]
        folder_name = f"{p_id}_P"
        
        if folder_name not in existing_folders:
            logger.warning(f"Participant {p_id} listed in split CSV but participant folder not found. Skipping participant.")
            participants_skipped_no_folder += 1
            continue
            
        participants_processed += 1

        if split == "train":
            train_subjects += 1
        else:
            dev_subjects += 1

        participant_dir = dataset_root / folder_name

        record = {
            "participant_id": p_id,
            "split": split,
            "gender": row["Gender"],
            "phq8_binary": row["PHQ8_Binary"],
            "phq8_score": row["PHQ8_Score"]
        }

        has_all_files = True
        
        # Check files using relative paths
        for key, suffix in REQUIRED_FILES.items():
            file_name = f"{p_id}{suffix}"
            expected_file = participant_dir / file_name
            if expected_file.exists():
                record[key] = f"{folder_name}/{file_name}"
            else:
                record[key] = None
                has_all_files = False
                stat_key = key.replace("_path", "")
                missing_stats[stat_key] += 1
        
        if not has_all_files:
            participants_with_missing_modalities += 1
            
        record["has_all_files"] = has_all_files
        metadata_records.append(record)

    logger.info(f"Participants processed: {participants_processed}")
    logger.info(f"Participants skipped: {participants_skipped_no_folder}")

    # Save to CSV
    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv_path = paths.metadata_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    logger.info(f"Metadata CSV saved to: {metadata_csv_path}")

    # Generate JSON summary
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root.absolute()),
        "participants_in_split_files": participants_in_split_files,
        "participant_folders_found": participant_folders_found,
        "participants_processed": participants_processed,
        "participants_skipped_no_folder": participants_skipped_no_folder,
        "participants_with_missing_modalities": participants_with_missing_modalities,
        "train_subjects": train_subjects,
        "dev_subjects": dev_subjects,
        "missing_audio": missing_stats["audio"],
        "missing_transcript": missing_stats["transcript"],
        "missing_covarep": missing_stats["covarep"],
        "missing_formant": missing_stats["formant"],
        "missing_au": missing_stats["au"],
        "missing_pose": missing_stats["pose"],
        "missing_gaze": missing_stats["gaze"],
        "missing_features": missing_stats["features"],
        "missing_features3d": missing_stats["features3d"],
        "missing_hog": missing_stats["hog"]
    }
    
    summary_json_path = paths.metadata_dir / "metadata_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Metadata summary JSON saved to: {summary_json_path}")
    logger.info("Metadata generation completed successfully.")

def main() -> None:
    """Main entry point for generating metadata."""
    generate_metadata()

if __name__ == "__main__":
    main()
