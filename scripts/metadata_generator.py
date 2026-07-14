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
    # Check for missing columns (case-insensitive where possible, but we'll exact match first)
    missing_cols = [col for col in required_columns if col not in combined_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in labels: {missing_cols}")
        logger.error(f"Available columns: {list(combined_df.columns)}")
        return

    total_subjects = len(combined_df)
    logger.info(f"Total participants found in splits: {total_subjects}")

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

    train_subjects = 0
    dev_subjects = 0

    for _, row in tqdm(combined_df.iterrows(), total=total_subjects, desc="Processing Participants"):
        p_id = str(int(row["Participant_ID"]))
        split = row["split"]
        
        if split == "train":
            train_subjects += 1
        else:
            dev_subjects += 1

        participant_dir = dataset_root / f"{p_id}_P"

        record = {
            "participant_id": p_id,
            "split": split,
            "gender": row["Gender"],
            "phq8_binary": row["PHQ8_Binary"],
            "phq8_score": row["PHQ8_Score"]
        }

        has_all_files = True
        
        # Check files
        for key, suffix in REQUIRED_FILES.items():
            expected_file = participant_dir / f"{p_id}{suffix}"
            if expected_file.exists():
                # Store absolute path
                record[key] = str(expected_file.absolute())
            else:
                record[key] = None
                has_all_files = False
                # Increment missing stat based on key (strip "_path")
                stat_key = key.replace("_path", "")
                missing_stats[stat_key] += 1
                logger.warning(f"Participant {p_id} missing file: {expected_file.name}")
        
        record["has_all_files"] = has_all_files
        metadata_records.append(record)

    # Save to CSV
    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv_path = paths.metadata_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    logger.info(f"Metadata CSV saved to: {metadata_csv_path}")

    # Generate JSON summary
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root.absolute()),
        "total_subjects": total_subjects,
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
    logger.info("Metadata generation completed.")

def main() -> None:
    """Main entry point for generating metadata."""
    generate_metadata()

if __name__ == "__main__":
    main()
