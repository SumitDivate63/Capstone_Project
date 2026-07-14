"""
Dataset Validator for the DAIC-WOZ dataset.

This module validates the dataset structure and data completeness based on
the generated metadata without altering or preprocessing any underlying data.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.path_config import PathConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Production-quality dataset validator for Multimodal Depression Detection."""

    def __init__(self):
        self.paths = PathConfig()
        self.paths.create_directories()
        
        self.report_dir = self.paths.output_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_root = self.paths.dataset_root
        
        metadata_csv_path = self.paths.metadata_dir / "metadata.csv"
        if not metadata_csv_path.exists():
            logger.error(f"Metadata file not found: {metadata_csv_path}")
            raise FileNotFoundError(f"Missing {metadata_csv_path}")
        
        self.metadata_df = pd.read_csv(metadata_csv_path)
        logger.info(f"Loaded metadata from {metadata_csv_path} with {len(self.metadata_df)} records.")
        
        self.participants_checked = len(self.metadata_df)
        
        self.failed_participants: Set[str] = set()
        self.warning_participants: Set[str] = set()
        
        self.category_fails = {
            "Metadata": set(),
            "File Integrity": set(),
            "Transcript": set(),
            "Audio": set(),
            "Visual": set(),
            "Cross-Modality": set()
        }
        
        self.issues: List[Dict[str, Any]] = []
        
    def _add_issue(self, category: str, severity: str, metric_category: str, message: str, p_id: Optional[str] = None):
        """Helper to record issues systematically based on severity."""
        self.issues.append({
            "category": category,
            "severity": severity,
            "participant_id": p_id,
            "message": message
        })
        if p_id:
            if severity == "FAIL":
                self.failed_participants.add(p_id)
                self.category_fails[metric_category].add(p_id)
            elif severity == "WARNING":
                self.warning_participants.add(p_id)
            
    def validate_metadata(self):
        """Stage 1: Metadata Validation"""
        logger.info("Executing Stage 1: Metadata Validation")
        df = self.metadata_df
        
        duplicates = df[df.duplicated()]
        if not duplicates.empty:
            msg = f"Found {len(duplicates)} duplicate full rows in metadata."
            self._add_issue("duplicate_rows", "FAIL", "Metadata", msg)
            
        if "participant_id" in df.columns:
            id_counts = df["participant_id"].value_counts()
            dupe_ids = id_counts[id_counts > 1].index.tolist()
            for pid in dupe_ids:
                self._add_issue("duplicate_ids", "FAIL", "Metadata", f"Participant ID {pid} is duplicated.", str(pid))
        else:
            self._add_issue("errors", "FAIL", "Metadata", "Missing participant_id column in metadata.")
            return

        for idx, row in df.iterrows():
            p_id = str(row.get("participant_id", "Unknown"))
            
            split = row.get("split")
            if pd.isna(split) or split not in ["train", "dev", "test"]:
                self._add_issue("invalid_labels", "FAIL", "Metadata", f"[{p_id}] Invalid split: {split}", p_id)
                
            score = row.get("phq8_score")
            if pd.isna(score):
                self._add_issue("missing_optional", "WARNING", "Metadata", f"[{p_id}] Null PHQ8 score.", p_id)
            else:
                try:
                    score_val = float(score)
                    if not (0 <= score_val <= 24):
                        self._add_issue("invalid_labels", "FAIL", "Metadata", f"[{p_id}] PHQ8 score out of bounds: {score_val}", p_id)
                except ValueError:
                    self._add_issue("invalid_labels", "FAIL", "Metadata", f"[{p_id}] Non-numeric PHQ8 score: {score}", p_id)

            binary = row.get("phq8_binary")
            if pd.isna(binary) or (str(int(float(binary))) not in ["0", "1"] if not pd.isna(binary) else False):
                self._add_issue("invalid_labels", "FAIL", "Metadata", f"[{p_id}] Invalid PHQ8 binary: {binary}", p_id)
                
            gender = row.get("gender")
            if pd.isna(gender) or str(gender) not in ["0", "1", "0.0", "1.0", 0, 1]:
                self._add_issue("invalid_labels", "FAIL", "Metadata", f"[{p_id}] Invalid gender: {gender}", p_id)

    def validate_files(self):
        """Stage 2: File existence validation"""
        logger.info("Executing Stage 2: File Validation")
        df = self.metadata_df
        
        modality_keys = {
            "audio_path": "missing_audio",
            "transcript_path": "missing_transcript",
            "covarep_path": "missing_covarep",
            "formant_path": "missing_formant",
            "au_path": "missing_au",
            "pose_path": "missing_pose",
            "gaze_path": "missing_gaze",
            "features_path": "missing_features",
            "features3d_path": "missing_features3d",
            "hog_path": "missing_hog"
        }
        
        for idx, row in df.iterrows():
            p_id = str(row["participant_id"])
            for col, issue_key in modality_keys.items():
                rel_path_val = row.get(col)
                metric_cat = "Visual" if col in ["au_path", "pose_path", "gaze_path", "features_path", "features3d_path", "hog_path"] else "Audio" if col in ["audio_path", "covarep_path", "formant_path"] else "Transcript"
                
                if pd.isna(rel_path_val) or not str(rel_path_val).strip():
                    self._add_issue(issue_key, "FAIL", "File Integrity", f"[{p_id}] Missing reference for {col} in metadata.", p_id)
                    continue
                
                full_path = self.dataset_root / str(rel_path_val)
                if not full_path.exists():
                    self._add_issue(issue_key, "FAIL", "File Integrity", f"[{p_id}] File does not exist: {full_path.name}", p_id)

    def validate_transcripts(self):
        """Stage 3: Transcript Validation"""
        logger.info("Executing Stage 3: Transcript Validation")
        df = self.metadata_df
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating Transcripts"):
            p_id = str(row["participant_id"])
            rel_path = row.get("transcript_path")
            if pd.isna(rel_path):
                continue
                
            full_path = self.dataset_root / str(rel_path)
            if not full_path.exists():
                continue
                
            try:
                trans_df = pd.read_csv(full_path, on_bad_lines='skip', engine='python')
                if len(trans_df) == 0:
                    self._add_issue("errors", "FAIL", "Transcript", f"[{p_id}] Transcript is empty.", p_id)
                    continue
                
                cols = [c.lower().strip() for c in trans_df.columns]
                
                if not any("speaker" in c for c in cols):
                    self._add_issue("errors", "FAIL", "Transcript", f"[{p_id}] 'speaker' column missing in transcript.", p_id)
                    continue
                    
                speaker_col = trans_df.columns[[i for i, c in enumerate(cols) if "speaker" in c][0]]
                speakers = trans_df[speaker_col].dropna().astype(str).str.lower().str.strip().unique()
                
                if not any("ellie" in s for s in speakers):
                    self._add_issue("warnings", "WARNING", "Transcript", f"[{p_id}] Ellie utterances missing.", p_id)
                    
                if not any("participant" in s for s in speakers):
                    self._add_issue("errors", "FAIL", "Transcript", f"[{p_id}] Participant utterances missing.", p_id)
            except Exception as e:
                self._add_issue("errors", "FAIL", "Transcript", f"[{p_id}] Transcript read error: {e}", p_id)

    def validate_covarep(self):
        """Stage 4: COVAREP Validation"""
        logger.info("Executing Stage 4: COVAREP Validation")
        df = self.metadata_df
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating COVAREP"):
            p_id = str(row["participant_id"])
            rel_path = row.get("covarep_path")
            if pd.isna(rel_path):
                continue
                
            full_path = self.dataset_root / str(rel_path)
            if not full_path.exists():
                continue
                
            try:
                cov_df = pd.read_csv(full_path, header=None, low_memory=False)
                if len(cov_df) == 0 or len(cov_df.columns) == 0:
                    self._add_issue("errors", "FAIL", "Audio", f"[{p_id}] COVAREP file empty or 0 columns.", p_id)
                    continue
                    
                numeric_df = cov_df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) != len(cov_df.columns):
                    self._add_issue("errors", "FAIL", "Audio", f"[{p_id}] COVAREP contains non-numeric columns.", p_id)
                    
                if cov_df.isna().values.any():
                    self._add_issue("nan_values_found", "WARNING", "Audio", f"[{p_id}] COVAREP contains NaN values.", p_id)
                    
                if np.isinf(numeric_df.values).any():
                    # Calculate per column stats
                    for col in numeric_df.columns:
                        col_series = numeric_df[col]
                        infs = np.isinf(col_series.values)
                        inf_count = infs.sum()
                        if inf_count > 0:
                            inf_perc = (inf_count / len(col_series)) * 100
                            msg = f"Participant {p_id} | Column {col} | Infinity values | Count : {inf_count} | Percentage : {inf_perc:.2f}%"
                            self._add_issue("errors", "WARNING", "Audio", msg, p_id)
            except Exception as e:
                self._add_issue("errors", "FAIL", "Audio", f"[{p_id}] COVAREP read error: {e}", p_id)

    def validate_formant(self):
        """Stage 5: FORMANT Validation"""
        logger.info("Executing Stage 5: FORMANT Validation")
        df = self.metadata_df
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating FORMANT"):
            p_id = str(row["participant_id"])
            rel_path = row.get("formant_path")
            if pd.isna(rel_path):
                continue
                
            full_path = self.dataset_root / str(rel_path)
            if not full_path.exists():
                continue
                
            try:
                form_df = pd.read_csv(full_path, header=None, low_memory=False)
                if len(form_df) == 0:
                    self._add_issue("errors", "FAIL", "Audio", f"[{p_id}] FORMANT file empty.", p_id)
                    continue
                
                if form_df.isna().values.any():
                    self._add_issue("nan_values_found", "WARNING", "Audio", f"[{p_id}] FORMANT contains NaN values.", p_id)
                    
            except Exception as e:
                self._add_issue("errors", "FAIL", "Audio", f"[{p_id}] FORMANT read error: {e}", p_id)

    def validate_openface(self):
        """Stage 6: OpenFace Validation"""
        logger.info("Executing Stage 6: OpenFace Validation")
        df = self.metadata_df
        
        keys = ["au_path", "pose_path", "gaze_path", "features_path", "features3d_path"]
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating OpenFace Modalities"):
            p_id = str(row["participant_id"])
            frame_counts = {}
            for col in keys:
                rel_path = row.get(col)
                if pd.isna(rel_path):
                    continue
                full_path = self.dataset_root / str(rel_path)
                if not full_path.exists():
                    continue
                    
                try:
                    mod_df = pd.read_csv(full_path, sep=",", low_memory=False)
                    frame_counts[col] = len(mod_df)
                    if len(mod_df) == 0:
                        self._add_issue("errors", "FAIL", "Visual", f"[{p_id}] {full_path.name} is empty.", p_id)
                        
                    if mod_df.isna().values.any():
                        self._add_issue("nan_values_found", "WARNING", "Visual", f"[{p_id}] {full_path.name} contains NaN.", p_id)
                except Exception as e:
                    self._add_issue("errors", "FAIL", "Visual", f"[{p_id}] OpenFace {col} read error: {e}", p_id)
            
            if len(frame_counts) > 1:
                counts = list(frame_counts.values())
                if len(set(counts)) != 1:
                    msg = f"[{p_id}] Frame mismatch across visual modalities. Counts: {frame_counts}"
                    self._add_issue("frame_mismatch", "FAIL", "Visual", msg, p_id)

    def validate_cross_modality(self):
        """Stage 7: Cross-Modality Validation"""
        logger.info("Executing Stage 7: Cross-Modality Validation")
        df = self.metadata_df
        
        for idx, row in df.iterrows():
            p_id = str(row["participant_id"])
            
            has_audio = not pd.isna(row.get("audio_path"))
            has_trans = not pd.isna(row.get("transcript_path"))
            has_vis = not pd.isna(row.get("au_path")) 
            
            if not (has_audio and has_trans and has_vis):
                missing = []
                if not has_audio: missing.append("Audio")
                if not has_trans: missing.append("Transcript")
                if not has_vis: missing.append("Visual (AU)")
                self._add_issue("warnings", "WARNING", "Cross-Modality", f"[{p_id}] Is missing core modalities: {', '.join(missing)}", p_id)

    def generate_reports(self):
        """Generate final evaluation reports (JSON, CSV, TXT)"""
        logger.info("Generating reports...")
        
        failed = len(self.failed_participants)
        passed_with_warnings = len(self.warning_participants - self.failed_participants)
        passed = self.participants_checked - failed - passed_with_warnings
        
        # Calculate dataset statistics
        df = self.metadata_df
        train_count = len(df[df["split"] == "train"]) if "split" in df.columns else 0
        dev_count = len(df[df["split"] == "dev"]) if "split" in df.columns else 0
        
        male_count = len(df[df["gender"].astype(str).isin(["1", "1.0"])]) if "gender" in df.columns else 0
        female_count = len(df[df["gender"].astype(str).isin(["0", "0.0"])]) if "gender" in df.columns else 0
        
        depressed_count = len(df[df["phq8_binary"].astype(str).isin(["1", "1.0"])]) if "phq8_binary" in df.columns else 0
        non_depressed_count = len(df[df["phq8_binary"].astype(str).isin(["0", "0.0"])]) if "phq8_binary" in df.columns else 0
        
        scores = pd.to_numeric(df["phq8_score"], errors='coerce').dropna()
        avg_score = scores.mean() if not scores.empty else 0
        med_score = scores.median() if not scores.empty else 0
        min_score = scores.min() if not scores.empty else 0
        max_score = scores.max() if not scores.empty else 0

        # Calculate health metrics
        def get_health(cat_name):
            fails = len(self.category_fails.get(cat_name, set()))
            return max(0, 100 * (1 - (fails / max(1, self.participants_checked))))

        metrics = {
            "Metadata": get_health("Metadata"),
            "File Integrity": get_health("File Integrity"),
            "Transcript": get_health("Transcript"),
            "Audio": get_health("Audio"),
            "Visual": get_health("Visual"),
            "Cross-Modality": get_health("Cross-Modality")
        }
        overall_health = sum(metrics.values()) / len(metrics) if metrics else 100.0

        # 1. JSON Report
        json_report = {
            "generation_timestamp": datetime.now().isoformat(),
            "participants_checked": self.participants_checked,
            "participants_passed": passed,
            "participants_passed_with_warnings": passed_with_warnings,
            "participants_failed": failed,
            "health_metrics": metrics,
            "overall_health": overall_health,
            "issues": self.issues
        }
        json_path = self.report_dir / "validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=4)
            
        # 2. CSV Report
        csv_path = self.report_dir / "validation_report.csv"
        pd.DataFrame(self.issues).to_csv(csv_path, index=False)
        
        # 3. TXT Report
        txt_path = self.report_dir / "validation_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("=================================================\n")
            f.write("DATASET VALIDATION REPORT\n")
            f.write("=================================================\n")
            f.write(f"Dataset Root          : {self.paths.dataset_root}\n")
            f.write(f"Metadata File         : {self.paths.metadata_dir / 'metadata.csv'}\n")
            f.write(f"Generation Time       : {json_report['generation_timestamp']}\n\n")
            
            f.write(f"Total Participants    : {self.participants_checked}\n")
            f.write(f"Train Participants    : {train_count}\n")
            f.write(f"Development Participants: {dev_count}\n")
            f.write(f"Male Participants     : {male_count}\n")
            f.write(f"Female Participants   : {female_count}\n")
            f.write(f"Depressed Participants: {depressed_count}\n")
            f.write(f"Non-depressed         : {non_depressed_count}\n\n")
            
            if not scores.empty:
                f.write(f"Average PHQ8          : {avg_score:.2f}\n")
                f.write(f"Median PHQ8           : {med_score:.2f}\n")
                f.write(f"Minimum PHQ8          : {min_score:.2f}\n")
                f.write(f"Maximum PHQ8          : {max_score:.2f}\n\n")
            
            f.write(f"Passed                : {passed}\n")
            f.write(f"Passed With Warnings  : {passed_with_warnings}\n")
            f.write(f"Failed                : {failed}\n")
            
            f.write("--------------------------------------------\n")
            for cat, score in metrics.items():
                status = "PASS" if score == 100 else ("WARNING" if score >= 90 else "FAIL")
                f.write(f"{cat.ljust(22)} {status}\n")
            f.write("--------------------------------------------\n")
            f.write(f"Overall Dataset Health  {overall_health:.1f}%\n")
            f.write("=================================================\n")
        
        logger.info(f"Reports saved to {self.report_dir}")

    def run(self):
        """Execute full validation pipeline."""
        logger.info("Starting Dataset Validation Pipeline")
        self.validate_metadata()
        self.validate_files()
        self.validate_transcripts()
        self.validate_covarep()
        self.validate_formant()
        self.validate_openface()
        self.validate_cross_modality()
        self.generate_reports()
        logger.info("Dataset Validation Pipeline Completed.")


def main():
    try:
        validator = DatasetValidator()
        validator.run()
    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
