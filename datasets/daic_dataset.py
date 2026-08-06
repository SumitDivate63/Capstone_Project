"""
Dataset module for Explainable Multimodal Depression Detection.
Loads validated data into memory without preprocessing.
"""

from typing import Dict, Any
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.path_config import PathConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class DAICDataset(Dataset):
    """
    PyTorch Dataset for DAIC-WOZ.
    Loads raw metadata and paths strictly from metadata.csv.
    Does NOT perform preprocessing or tensor conversion.
    """

    def __init__(
        self,
        split: str = "train",
        load_visual: bool = True,
        load_audio: bool = True,
        load_text: bool = True
    ):
        """
        Initialize the DAICDataset.

        Args:
            split: Dataset split to load ('train', 'dev', 'test', 'all').
            load_visual: Whether to load visual features.
            load_audio: Whether to load audio features.
            load_text: Whether to load text features.
        """
        self.split = split
        self.load_visual = load_visual
        self.load_audio = load_audio
        self.load_text = load_text

        self.path_config = PathConfig()
        self.metadata_path = self.path_config.metadata_dir / "metadata.csv"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        # Load all metadata
        self.metadata_df = pd.read_csv(self.metadata_path)

        # Filter by split if not "all"
        if self.split != "all":
            self.metadata_df = self.metadata_df[self.metadata_df["split"] == self.split].reset_index(drop=True)

        if len(self.metadata_df) == 0:
            raise ValueError(f"No participants found for split: {self.split}")

        logger.info("DAICDataset initialized.")
        logger.info(f"Number of participants: {len(self.metadata_df)}")
        logger.info(f"Selected split: {self.split}")
        logger.info(f"Selected modalities - Visual: {self.load_visual}, Audio: {self.load_audio}, Text: {self.load_text}")

    def __len__(self) -> int:
        """Return the number of participants in the dataset."""
        return len(self.metadata_df)

    def _resolve_path(self, path_str: Any, participant_id: int, modality: str) -> Path:
        """Helper to resolve and validate file paths."""
        if pd.isna(path_str):
            raise ValueError(
                f"Missing {modality} path in metadata for participant ID {participant_id}"
            )
        
        path = Path(str(path_str))
        if not path.is_absolute():
            path = self.path_config.dataset_root / path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {modality} file for participant ID {participant_id} at {path}"
            )

        return path

    def _load_csv(self, path_str: Any, participant_id: int, modality: str, **kwargs) -> pd.DataFrame:
        """Helper to load a CSV file safely."""
        path = self._resolve_path(path_str, participant_id, modality)
        
        # Merge kwargs with our defaults
        read_kwargs = {"low_memory": False}
        read_kwargs.update(kwargs)
        
        df = pd.read_csv(path, **read_kwargs)
        
        # Robust delimiter detection for transcripts which use tab separation in DAIC-WOZ
        if modality == "Transcript" and len(df.columns) == 1:
            df = pd.read_csv(path, sep="\t", **read_kwargs)
            
        return df

    def _load_visual(self, row: pd.Series, participant_id: int) -> Dict[str, Any]:
        """Load visual features."""
        return {
            "au": self._load_csv(row.get("au_path"), participant_id, "AU"),
            "pose": self._load_csv(row.get("pose_path"), participant_id, "Pose"),
            "gaze": self._load_csv(row.get("gaze_path"), participant_id, "Gaze"),
            "features": self._load_csv(row.get("features_path"), participant_id, "Features"),
            "features3d": self._load_csv(row.get("features3d_path"), participant_id, "Features3D"),
            "hog": self._resolve_path(row.get("hog_path"), participant_id, "HOG")
        }

    def _load_audio(self, row: pd.Series, participant_id: int) -> Dict[str, Any]:
        """Load audio features."""
        return {
            "covarep": self._load_csv(row.get("covarep_path"), participant_id, "COVAREP", header=None),
            "formant": self._load_csv(row.get("formant_path"), participant_id, "Formant", header=None),
            "waveform": self._resolve_path(row.get("audio_path"), participant_id, "Waveform")
        }

    def _load_text(self, row: pd.Series, participant_id: int) -> Dict[str, Any]:
        """Load text features."""
        return {
            "transcript": self._load_csv(row.get("transcript_path"), participant_id, "Transcript")
        }

    def _load_labels(self, row: pd.Series) -> Dict[str, int]:
        """Load target labels."""
        phq8_score = int(row["phq8_score"])
        phq8_binary = 1 if phq8_score >= 10 else 0
        return {
            "phq8_score": phq8_score,
            "phq8_binary": phq8_binary
        }

    def _load_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Load participant metadata."""
        return {
            "gender": int(row["gender"]),
            "split": str(row["split"])
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single participant's data.

        Args:
            idx: Index of the participant.

        Returns:
            Dictionary containing visual, audio, text, labels, and metadata.
        """
        row = self.metadata_df.iloc[idx]
        metadata_id = int(row["participant_id"])
        participant_id = metadata_id  # Derived from iteration

        assert participant_id == metadata_id, f"ID mismatch: {participant_id} != {metadata_id}"
        
        labels = self._load_labels(row)
        assert labels["phq8_binary"] in [0, 1], f"Invalid label {labels['phq8_binary']} for {participant_id}"

        item = {
            "participant_id": participant_id,
            "labels": labels,
            "metadata": self._load_metadata(row)
        }

        if self.load_visual:
            item["visual"] = self._load_visual(row, participant_id)

        if self.load_audio:
            item["audio"] = self._load_audio(row, participant_id)

        if self.load_text:
            item["text"] = self._load_text(row, participant_id)

        return item
