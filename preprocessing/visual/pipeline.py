import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .preprocessor import VisualPreprocessor
from .normalizer import VisualNormalizer
from .windowing import generate_windows_from_df
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VisualPreprocessingConfig:
    window_size: int = 150
    stride: int = 30
    normalization: str = "standard"
    interpolation: str = "linear"


class VisualPreprocessingPipeline:
    """
    High-level API for Visual Preprocessing.
    Consumes raw Visual Dicts from DAICDataset and outputs Standardized Temporal PyTorch Tensors.
    """

    def __init__(self, config: VisualPreprocessingConfig):
        """
        Initializes the entire Visual Pipeline with parameters.
        """
        self.config = config
        self.preprocessor = VisualPreprocessor(interpolation_method=config.interpolation)
        self.normalizer = VisualNormalizer(method=config.normalization)
        self.feature_columns = []

    def _determine_feature_columns(self, df: pd.DataFrame) -> None:
        """Determines which columns contain continuous numeric data suitable for tensors."""
        # Excluding auxiliary indices created during sync like 'frame'
        ignore_cols = ["frame"]
        self.feature_columns = [
            col for col in df.columns 
            if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    def fit(self, visual_data_list: list[Dict[str, pd.DataFrame]]) -> None:
        """
        Fits the normalizer entirely on a collection of training participants.

        Args:
            visual_data_list: List of 'visual' dict outputs from DAICDataset.
        """
        logger.info("Starting Pipeline Fit on training data...")
        aggregated_dfs = []
        
        for idx, visual_data in enumerate(visual_data_list):
            try:
                cleansed_df = self.preprocessor.process_participant(participant_id=idx, visual_data=visual_data)
                aggregated_dfs.append(cleansed_df)
            except Exception as e:
                logger.error(f"Skipping participant {idx} in fitting due to errors: {e}")
                
        if not aggregated_dfs:
            raise ValueError("No valid participants available to fit the scaler.")
            
        master_df = pd.concat(aggregated_dfs, ignore_index=True)
        self._determine_feature_columns(master_df)
        self.normalizer.fit(master_df, self.feature_columns)
        logger.info("Pipeline Normalizer fit successful.")

    def transform(self, participant_id: int, visual_data: Dict[str, Any]) -> torch.Tensor:
        """
        Transforms a single participant's raw data into windowed tensors.

        Args:
            participant_id: DAIC-WOZ participant numerical ID.
            visual_data: The 'visual' component obtained from DAICDataset.

        Returns:
            torch.FloatTensor of shape (Windows, Window_Size, Feature_Dimension).
        """
        # Orchestrate Preprocessing (Clean & Sync)
        cleansed_df = self.preprocessor.process_participant(participant_id, visual_data)
        
        if not self.feature_columns:
           self._determine_feature_columns(cleansed_df)
           
        missing_cols = [c for c in self.feature_columns if c not in cleansed_df.columns]
        if missing_cols:
             raise KeyError(f"Feature columns {missing_cols} missing in participant {participant_id}")
             
        # Normalize
        normalized_df = self.normalizer.transform(cleansed_df)
        
        # Window & Tensor Gen
        windows_tensor = generate_windows_from_df(
            normalized_df, 
            self.feature_columns, 
            self.config.window_size, 
            self.config.stride
        )
        
        logger.info(
            f"Participant {participant_id}: Processed Tensor Formed -> "
            f"Shape: {list(windows_tensor.shape)}, Type: {self.config.normalization}"
        )
        return windows_tensor


    def fit_transform(self, participant_id: int, visual_data: Dict[str, Any]) -> torch.Tensor:
        """
        Normally fit is calculated over ENTIRE training splits. 
        This is provided for isolated logic conforming strictly to Scikit norms.
        """
        cleansed_df = self.preprocessor.process_participant(participant_id, visual_data)
        self._determine_feature_columns(cleansed_df)
        
        normalized_df = self.normalizer.fit_transform(cleansed_df, self.feature_columns)
        
        windows_tensor = generate_windows_from_df(
            normalized_df, 
            self.feature_columns, 
            self.config.window_size, 
            self.config.stride
        )
        return windows_tensor

    def save_scaler(self, path_str: str) -> None:
        """Saves Standardizers to disk."""
        self.normalizer.save(Path(path_str))

    def load_scaler(self, path_str: str) -> None:
        """Loads Standardizers from disk."""
        self.normalizer.load(Path(path_str))
