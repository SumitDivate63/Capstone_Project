import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from .preprocessor import AudioPreprocessor
from .normalizer import AudioNormalizer
from .windowing import generate_windows_from_df
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AudioPreprocessingConfig:
    window_size: int = 150
    stride: int = 30
    normalization: str = "standard"
    interpolation: str = "linear"

class AudioPreprocessingPipeline:
    """
    Main Interface masking underlying complexities routing raw, misaligned auditory COVAREP/FORMANT metadata 
    directly into PyTorch-friendly temporal primitives natively.
    """

    def __init__(self, config: AudioPreprocessingConfig):
        self.config = config
        self.preprocessor = AudioPreprocessor(interpolation_method=config.interpolation)
        self.normalizer = AudioNormalizer(method=config.normalization)
        self.feature_columns: List[str] = []

    def _determine_feature_columns(self, df: pd.DataFrame) -> None:
        """Inspects concatenated tables isolating purely calculable parameters."""
        ignore_cols = ["frame", "timestamp"]
        self.feature_columns = [
            col for col in df.columns 
            if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
    def fit(self, audio_data_list: List[Dict[str, Any]]) -> None:
        """
        Operates entirely across exhaustive training participant bounds modeling normalization state.

        Args:
            audio_data_list: Complete set of "audio" dictionary objects output by DAICDataset iterable.
        """
        logger.info("Initializing Audio Pipeline Fit cycle...")
        aggregated_dfs = []
        
        for idx, audio_data in enumerate(audio_data_list):
            try:
                aggregated_dfs.append(self.preprocessor.process_participant(idx, audio_data))
            except Exception as e:
                logger.error(f"Participant {idx} failed during scalar profiling mapping: {e}")
                
        if not aggregated_dfs:
            raise ValueError("Pipeline fit failed - no participant arrays generated valid sync conditions.")
            
        master_df = pd.concat(aggregated_dfs, ignore_index=True)
        self._determine_feature_columns(master_df)
        self.normalizer.fit(master_df, self.feature_columns)
        logger.info("Audio Pipeline Normalization successful.")

    def transform(self, participant_id: int, audio_data: Dict[str, Any]) -> torch.Tensor:
        """
        Invoked granularly post-training to route DAICDataset structures natively into network spaces.

        Args:
            participant_id: DAIC-WOZ global tracker
            audio_data: `dataset[participant]["audio"]` component block.

        Returns:
            torch.FloatTensor (Temporal Chunk, Array Length, Modality Span)
        """
        df = self.preprocessor.process_participant(participant_id, audio_data)
        
        if not self.feature_columns:
           self._determine_feature_columns(df)
           
        norm_df = self.normalizer.transform(df)
        
        tensor = generate_windows_from_df(
            norm_df, 
            self.feature_columns, 
            self.config.window_size, 
            self.config.stride
        )
        
        logger.info(
            f"Participant {participant_id} | Final Audio Output Shape: {list(tensor.shape)} "
            f"using {self.config.normalization} scalar transforms."
        )
        return tensor

    def fit_transform(self, participant_id: int, audio_data: Dict[str, Any]) -> torch.Tensor:
        """Convenience method blending both pipelines (Rarely utilized during proper Train/Dev/Test splits)."""
        df = self.preprocessor.process_participant(participant_id, audio_data)
        self._determine_feature_columns(df)
        
        norm_df = self.normalizer.fit_transform(df, self.feature_columns)
        return generate_windows_from_df(
            norm_df, 
            self.feature_columns, 
            self.config.window_size, 
            self.config.stride
        )

    def save_scaler(self, path_str: str) -> None:
        self.normalizer.save(Path(path_str))

    def load_scaler(self, path_str: str) -> None:
        self.normalizer.load(Path(path_str))
