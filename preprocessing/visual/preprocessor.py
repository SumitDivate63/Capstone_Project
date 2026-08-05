import pandas as pd
from typing import Dict, Any, List

from .frame_utils import validate_frames, synchronize_frames, handle_infinities, interpolate_missing, clean_invalid_values
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualPreprocessor:
    """
    Orchestrates the frame validation, aggregation and cleansing mechanisms 
    sequentially.
    """
    
    def __init__(self, interpolation_method: str = "linear"):
        self.interpolation_method = interpolation_method

    def process_participant(self, participant_id: int, visual_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Executes internal pipeline logic for a single participant's raw data: 
        Validates -> Synchronizes -> Infs -> NaNs -> Outputs Cleared DF.

        Args:
            participant_id: ID for logging.
            visual_data: 'visual' dict extracted strictly from DAICDataset.

        Returns:
            Fully synchronized and cleansed pandas dataframe of numerical features.
        """
        # HOG is binary path, so we exclude it.
        dfs_to_process = {}
        for key, value in visual_data.items():
            if isinstance(value, pd.DataFrame):
                dfs_to_process[key] = value
                
        # 0. Clean string artifacts and infs
        dfs_to_process = clean_invalid_values(dfs_to_process, participant_id)

        # 1. Validate frames
        try:
            validate_frames(dfs_to_process)
        except Exception as e:
            logger.error(f"Participant {participant_id} Validation Failed: {e}")
            raise
            
        initial_frame_count = min([len(df) for df in dfs_to_process.values()])
            
        # 2. Synchronize
        synced_df = synchronize_frames(dfs_to_process)
        synced_frame_count = len(synced_df)
        
        removed = initial_frame_count - synced_frame_count
        if removed > 0:
            logger.info(f"Participant {participant_id}: Removed {removed} unmatched frames during sync.")

        # 3. Handle infinities
        cleansed_df = handle_infinities(synced_df)
        
        # 4. Handle NaNs
        na_count = cleansed_df.isnull().sum().sum()
        cleansed_df = interpolate_missing(cleansed_df, method=self.interpolation_method)
        
        if na_count > 0:
             logger.info(f"Participant {participant_id}: Interpolated {na_count} missing/infinite values.")
             
        # Extract purely feature columns, exclude auxiliary identifier cols like 'frame' if needed
        # The prompt requires column-wise fusion which is achieved inherently via `synchronize_frames`.
        
        return cleansed_df
