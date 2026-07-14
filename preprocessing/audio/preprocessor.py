import pandas as pd
from typing import Dict, Any

from .frame_utils import validate_frames, synchronize_frames, handle_infinities, interpolate_missing
from utils.logger import get_logger

logger = get_logger(__name__)

class AudioPreprocessor:
    """
    Execution gateway isolating noisy, uncoordinated COVAREP + FORMANT data structures, filtering
    away gaps iteratively to unearth a perfectly sequenced multimodal foundation.
    """
    
    def __init__(self, interpolation_method: str = "linear"):
        self.interpolation_method = interpolation_method

    def process_participant(self, participant_id: int, audio_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Invokes deterministic routines sequentially filtering raw datasets down towards cleanly bounded values.

        Args:
            participant_id: ID for granular trace logging.
            audio_data: The entire sub-payload obtained querying specific Dataset agents. (covarep, formant, etc)

        Returns:
            Concatenated, synchronized dataframe immune to NaNs & nulls.
        """
        # Filter for DataFrames exclusively (Ignore pure waveforms which manifest as `pathlib.Path` structures).
        dfs_to_process = {k: v for k, v in audio_data.items() if isinstance(v, pd.DataFrame)}
        
        if not dfs_to_process:
            raise ValueError(f"Participant {participant_id}: Missing processable pandas arrays inside Audio data payload.")

        # 1. Verification Pass
        try:
            validate_frames(dfs_to_process)
        except Exception as e:
            logger.error(f"Participant {participant_id}: Found invalid chronological frame sequencing! Exception: {e}")
            raise

        initial_rows = sum([len(df) for df in dfs_to_process.values()]) // len(dfs_to_process)

        # 2. Alignment Synchronization
        synced_df = synchronize_frames(dfs_to_process)
        synced_rows = len(synced_df)
        
        removed = initial_rows - synced_rows
        
        # 3. Suppress Infinities (Very critical for Audio DAIC-WOZ logic where mathematical anomalies abound)
        cleaned_df, inf_count = handle_infinities(synced_df)

        # 4. Correct Null/Empty ranges (Gap traversing)
        final_df, nan_count = interpolate_missing(cleaned_df, method=self.interpolation_method)
        
        logger.info(
            f"Participant {participant_id} Summary | Original Rows: {initial_rows} | "
            f"Sync Drops: {removed} | Infs Reset: {inf_count} | Nulls Rebuilt: {nan_count}"
        )

        return final_df
