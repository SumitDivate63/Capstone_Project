import pandas as pd
import numpy as np
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_frames(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validates frame ordering, duplicates, and missing indices.

    Args:
        dfs: Dictionary of modality name to pandas DataFrame.

    Raises:
        ValueError: If corrupted or duplicated rows exist.
        KeyError: If 'frame' column is missing.
    """
    for name, df in dfs.items():
        if "frame" not in df.columns:
            raise KeyError(f"Missing 'frame' column in {name} DataFrame.")
        
        if df["frame"].isnull().any():
            raise ValueError(f"Invalid (NaN) timestamps/frames found in {name}.")
            
        if df["frame"].duplicated().any():
            raise ValueError(f"Duplicated frames detected in {name}.")
            
        if not df["frame"].is_monotonic_increasing:
            raise ValueError(f"Frames are not monotonically increasing in {name}.")


def synchronize_frames(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Synchronizes DataFrames safely using the common 'frame' index. 
    Removes unmatched frames.

    Args:
        dfs: Dictionary of DataFrames with a 'frame' column.

    Returns:
        A perfectly synchronized, concatenated DataFrame.
    """
    if not dfs:
        raise ValueError("Empty dictionary provided for synchronization.")
        
    synced_df = None
    for name, df in dfs.items():
        # Set frame as index for clean merging
        df_indexed = df.set_index("frame").add_prefix(f"{name}_")
        
        if synced_df is None:
            synced_df = df_indexed
        else:
            # Inner join to keep only matching frames across all modalities
            synced_df = synced_df.join(df_indexed, how="inner")
            
    if synced_df is None or synced_df.empty:
        raise ValueError("Synchronization failed: No overlapping frames found across modalities.")
        
    return synced_df.reset_index()


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces +Inf and -Inf with NaN for future interpolation.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df

def clean_invalid_values(dfs: Dict[str, pd.DataFrame], participant_id: int) -> Dict[str, pd.DataFrame]:
    """
    Detects invalid values like -1.#IND, INF, converts to numeric, logs counts.
    Raises ValueError if completely unrecoverable.
    """
    invalid_strs = ["-1.#IND", "1.#IND", "INF", "-INF", "NaN", "nan", "inf", "Infinity"]
    cleaned_dfs = {}
    
    total_invalid = 0
    cols_affected = set()
    rows_affected = set()
    unrecoverable = False
    
    for name, df in dfs.items():
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object or str(df_clean[col].dtype).startswith('str'):
                mask = df_clean[col].astype(str).str.strip().isin(invalid_strs)
                if mask.any():
                    total_invalid += mask.sum()
                    cols_affected.add(col)
                    rows_affected.update(df_clean.index[mask].tolist())
                    
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                inf_mask = np.isinf(df_clean[col])
                if inf_mask.any():
                    total_invalid += inf_mask.sum()
                    cols_affected.add(col)
                    rows_affected.update(df_clean.index[inf_mask].tolist())
                    df_clean.loc[inf_mask, col] = np.nan
                    
        # Check if all feature rows are null
        feat_cols = [c for c in df_clean.columns if c != 'frame']
        if len(feat_cols) > 0 and df_clean[feat_cols].isna().all().all():
            unrecoverable = True

        cleaned_dfs[name] = df_clean
        
    if unrecoverable:
        print(f"Participant {participant_id} skipped:")
        print("Reason: Invalid feature values")
        raise ValueError("corrupted feature file")
        
    if total_invalid > 0:
        print(f"Participant: {participant_id}")
        print("Reason: Invalid feature values")
        print(f"Rows affected: {len(rows_affected)}")
        print(f"Columns affected: {len(cols_affected)}")
        print("Action taken: Replaced with np.nan")
            
    return cleaned_dfs


def interpolate_missing(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """
    Replaces NaN values using linear interpolation, followed by forward fill, 
    and backward fill to handle edge cases.
    
    Args:
        df: Input DataFrame containing NaNs.
        method: Interpolation method (default: linear).

    Returns:
        Clean DataFrame without missing values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 1. Interpolate
    kwargs = {}
    if method == "linear":
        kwargs = {"method": "linear"}
    else:
        # Fallback to nearest or similar if requested
        kwargs = {"method": method}
        
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both", **kwargs)
    
    # 2. Forward fill for boundary NaNs
    df[numeric_cols] = df[numeric_cols].ffill()
    
    # 3. Backward fill for remaining starting NaNs
    df[numeric_cols] = df[numeric_cols].bfill()
    
    if df[numeric_cols].isnull().any().any():
         logger.warning("NaNs remain after interpolation and filling. Setting to 0.")
         df[numeric_cols] = df[numeric_cols].fillna(0)
         
    return df
