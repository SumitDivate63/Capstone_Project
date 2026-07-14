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
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with Inf replaced by NaN.
    """
    df = df.copy()
    # Replace purely numeric inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


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
