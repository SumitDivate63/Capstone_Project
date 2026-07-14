import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_frames(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validates frame/timestamp ordering, duplicates, and missing indices.

    Args:
        dfs: Dictionary of modality name to pandas DataFrame.

    Raises:
        ValueError: If corrupted, duplicated, or non-monotonic rows exist.
        KeyError: If synchronization column is missing.
    """
    sync_col = "frame"
    
    for name, df in dfs.items():
        if sync_col not in df.columns:
            # Fallback to timestamp if frame isn't present but keep standard 'frame' preference
            if "timestamp" in df.columns:
                sync_col = "timestamp"
            else:
                raise KeyError(f"Missing '{sync_col}' or 'timestamp' column in {name} DataFrame.")
        
        if df[sync_col].isnull().any():
            raise ValueError(f"Invalid (NaN) {sync_col} values found in {name}.")
            
        if df[sync_col].duplicated().any():
            raise ValueError(f"Duplicated {sync_col}s detected in {name}.")
            
        if not df[sync_col].is_monotonic_increasing:
            raise ValueError(f"{sync_col}s are not monotonically increasing in {name}.")


def synchronize_frames(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Synchronizes DataFrames safely using the common index (frame or timestamp).
    Removes unmatched rows.

    Args:
        dfs: Dictionary of DataFrames (e.g., COVAREP, FORMANT).

    Returns:
        A precisely synchronized, concatenated DataFrame.
    """
    if not dfs:
        raise ValueError("Empty dictionary provided for synchronization.")
        
    sync_col = "frame" if "frame" in list(dfs.values())[0].columns else "timestamp"
    
    synced_df = None
    for name, df in dfs.items():
        if sync_col not in df.columns:
            raise KeyError(f"Column '{sync_col}' not found in {name}. Cannot synchronize.")
            
        df_indexed = df.set_index(sync_col).add_prefix(f"{name}_")
        
        if synced_df is None:
            synced_df = df_indexed
        else:
            synced_df = synced_df.join(df_indexed, how="inner")
            
    if synced_df is None or synced_df.empty:
        raise ValueError("Synchronization failed: No overlapping temporal frames found.")
        
    return synced_df.reset_index()


def handle_infinities(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Replaces +Inf and -Inf with NaN for future interpolation.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (DataFrame with Inf replaced by NaN, Count of Infinities replaced).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate how many infs exist before replacing
    inf_count = np.isinf(df[numeric_cols]).values.sum()
    
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df, int(inf_count)


def interpolate_missing(df: pd.DataFrame, method: str = "linear") -> Tuple[pd.DataFrame, int]:
    """
    Replaces NaN values using provided interpolation, followed by forward and backward fill.
    
    Args:
        df: Input DataFrame containing NaNs.
        method: Interpolation method (default: linear).

    Returns:
        Tuple of (Clean DataFrame without missing values, Count of NaNs replaced).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    na_count = df[numeric_cols].isnull().values.sum()
    
    if na_count == 0:
        return df, 0
        
    kwargs = {"method": method} if method in ["linear", "nearest", "polynomial", "spline"] else {"method": "linear"}
    
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both", **kwargs)
    df[numeric_cols] = df[numeric_cols].ffill()
    df[numeric_cols] = df[numeric_cols].bfill()
    
    # Emergency fallback
    if df[numeric_cols].isnull().any().any():
         logger.warning("Emergency zero-fill applied. Some NaNs defied interpolation/filling sequences.")
         df[numeric_cols] = df[numeric_cols].fillna(0)
         
    return df, int(na_count)
