import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_frames(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validates presence of data without assuming frame/timestamp columns.

    Args:
        dfs: Dictionary of modality name to pandas DataFrame.

    Raises:
        ValueError: If a dataframe is completely empty.
    """
    for name, df in dfs.items():
        if df.empty:
            raise ValueError(f"DataFrame {name} is completely empty.")


def synchronize_frames(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Synchronizes DataFrames safely using row order only as DAIC-WOZ audio does not contain timestamps.
    Removes unmatched rows.

    Args:
        dfs: Dictionary of DataFrames (COVAREP, FORMANT).

    Returns:
        A precisely synchronized, concatenated DataFrame.
    """
    if not dfs:
        raise ValueError("Empty dictionary provided for synchronization.")
        
    if "covarep" not in dfs or "formant" not in dfs:
        raise ValueError("Both 'covarep' and 'formant' DataFrames are required.")

    covarep = dfs["covarep"]
    formant = dfs["formant"]
    
    rows = min(len(covarep), len(formant))
    
    covarep = covarep.iloc[:rows].reset_index(drop=True)
    formant = formant.iloc[:rows].reset_index(drop=True)
    
    # Prefix columns to avoid duplicate column index issues natively
    covarep.columns = [f"covarep_{c}" for c in covarep.columns]
    formant.columns = [f"formant_{c}" for c in formant.columns]
    
    audio_df = pd.concat([covarep, formant], axis=1)
    
    return audio_df


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
