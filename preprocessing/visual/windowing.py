import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_sliding_windows(
    data: np.ndarray, 
    window_size: int, 
    stride: int
) -> torch.Tensor:
    """
    Generates temporal, overlapping sliding windows over standardized frame sequences.

    Args:
        data: Numpy array of shape (Number_of_frames, Feature_Dimension).
        window_size: Number of frames per window.
        stride: Hop size between consecutive windows.

    Returns:
        torch.FloatTensor of shape (Number_of_Windows, Window_Size, Feature_Dimension).
    """
    num_frames, feature_dim = data.shape
    
    if num_frames < window_size:
        raise ValueError(f"Insufficient frames ({num_frames}) for window size ({window_size}).")

    shape = ((num_frames - window_size) // stride + 1, window_size, feature_dim)
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    windows_copy = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()
    
    # Needs to be a continuous float tensor
    tensor = torch.tensor(windows_copy, dtype=torch.float32)
    
    # Verify shape
    expected_shape = (window_size, 393)
    for i in range(tensor.size(0)):
        if tuple(tensor[i].shape) != expected_shape:
            raise ValueError(f"Invalid window shape {tuple(tensor[i].shape)}. Expected {expected_shape}.")
            
    return tensor


def generate_windows_from_df(
    df: pd.DataFrame, 
    feature_columns: List[str], 
    window_size: int, 
    stride: int
) -> torch.Tensor:
    """
    Extracts features from DataFrame and generates sequence tensors.

    Args:
        df: The normalized, clean dataframe.
        feature_columns: Columns to extract for tensor creation.
        window_size: Window size.
        stride: Stride size.

    Returns:
        torch.FloatTensor prepared for Visual Agent consumption.
    """
    if not feature_columns:
        raise ValueError("Feature columns mapping is empty.")
        
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
         raise KeyError(f"Missing columns for window generation: {missing}")

    data_matrix = df[feature_columns].to_numpy(dtype=np.float32)
    return generate_sliding_windows(data_matrix, window_size, stride)
