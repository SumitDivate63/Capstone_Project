import torch
import numpy as np
import pandas as pd
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

def generate_sliding_windows(
    data: np.ndarray, 
    window_size: int, 
    stride: int
) -> torch.Tensor:
    """
    Strided tensor generation partitioning audio frames optimally.

    Args:
        data: Matrix containing (N_frames, Features)
        window_size: Length of each spatial chunk
        stride: Offset distance traversing longitudinally

    Returns:
        Prepared torch.FloatTensor (Windows, Size, Feature_Dimension)
    """
    num_frames, feature_dim = data.shape
    
    if num_frames < window_size:
        logger.warning(
            f"Insufficient continuous frames ({num_frames}) to populate requested block ({window_size}). "
            f"Returning null tensor."
        )
        return torch.empty((0, window_size, feature_dim), dtype=torch.float32)

    shape = ((num_frames - window_size) // stride + 1, window_size, feature_dim)
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return torch.tensor(windows.copy(), dtype=torch.float32)

def generate_windows_from_df(
    df: pd.DataFrame, 
    feature_columns: List[str], 
    window_size: int, 
    stride: int
) -> torch.Tensor:
    """
    Translates Panda tabular frames rapidly into Deep Learning primitives.

    Args:
        df: Processed, normalized DataFrame of Audio features.
        feature_columns: Numerical array target boundaries.
        window_size: Temporal span parameters.
        stride: Temporal leap length step parameters.

    Returns:
        PyTorch float array directly addressable by models.
    """
    if not feature_columns:
        raise ValueError("Cannot formulate tensors devoid of declared continuous arrays (feature mapping missing).")
        
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
         raise KeyError(f"Window builder isolated from missing keys: {missing}")

    data_matrix = df[feature_columns].to_numpy(dtype=np.float32)
    return generate_sliding_windows(data_matrix, window_size, stride)
