import torch
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path
import joblib

from utils.logger import get_logger

logger = get_logger(__name__)


class VisualNormalizer:
    """
    Handles feature normalization for visual data.
    Fits only on training data and retains state for dev/test.
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize the Normalizer.

        Args:
            method: Feature scaling method: 'standard', 'minmax', or 'robust'.
        """
        self.method = method.lower()
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        self.is_fitted = False
        self.feature_columns = None


    def fit(self, df: pd.DataFrame, feature_columns: list) -> None:
        """
        Fit the scaler on the designated training features.
        
        Args:
            df: Training DataFrame.
            feature_columns: Columns to fit on.
        """
        if not feature_columns:
            raise ValueError("No feature columns provided for fitting scaler.")
            
        self.feature_columns = feature_columns
        self.scaler.fit(df[self.feature_columns])
        self.is_fitted = True
        logger.info(f"Fitted {self.method} scaler on {len(feature_columns)} features.")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the features using the fitted scaler.
        
        Args:
            df: DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first, or use a fitted scaler.")
            
        # Ensure we don't mutate input randomly outside
        df_out = df.copy()
        missing_cols = [col for col in self.feature_columns if col not in df_out.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from DataFrame during transform: {missing_cols}")
            
        df_out[self.feature_columns] = self.scaler.transform(df_out[self.feature_columns])
        return df_out


    def fit_transform(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Fits and transforms in a single pass.
        """
        self.fit(df, feature_columns)
        return self.transform(df)


    def save(self, filepath: Path) -> None:
        """Save the fitted scaler state to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "columns": self.feature_columns}, filepath)
        logger.info(f"Saved fitted normalizer to {filepath}")


    def load(self, filepath: Path) -> None:
        """Load a fitted scaler state from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
            
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.feature_columns = data["columns"]
        self.is_fitted = True
        logger.info(f"Loaded strictly fitted normalizer from {filepath}")
