import pandas as pd
from typing import List, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

from utils.logger import get_logger

logger = get_logger(__name__)

class AudioNormalizer:
    """
    Scikit-Learn driven Normalization interface specifically structured for Audio modalities.
    Ensures state is fit securely strictly on Training pipelines.
    """
    
    def __init__(self, method: str = "standard"):
        """
        Initializes chosen Scaler logic.
        
        Args:
            method: 'standard', 'minmax', or 'robust'.
        """
        self.method = method.lower()
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling methodology: {method}")
            
        self.is_fitted = False
        self.feature_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> None:
        """
        Aggregates distribution markers purely from provided dataframe rows.
        Never to be called against generalized Dev/Test folds manually!
        """
        if not feature_columns:
            raise ValueError("Empty feature columns requested for fitting.")
            
        self.feature_columns = feature_columns
        self.scaler.fit(df[self.feature_columns])
        self.is_fitted = True
        logger.info(f"Audio Scaler ('{self.method}') initialized optimally spanning {len(feature_columns)} features.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes feature projection onto identical dimensional spaces.
        """
        if not self.is_fitted or not self.feature_columns:
            raise RuntimeError("Transform aborted. Scaler not fit.")
            
        df_out = df.copy()
        missing = set(self.feature_columns) - set(df_out.columns)
        if missing:
            raise KeyError(f"Cannot transform. Target variables missing: {missing}")
            
        df_out[self.feature_columns] = self.scaler.transform(df_out[self.feature_columns])
        return df_out
        
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        self.fit(df, feature_columns)
        return self.transform(df)

    def save(self, filepath: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot persist unfitted normalization context.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "columns": self.feature_columns}, filepath)
        logger.info(f"Persisted Audio Normalizer state externally to {filepath}")

    def load(self, filepath: Path) -> None:
        if not filepath.exists():
            raise FileNotFoundError(f"Missing valid Scaler binary at {filepath}")
        payload = joblib.load(filepath)
        self.scaler = payload["scaler"]
        self.feature_columns = payload["columns"]
        self.is_fitted = True
        logger.info(f"Re-hydrated valid Audio Scaler spanning {len(self.feature_columns)} attributes.")
