"""Main configuration setup."""
from dataclasses import dataclass
from .path_config import PathConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig

@dataclass
class AppConfig:
    paths: PathConfig
    model: ModelConfig
    training: TrainingConfig
