"""Training configuration dataclasses."""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Hyperparameters and training settings."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
