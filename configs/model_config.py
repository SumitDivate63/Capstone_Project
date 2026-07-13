"""Model configuration dataclasses."""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Settings for the AI models."""
    d_model: int = 768
    num_heads: int = 8
