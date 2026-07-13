"""Base model class."""
import torch.nn as nn

class BaseModel(nn.Module):
    """Abstract base model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        pass
