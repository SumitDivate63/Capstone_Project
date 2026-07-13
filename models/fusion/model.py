"""Fusion model."""
from ..base.base_model import BaseModel

class FusionModel(BaseModel):
    """
    Fusion architecture.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x):
        pass
