"""Audio Agent model."""
from ..base.base_model import BaseModel

class AudioAgentModel(BaseModel):
    """
    Audio Agent architecture.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x):
        pass
