import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

class AttentionPooling(nn.Module):
    """
    Learns temporal importance weights across windows to pool 
    variable length encoded sequences into a single dense representation.
    """
    def __init__(self, embed_dim: int = 256):
        """
        Args:
            embed_dim: Expected dimensionality matching the Encoder output space.
        """
        super().__init__()
        
        # Maps embedded dimensions directly to un-normalized scalar scores
        self.attention_weights = nn.Linear(embed_dim, 1, bias=False)
        
        logger.info(f"Initialized AttentionPooling (Embed Dim: {embed_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes sequence pooling natively.
        
        Args:
            x: Input tensor of shape (Batch_Size, Temporal_Steps, Embedding_Dim)
            
        Returns:
            Pooled tensor of shape (Batch_Size, Embedding_Dim)
        """
        # (B, T, D) -> (B, T, 1)
        scores = self.attention_weights(x)
        
        # Softmax specifically tracking across the temporal sequences
        weights = torch.softmax(scores, dim=1)
        
        # Weight application & summation crunching the Sequence bounds
        pooled = torch.sum(weights * x, dim=1)
        
        return pooled
