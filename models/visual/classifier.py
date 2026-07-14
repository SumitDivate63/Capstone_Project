import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

class VisualClassifier(nn.Module):
    """
    Standalone visual decision head mappings translating pooled embeddings continuously 
    down into final diagnostic categorical spaces.
    """
    def __init__(self, input_dim: int = 256, num_classes: int = 2):
        """
        Constraints topology explicitly to PyTorch fully-connected definitions.
        
        Args:
            input_dim: Final dimension count post-pooling.
            num_classes: Classification bounds matching binary depression targets. 
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        logger.info(f"Initialized VisualClassifier (Input: {input_dim}, Output classes: {num_classes})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Outputs pure raw logits mappings securely bypassing early softmax restrictions.
        
        Args:
            x: Pooled Tensor (Batch_Size, Embedding_Dim)
            
        Returns:
            Classification Logits (Batch_Size, num_classes)
        """
        return self.classifier(x)
