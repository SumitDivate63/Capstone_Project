import torch
import torch.nn as nn
from models.visual.transformer_encoder import VisualTransformerEncoder
from models.visual.attention_pooling import AttentionPooling
from models.visual.classifier import VisualClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualModel(nn.Module):
    """
    Comprehensive unified macro-architecture composing preprocessing targets (B, T, 393) 
    continuously down into analytical Logits.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Feature Map
        self.encoder = VisualTransformerEncoder(input_dim=393, d_model=256)
        
        # 2. Sequential Summarization
        self.pooling = AttentionPooling(embed_dim=256)
        
        # 3. Categorization
        self.classifier = VisualClassifier(input_dim=256, num_classes=2)
        
        logger.info("Successfully assembled unified VisualModel topology.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes complete Visual graph transitions end-to-end natively.
        
        Args:
            x: Input Temporal Standardized Vector constraints -> (B, T, 393)
            
        Returns:
            Classification diagnostic evaluations -> (B, 2)
        """
        encoded = self.encoder(x)
        pooled = self.pooling(encoded)
        logits = self.classifier(pooled)
        return logits
