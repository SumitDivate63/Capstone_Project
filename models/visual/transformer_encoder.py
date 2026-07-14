import torch
import torch.nn as nn
from typing import Optional

from models.visual.positional_encoding import PositionalEncoding
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualTransformerEncoder(nn.Module):
    """
    Reusable Temporal Transformer Encoder tailored specially for continuous numerical 
    visual descriptors from DAIC-WOZ preprocessing bounds.
    """

    def __init__(
        self,
        input_dim: int = 393,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """
        Construct the Visual Transformer sequence topology mapping directly from sequence matrices.

        Args:
            input_dim: Raw visual feature dimensional count (standard 393 for CLNF OpenFace array).
            d_model: Target internal embedding size.
            num_layers: Count of sequential Transformer blocks.
            nhead: Distinct attention mapping paths inside standard Multi-Head formulations.
            dim_feedforward: Expanding dimensional node-count applied natively within Transformer chunks.
            dropout: Probability factor blocking neuron dependencies avoiding over-fitting states.
            max_len: Safe constraint boundary mapping maximal temporal sliding lengths natively.
        """
        super().__init__()

        # Linear expansion mapping standardized sequences seamlessly down into isolated embedding spans
        self.input_projection = nn.Linear(input_dim, d_model)

        # Standard sequential index anchoring
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Unified architectural module defining single pass encoder topology
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        logger.info(
            f"Initialized VisualTransformerEncoder -> (Projection: {input_dim}->{d_model}, "
            f"Layers: {num_layers}, Heads: {nhead})"
        )

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes sequence projection seamlessly avoiding loops mapping sequences end-to-end natively.

        Args:
            x: Clean continuous PyTorch temporal visual output array (Batch, Window_Size, Feature_Dimension)
            src_key_padding_mask: Optional native masking padding limits to ignore during transformer aggregation.

        Returns:
            Torch structural mapping corresponding purely over hidden spaces -> (Batch, Window_Size, Embedding_Dimension)
        """
        # (B, T, 393) -> (B, T, 256)
        x = self.input_projection(x)

        # Add Temporal Order Signatures
        x = self.pos_encoder(x)

        # Calculate pure Transformer representation boundaries natively outputting mapping matrices
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return out
