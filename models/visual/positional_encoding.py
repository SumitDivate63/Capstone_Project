import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding to inject temporal 
    order awareness into the encoder inputs natively without learnable parameters.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding matrix.

        Args:
            d_model: The embedded dimensionality (expected output dimension).
            max_len: The maximum chronological sequence length to compute frequencies for.
        """
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (1, Seq_Len, d_model) matching batch_first tensors
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings dynamically bounded to the sequence input length.

        Args:
            x: Input tensor of shape (Batch_Size, Sequence_Length, Embedding_Dimension).

        Returns:
            Tensor identical in shape with synchronized temporal oscillations injected.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
