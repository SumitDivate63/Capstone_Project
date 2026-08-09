"""
Audio Agent Model for DAIC-WOZ Depression Detection.

Architecture:
    Input: (B, T, F_dim) — sliding windows of COVAREP+FORMANT features
    → Linear projection + Positional Encoding
    → Transformer Encoder (d=128, layers=3, heads=4)
    → Attention Pooling (temporal)
    → MLP Classifier (128 → 64 → 2)

VRAM estimate at batch=16: ~500 MB (fp32). Safe for simultaneous training
with the text agent on GTX 1650 4 GB.

F_dim is dynamic (determined by the audio preprocessing pipeline at runtime),
so the model accepts input_dim as a constructor argument.
"""

import torch
import torch.nn as nn
from models.visual.positional_encoding import PositionalEncoding
from models.visual.attention_pooling import AttentionPooling
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder for COVAREP+FORMANT acoustic features.
    Mirrors the VisualTransformerEncoder pattern but with smaller d_model
    to fit within VRAM budget during simultaneous training.
    """

    def __init__(
        self,
        input_dim: int = 77,
        d_model: int = 128,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """
        Args:
            input_dim:       Feature dim from COVAREP+FORMANT concat (≈77).
            d_model:         Internal embedding dimension.
            num_layers:      Number of Transformer encoder layers.
            nhead:           Number of attention heads (must divide d_model).
            dim_feedforward: FFN hidden dimension.
            dropout:         Dropout probability.
            max_len:         Safety cap for positional encoding.
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        logger.info(
            f"AudioTransformerEncoder: {input_dim}→{d_model}, "
            f"layers={num_layers}, heads={nhead}, ff={dim_feedforward}"
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T, d_model)
        """
        x = self.input_projection(x)       # (B, T, input_dim) → (B, T, d_model)
        x = self.pos_encoder(x)            # add positional encoding
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out


class AudioClassifier(nn.Module):
    """MLP classification head for depression binary prediction."""

    def __init__(self, input_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        logger.info(f"AudioClassifier: {input_dim}→64→{num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model) pooled representation
        Returns:
            (B, num_classes) logits
        """
        return self.classifier(x)


class AudioModel(nn.Module):
    """
    Complete Audio Agent for DAIC-WOZ depression classification.

    Pipeline:
        (B, T, F_dim) →  AudioTransformerEncoder → AttentionPooling → AudioClassifier → (B, 2)

    The embedding of the pooled representation (B, d_model) is accessible via
    self.get_embedding(x) for use in later multimodal fusion.
    """

    def __init__(self, input_dim: int = 77, d_model: int = 128):
        """
        Args:
            input_dim: Acoustic feature dimension (set dynamically from pipeline).
            d_model:   Embedding dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.encoder  = AudioTransformerEncoder(input_dim=input_dim, d_model=d_model)
        self.pooling  = AttentionPooling(embed_dim=d_model)
        self.classifier = AudioClassifier(input_dim=d_model, num_classes=2)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"AudioModel assembled. Total parameters: {total_params:,}")

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the pooled d_model-dimensional embedding before the classifier.
        Used for multimodal fusion.

        Args:
            x: (B, T, input_dim) window tensor
        Returns:
            (B, d_model) embedding
        """
        encoded = self.encoder(x)
        return self.pooling(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — batch of audio windows
        Returns:
            (B, 2) raw logits
        """
        embedding = self.get_embedding(x)
        return self.classifier(embedding)
