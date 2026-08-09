"""
Text Agent Model for DAIC-WOZ Depression Detection.

Architecture:
    Input: token_ids (B, L) + attention_mask (B, L)
    → Embedding(vocab_size, d_model) + PositionalEncoding
    → Transformer Encoder (d=128, layers=2, heads=4)
    → Masked Attention Pooling (ignores padding)
    → MLP Classifier (128 → 64 → 2)

Uses the EXISTING TextPreprocessingPipeline output format:
    - token_ids: LongTensor(B, 512)
    - attention_mask: LongTensor(B, 512)  [1=real, 0=padding]

Text is participant-level — one sequence per participant (no windowing),
so participant-level prediction requires no aggregation.

VRAM estimate at batch=8: ~450 MB (fp32). Safe for simultaneous training
with the audio agent on GTX 1650 4 GB.
"""

import torch
import torch.nn as nn
from models.visual.positional_encoding import PositionalEncoding
from utils.logger import get_logger

logger = get_logger(__name__)


class MaskedAttentionPooling(nn.Module):
    """
    Attention pooling that respects the padding mask.
    Padding positions (mask=0) receive -inf before softmax so they
    contribute zero weight to the pooled representation.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1, bias=False)
        logger.info(f"MaskedAttentionPooling: d_model={d_model}")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:               (B, L, d_model) encoder output
            attention_mask:  (B, L) binary mask — 1 for real tokens, 0 for padding
        Returns:
            (B, d_model) pooled representation
        """
        scores = self.attention_weights(x)  # (B, L, 1)

        if attention_mask is not None:
            # Mask padding positions to -inf before softmax
            mask_expanded = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        weights = torch.softmax(scores, dim=1)   # (B, L, 1)
        pooled  = torch.sum(weights * x, dim=1)  # (B, d_model)
        return pooled


class TextTransformerEncoder(nn.Module):
    """
    Transformer encoder operating on token embeddings for clinical transcript modeling.
    Smaller than the visual encoder to conserve VRAM during parallel training.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 128,
        max_seq_len: int = 512,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size:      Size of the custom vocabulary (from TextPreprocessingPipeline).
            d_model:         Embedding and Transformer hidden dimension.
            max_seq_len:     Maximum sequence length (should match pipeline config).
            num_layers:      Number of Transformer encoder layers.
            nhead:           Attention heads (must divide d_model).
            dim_feedforward: FFN hidden dimension.
            dropout:         Dropout probability.
        """
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        logger.info(
            f"TextTransformerEncoder: vocab={vocab_size}, d={d_model}, "
            f"maxlen={max_seq_len}, layers={num_layers}, heads={nhead}"
        )

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_ids:      (B, L) LongTensor
            attention_mask: (B, L) binary mask — 0 for padding positions
        Returns:
            (B, L, d_model) contextual token representations
        """
        # Convert attention_mask (1=real, 0=pad) to src_key_padding_mask (True=mask/ignore)
        src_padding_mask = None
        if attention_mask is not None:
            src_padding_mask = (attention_mask == 0)  # (B, L), True = ignored

        x = self.embedding(token_ids)         # (B, L, d_model)
        x = self.pos_encoder(x)               # add positional encoding
        out = self.transformer_encoder(x, src_key_padding_mask=src_padding_mask)
        return out


class TextClassifier(nn.Module):
    """MLP classification head."""

    def __init__(self, input_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        logger.info(f"TextClassifier: {input_dim}→64→{num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TextModel(nn.Module):
    """
    Complete Text Agent for DAIC-WOZ depression classification.

    Pipeline:
        token_ids/attention_mask → TextTransformerEncoder
        → MaskedAttentionPooling → TextClassifier → (B, 2) logits

    The pooled embedding (B, d_model) is accessible via get_embedding()
    for use in later multimodal fusion.

    Text is participant-level: one sequence (512 tokens) per participant.
    No window aggregation is needed — each forward pass IS one participant.
    """

    def __init__(self, vocab_size: int = 30000, d_model: int = 128, max_seq_len: int = 512):
        """
        Args:
            vocab_size:  Must match the vocabulary built by TextPreprocessingPipeline.
            d_model:     Embedding dimension.
            max_seq_len: Must match TextPreprocessingConfig.max_sequence_length.
        """
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        self.encoder    = TextTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
        )
        self.pooling    = MaskedAttentionPooling(d_model=d_model)
        self.classifier = TextClassifier(input_dim=d_model, num_classes=2)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"TextModel assembled. Total parameters: {total_params:,}")

    def get_embedding(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the pooled d_model-dimensional embedding before the classifier.
        Used for multimodal fusion.
        """
        encoded = self.encoder(token_ids, attention_mask)
        return self.pooling(encoded, attention_mask)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids:      (B, L) LongTensor
            attention_mask: (B, L) binary mask
        Returns:
            (B, 2) raw logits
        """
        embedding = self.get_embedding(token_ids, attention_mask)
        return self.classifier(embedding)
