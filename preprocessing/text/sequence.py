import torch
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

def pad_sequence(token_ids: List[int], max_length: int, pad_id: int = 0) -> List[int]:
    """Pads or truncates inputs to rigid length architectures."""
    if len(token_ids) >= max_length:
        return token_ids[:max_length]
    return token_ids + [pad_id] * (max_length - len(token_ids))

def generate_attention_mask(token_ids: List[int], max_length: int) -> List[int]:
    """Creates binary masking mappings for active vectors."""
    length = min(len(token_ids), max_length)
    return [1] * length + [0] * (max_length - length)

def prepare_sequence(
    token_ids: List[int], 
    max_length: int, 
    pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assembles completely formatted PyTorch primitive targets natively.
    """
    sequence_length = min(len(token_ids), max_length)
    mask = generate_attention_mask(token_ids, max_length)
    padded_ids = pad_sequence(token_ids, max_length, pad_id)
    
    t_ids = torch.tensor(padded_ids, dtype=torch.long)
    t_mask = torch.tensor(mask, dtype=torch.long)
    t_len = torch.tensor([sequence_length], dtype=torch.long)
    
    return t_ids, t_mask, t_len
