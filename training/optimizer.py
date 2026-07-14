import torch
import torch.nn as nn

def create_optimizer(model: nn.Module, learning_rate: float = 3e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Instantiates the structural AdamW optimizer bindings scaling against the designated visual topologies natively.
    """
    return torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
