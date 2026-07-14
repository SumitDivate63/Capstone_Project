import torch.nn as nn

def create_loss() -> nn.Module:
    """
    Returns the loss function utilized for binary depression classification spaces natively.
    """
    return nn.CrossEntropyLoss()
