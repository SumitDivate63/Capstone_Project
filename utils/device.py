"""GPU utility."""
import torch
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Detect CUDA, MPS, or fallback to CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device
