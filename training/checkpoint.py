import torch
import torch.nn as nn
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(
    model: nn.Module, 
    epoch: int, 
    best_f1: float, 
    is_best: bool,
    optimizer: torch.optim.Optimizer = None,
    scheduler_state: dict = None,
    best_accuracy: float = -1.0,
    fold: int = 1,
    seed: int = 42,
    save_dir: str = "outputs/checkpoints/visual/"
) -> None:
    """
    Standardizing checkpoint bounds isolating best performance natively.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    state = {
        'fold': fold,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_f1': best_f1,
        'best_accuracy': best_accuracy,
        'seed': seed
    }
    
    # Store optimizer state dict if provided
    state['optimizer_state_dict'] = optimizer.state_dict() if optimizer else None
    state['scheduler_state_dict'] = scheduler_state
    
    last_path = path / "last_model.pt"
    torch.save(state, str(last_path))
    
    if is_best:
        best_path = path / "best_model.pt"
        torch.save(state, str(best_path))

def load_checkpoint(model: nn.Module, path_str: str) -> dict:
    """Retrieves checkpoints and validates weights scaling."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Safe checkpoint extraction failed. Not found: {path}")
        
    state = torch.load(str(path))
    model.load_state_dict(state['model_state_dict'])
    logger.info(f"Weight matrices bound matching Epoch {state['epoch']}, Memory State Restored.")
    
    return state
