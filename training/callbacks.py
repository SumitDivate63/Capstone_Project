"""Training callbacks like early stopping."""

class EarlyStopping:
    """Early stopping callback."""
    def __init__(self, patience: int = 5):
        self.patience = patience
        
    def step(self, metric: float) -> bool:
        pass
