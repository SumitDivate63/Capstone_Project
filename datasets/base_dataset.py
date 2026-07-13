"""Base dataset class."""
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Abstract base dataset."""
    def __init__(self, config):
        self.config = config
        
    def __len__(self):
        return 0
        
    def __getitem__(self, idx):
        pass
