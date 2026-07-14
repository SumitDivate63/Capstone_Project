"""
DataLoader module for Explainable Multimodal Depression Detection.
Provides loaders that yield cleanly batched dictionaries.
"""

from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader

from datasets.daic_dataset import DAICDataset


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch dictionaries without converting DataFrames to tensors.
    
    Transforms a list of dictionaries (each representing one participant)
    into a dictionary of lists. Recursively handles nested dictionaries
    (e.g., visual, audio, text components).

    Args:
        batch: List of dictionaries yielded by DAICDataset.

    Returns:
        A dictionary containing batched lists for each key/modality.
    """
    if not batch:
        return {}

    collated = {}
    keys = batch[0].keys()

    for key in keys:
        if isinstance(batch[0][key], dict):
            # Recurse one layer deep for modalities (visual, audio, text, labels, metadata)
            collated[key] = {
                sub_key: [item[key][sub_key] for item in batch]
                for sub_key in batch[0][key].keys()
            }
        else:
            collated[key] = [item[key] for item in batch]

    return collated


def build_dataset(
    split: str,
    load_visual: bool = True,
    load_audio: bool = True,
    load_text: bool = True
) -> DAICDataset:
    """
    Helper to instantiate DAICDataset.

    Args:
        split: Dataset split ('train', 'dev', 'test', 'all').
        load_visual: Whether to load visual data.
        load_audio: Whether to load audio data.
        load_text: Whether to load text data.

    Returns:
        Instantiated DAICDataset.
    """
    return DAICDataset(
        split=split,
        load_visual=load_visual,
        load_audio=load_audio,
        load_text=load_text
    )


def get_train_dataloader(
    batch_size: int = 8,
    shuffle: bool = True,
    load_visual: bool = True,
    load_audio: bool = True,
    load_text: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Get DataLoader for the training split.
    """
    dataset = build_dataset(
        split="train",
        load_visual=load_visual,
        load_audio=load_audio,
        load_text=load_text
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )


def get_dev_dataloader(
    batch_size: int = 8,
    shuffle: bool = False,
    load_visual: bool = True,
    load_audio: bool = True,
    load_text: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Get DataLoader for the dev/validation split.
    """
    dataset = build_dataset(
        split="dev",
        load_visual=load_visual,
        load_audio=load_audio,
        load_text=load_text
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )


def get_full_dataloader(
    batch_size: int = 8,
    shuffle: bool = False,
    load_visual: bool = True,
    load_audio: bool = True,
    load_text: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Get DataLoader for the entire dataset without filtering by split.
    """
    dataset = build_dataset(
        split="all",
        load_visual=load_visual,
        load_audio=load_audio,
        load_text=load_text
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
