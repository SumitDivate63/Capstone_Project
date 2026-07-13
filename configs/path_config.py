"""Path configuration dataclasses."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    """Directory and file paths."""
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
