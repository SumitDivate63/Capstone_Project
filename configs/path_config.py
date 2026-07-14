"""Path configuration for the Multimodal Depression Detection project."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class PathConfig:
    """Centralized path configuration."""

    # ----------------------------
    # Project Root
    # ----------------------------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # ----------------------------
    # Dataset Root
    # ----------------------------
    dataset_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DAIC_WOZ_ROOT",
                "/home/rit/Desktop/Capstone_G2/DAIC-WOZ"
            )
        )
    )

    # ----------------------------
    # Data Directories
    # ----------------------------
    data_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    # ----------------------------
    # Output Directories
    # ----------------------------
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    figure_dir: Path = field(init=False)
    prediction_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.metadata_dir = self.data_dir / "metadata"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"

        self.output_dir = self.project_root / "outputs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.figure_dir = self.output_dir / "figures"
        self.prediction_dir = self.output_dir / "predictions"

    def create_directories(self):
        """Create required project directories if they do not exist."""
        directories = [
            self.metadata_dir,
            self.processed_dir,
            self.cache_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.figure_dir,
            self.prediction_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
