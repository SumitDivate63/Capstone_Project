"""Helper utilities."""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

def create_dir(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def check_file(path: Path) -> bool:
    """Checks if a file exists."""
    return path.exists() and path.is_file()

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Path):
    with open(path, 'w') as f:
        yaml.dump(data, f)
