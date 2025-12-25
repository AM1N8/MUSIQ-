import yaml
from pathlib import Path
import torch
import numpy as np
from loguru import logger

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state: dict, filepath: str):
    """Save model checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, device: str = "cpu") -> dict:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    return (features - mean) / std