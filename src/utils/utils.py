"""
Utility Functions
=================

Common utilities for logging, seeding, metrics, and visualization.
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import sys
import matplotlib.pyplot as plt
import json


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Setup loguru logging
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add file handler
    logger.add(
        log_dir / "training_{time}.log",
        rotation="10 MB",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
        colorize=True
    )
    
    logger.info(f"Logging initialized | Level: {log_level} | Dir: {log_dir}")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seed set to {seed} (deterministic mode)")
    else:
        logger.info(f"Seed set to {seed}")


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_average(self, key: str, window: int = 100) -> float:
        """Get moving average of metric"""
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        values = self.history[key][-window:]
        return float(np.mean(values))
    
    def get_latest(self, key: str) -> float:
        """Get latest value of metric"""
        if key not in self.history or len(self.history[key]) == 0:
            return 0.0
        return self.history[key][-1]
    
    def save(self, path: str) -> None:
        """Save metrics history"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved metrics to {path}")
    
    def load(self, path: str) -> None:
        """Load metrics history"""
        with open(path, 'r') as f:
            self.history = json.load(f)
        logger.info(f"Loaded metrics from {path}")


def plot_training_curves(
    metrics_path: str,
    output_path: str,
    metrics_to_plot: Optional[List[str]] = None
) -> None:
    """
    Plot training curves from metrics file
    
    Args:
        metrics_path: Path to metrics JSON file
        output_path: Path to save plot
        metrics_to_plot: List of metric names to plot
    """
    with open(metrics_path, 'r') as f:
        history = json.load(f)
    
    if metrics_to_plot is None:
        metrics_to_plot = ['episode_reward', 'loss']
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in history]
    
    if not available_metrics:
        logger.warning(f"No metrics found to plot from {metrics_to_plot}")
        return
    
    # Create subplots
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4 * len(available_metrics)))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        values = history[metric]
        ax.plot(values, alpha=0.3)
        
        # Smooth curve
        if len(values) > 10:
            window = min(100, len(values) // 10)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(values)), smoothed, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {output_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get compute device
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def save_config_as_yaml(config, path: str) -> None:
    """Save Pydantic config to YAML file"""
    config.to_yaml(path)
    logger.info(f"Saved config to {path}")


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create directory structure for experiment
    
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / experiment_name
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    (exp_dir / "configs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    
    return exp_dir


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score (higher is better)
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute discounted returns
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        normalize: Whether to normalize returns
    
    Returns:
        Array of discounted returns
    """
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = np.array(returns)
    
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_system_info() -> None:
    """Print system information"""
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info("=" * 60)