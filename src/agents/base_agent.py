"""
Base Agent Class
=================

Abstract base class defining the interface for all RL agents.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for RL agents"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize base agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        self.training_step = 0
        self.episode_count = 0
        
        logger.info(
            f"Initialized {self.__class__.__name__} | "
            f"State dim: {state_dim} | Action dim: {action_dim} | Device: {device}"
        )
    
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Any:
        """
        Select an action given a state
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
        
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def learn(self, batch: Any) -> Dict[str, float]:
        """
        Update agent from a batch of experience
        
        Args:
            batch: Batch of transitions
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state to disk
        
        Args:
            path: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from disk
        
        Args:
            path: Path to load checkpoint from
        """
        pass
    
    def to_device(self, *tensors):
        """Move tensors to agent's device"""
        return [t.to(self.device) if t is not None else None for t in tensors]
    
    def train_mode(self) -> None:
        """Set agent to training mode"""
        pass
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode"""
        pass
    
    def update_training_step(self) -> None:
        """Increment training step counter"""
        self.training_step += 1
    
    def update_episode_count(self) -> None:
        """Increment episode counter"""
        self.episode_count += 1
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get agent state for checkpointing
        
        Returns:
            Dictionary containing agent state
        """
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load agent state from dictionary
        
        Args:
            state_dict: Dictionary containing agent state
        """
        self.training_step = state_dict.get('training_step', 0)
        self.episode_count = state_dict.get('episode_count', 0)