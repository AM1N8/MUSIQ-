"""
Deep Q-Network (DQN) Agent
===========================

Implements DQN with Double DQN and Dueling architecture options.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from pathlib import Path
from loguru import logger

from .base_agent import BaseAgent
from ..networks.networks import create_dqn_network
from ..replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with Double DQN and Dueling options"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50000,
        target_update_freq: int = 1000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        use_double_dqn: bool = True,
        use_prioritized_replay: bool = False,
        network_config: Dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        
        # Networks
        network_config = network_config or {}
        self.q_network = create_dqn_network(
            state_dim, action_dim, network_config
        ).to(self.device)
        
        self.target_network = create_dqn_network(
            state_dim, action_dim, network_config
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, state_dim, 1, device
            )
            self.use_per = True
            logger.info("Using Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size, state_dim, 1, device
            )
            self.use_per = False
        
        logger.success(
            f"DQN Agent initialized | "
            f"Double DQN: {use_double_dqn} | "
            f"Buffer size: {buffer_size:,}"
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action (integer)
        """
        # Exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def learn(self, batch: Any = None) -> Dict[str, float]:
        """
        Update Q-network from replay buffer
        
        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones_like(rewards)
            indices = None
        
        actions = actions.long()
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select action
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in PER
        if self.use_per and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            self.training_step / self.epsilon_decay
        )
        
        self.update_training_step()
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value_mean': current_q_values.mean().item(),
            'q_value_max': current_q_values.max().item()
        }
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer"""
        action_arr = np.array([action], dtype=np.float32)
        self.replay_buffer.add(state, action_arr, reward, next_state, done)
    
    def train_mode(self) -> None:
        """Set networks to training mode"""
        self.q_network.train()
    
    def eval_mode(self) -> None:
        """Set networks to evaluation mode"""
        self.q_network.eval()
    
    def save(self, path: str) -> None:
        """Save agent checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            **self.get_state_dict()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved DQN checkpoint to {path}")
    
    def load(self, path: str) -> None:
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.load_state_dict(checkpoint)
        
        logger.info(f"Loaded DQN checkpoint from {path}")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get full agent state"""
        state_dict = super().get_state_dict()
        state_dict.update({
            'epsilon': self.epsilon,
        })
        return state_dict