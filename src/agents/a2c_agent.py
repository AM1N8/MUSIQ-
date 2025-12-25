"""
A2C Agent Implementations
==================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from loguru import logger

from .base_agent import BaseAgent
from ..networks.networks import create_a2c_network, create_sac_networks
from ..replay_buffers.replay_buffer import RolloutBuffer, ReplayBuffer


# ============= A2C Agent =============

class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C) agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        network_config: Dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # Network
        network_config = network_config or {}
        self.network = create_a2c_network(
            state_dim, action_dim, network_config
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(10000, state_dim, action_dim, device)
        
        logger.success(f"A2C Agent initialized | n_steps: {n_steps}")
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action from policy
        
        Returns:
            action, value, log_prob
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.network(state_tensor)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            if training:
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)
            
            log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob.item()
    
    def learn(self, batch: Any = None) -> Dict[str, float]:
        """Update policy and value function"""
        if not self.rollout_buffer.is_ready(self.n_steps):
            return {}
        
        # Get rollout data
        states, actions, rewards, values, old_log_probs, dones = self.rollout_buffer.get()
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, values, dones)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        logits, new_values = self.network(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        new_log_probs = dist.log_prob(actions.squeeze())
        entropy = dist.entropy()
        
        # Policy loss
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        self.update_training_step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'advantage_mean': advantages.mean().item()
        }
    
    def _compute_returns(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Store transition in rollout buffer"""
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
    
    def train_mode(self) -> None:
        self.network.train()
    
    def eval_mode(self) -> None:
        self.network.eval()
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            **self.get_state_dict()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved A2C checkpoint to {path}")
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.load_state_dict(checkpoint)
        logger.info(f"Loaded A2C checkpoint from {path}")


