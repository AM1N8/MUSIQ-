"""
SAC Agent Implementations
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

# ============= SAC Agent =============

class SACAgent(BaseAgent):
    """Soft Actor-Critic (SAC) agent for continuous actions"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_update_interval: int = 1,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        network_config: Dict = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        network_config = network_config or {}
        self.policy, self.q1, self.q2 = create_sac_networks(
            state_dim, action_dim, network_config
        )
        self.policy = self.policy.to(self.device)
        self.q1 = self.q1.to(self.device)
        self.q2 = self.q2.to(self.device)
        
        # Target Q-networks
        self.q1_target = create_sac_networks(state_dim, action_dim, network_config)[1].to(self.device)
        self.q2_target = create_sac_networks(state_dim, action_dim, network_config)[2].to(self.device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
        
        logger.success(
            f"SAC Agent initialized | "
            f"Auto entropy: {automatic_entropy_tuning} | "
            f"Buffer size: {buffer_size:,}"
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, _ = self.policy.sample(state_tensor)
            else:
                mean, _ = self.policy(state_tensor)
                action = torch.tanh(mean)
            
            action = action.cpu().numpy()[0]
        
        return action
    
    def learn(self, batch: Any = None) -> Dict[str, float]:
        """Update policy and Q-functions"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_q_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        if self.training_step % self.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
        
        self.update_training_step()
        
        return {
            'loss': (q1_loss + q2_loss + policy_loss).item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_mode(self) -> None:
        self.policy.train()
        self.q1.train()
        self.q2.train()
    
    def eval_mode(self) -> None:
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'alpha': self.alpha,
            **self.get_state_dict()
        }
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved SAC checkpoint to {path}")
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.alpha = checkpoint['alpha']
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.load_state_dict(checkpoint)
        logger.info(f"Loaded SAC checkpoint from {path}")