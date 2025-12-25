"""
Experience Replay Buffers
==========================

Implements standard and prioritized experience replay for off-policy algorithms.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from collections import deque
import random


class ReplayBuffer:
    """Standard experience replay buffer"""
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        device: str = "cpu"
    ):
        """
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space (1 for discrete)
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Preallocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return self.size >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer"""
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        device: str = "cpu",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for tensors
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.position = 0
        self.size = 0
        
        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Priority tree (sum tree for efficient sampling)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition with maximum priority"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # New transitions get max priority
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            batch_size = self.size
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Get transitions
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def _get_beta(self) -> float:
        """Anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class NStepReplayBuffer:
    """N-step replay buffer for multi-step returns"""
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        device: str = "cpu",
        n_step: int = 3,
        gamma: float = 0.99
    ):
        """
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            device: Device for tensors
            n_step: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        self.buffer = ReplayBuffer(capacity, state_dim, action_dim, device)
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition and compute n-step return"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Compute n-step return
        n_step_reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
        
        # Get first state and action, last next_state and done
        first_state, first_action = self.n_step_buffer[0][:2]
        last_next_state, last_done = self.n_step_buffer[-1][3:]
        
        # Add to main buffer
        self.buffer.add(first_state, first_action, n_step_reward, last_next_state, last_done)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from buffer"""
        return self.buffer.sample(batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return self.buffer.is_ready(batch_size)


class RolloutBuffer:
    """Buffer for on-policy algorithms (A2C, PPO)"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        """Initialize rollout buffer for on-policy learning"""
        self.capacity = capacity
        self.device = device
        self.clear()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Add transition to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all stored transitions"""
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).unsqueeze(1).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, values, log_probs, dones
    
    def clear(self) -> None:
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self) -> int:
        return len(self.states)
    
    def is_ready(self, min_size: int = 1) -> bool:
        return len(self) >= min_size