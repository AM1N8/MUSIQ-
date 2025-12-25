"""
Trainers for Discrete Action Space Algorithms
==============================================

Implements training loops for DQN and A2C agents.
"""

import numpy as np
from typing import Dict
from loguru import logger

from .base_trainer import BaseTrainer
from ..agents.dqn_agent import DQNAgent
from ..agents.a2c_agent import A2CAgent


class DQNTrainer(BaseTrainer):
    """Trainer for DQN agent"""
    
    def train_episode(self) -> Dict[str, float]:
        """Train DQN for one episode"""
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        training_metrics = {}
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            action = self.agent.act(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            # Learn (if enough samples)
            if self.total_steps >= self.config.training.warmup_steps:
                metrics = self.agent.learn()
                if metrics:
                    training_metrics = metrics
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **training_metrics
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate DQN agent"""
        self.agent.eval_mode()
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = self.agent.act(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        self.agent.train_mode()
        
        mean_reward = np.mean(eval_rewards)
        self.eval_rewards.append(mean_reward)
        
        return {
            'mean_reward': mean_reward,
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }


class A2CTrainer(BaseTrainer):
    """Trainer for A2C agent"""
    
    def train_episode(self) -> Dict[str, float]:
        """Train A2C for one episode"""
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        training_metrics = {}
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            action, value, log_prob = self.agent.act(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition in rollout buffer
            self.agent.store_transition(
                state, action, reward, value, log_prob, terminated or truncated
            )
            
            # Learn every n_steps
            if (episode_length + 1) % self.agent.n_steps == 0:
                metrics = self.agent.learn()
                if metrics:
                    training_metrics = metrics
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
        
        # Learn at end of episode if buffer has data
        if len(self.agent.rollout_buffer) > 0:
            metrics = self.agent.learn()
            if metrics:
                training_metrics = metrics
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **training_metrics
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate A2C agent"""
        self.agent.eval_mode()
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _, _ = self.agent.act(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        self.agent.train_mode()
        
        mean_reward = np.mean(eval_rewards)
        self.eval_rewards.append(mean_reward)
        
        return {
            'mean_reward': mean_reward,
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }


# Factory function
def create_discrete_trainer(algorithm: str, agent, env, config, experiment_name=None):
    """Create appropriate trainer based on algorithm"""
    if algorithm == "dqn":
        return DQNTrainer(agent, env, config, experiment_name)
    elif algorithm == "a2c":
        return A2CTrainer(agent, env, config, experiment_name)
    else:
        raise ValueError(f"Unknown discrete algorithm: {algorithm}")