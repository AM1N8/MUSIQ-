"""
Trainer for Continuous Action Space Algorithms
===============================================

Implements training loop for SAC agent.
"""

import numpy as np
from typing import Dict
from loguru import logger

from .base_trainer import BaseTrainer
from ..agents.sac_agent import SACAgent


class SACTrainer(BaseTrainer):
    """Trainer for SAC agent"""
    
    def train_episode(self) -> Dict[str, float]:
        """Train SAC for one episode"""
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
        """Evaluate SAC agent"""
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


def create_continuous_trainer(algorithm: str, agent, env, config, experiment_name=None):
    """Create appropriate trainer based on algorithm"""
    if algorithm in ["sac", "cql"]:
        return SACTrainer(agent, env, config, experiment_name)
    else:
        raise ValueError(f"Unknown continuous algorithm: {algorithm}")