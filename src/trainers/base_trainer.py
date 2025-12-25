"""
Base Trainer Class
===================

Abstract base class for training RL agents with logging and checkpointing.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import time
import json


class BaseTrainer(ABC):
    """Base trainer for RL agents"""
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        config: Any,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer
        
        Args:
            agent: RL agent to train
            env: Training environment
            config: Training configuration
            experiment_name: Name for this experiment
        """
        self.agent = agent
        self.env = env
        self.config = config
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{config.training.algorithm}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Setup directories
        self.checkpoint_dir = Path(config.training.checkpoint_dir) / self.experiment_name
        self.log_dir = Path(config.training.log_dir) / self.experiment_name
        self.tensorboard_dir = Path(config.training.tensorboard_dir) / self.experiment_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        
        # Weights & Biases
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.training.wandb_project,
                    name=self.experiment_name,
                    config=config.model_dump()
                )
                self.wandb = wandb
                logger.info("Weights & Biases logging enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_reward = -np.inf
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
        logger.success(f"Trainer initialized | Experiment: {self.experiment_name}")
    
    @abstractmethod
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode
        
        Returns:
            Dictionary of episode metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def train(self) -> None:
        """Main training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        num_episodes = self.config.training.num_episodes
        eval_interval = self.config.training.eval_interval
        save_interval = self.config.training.save_interval
        log_interval = self.config.training.log_interval
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Train episode
            train_metrics = self.train_episode()
            
            # Log training metrics
            if episode % log_interval == 0:
                self._log_metrics(train_metrics, "train")
                self._print_progress(train_metrics)
            
            # Evaluate
            if episode % eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate(self.config.training.eval_episodes)
                self._log_metrics(eval_metrics, "eval")
                self._print_evaluation(eval_metrics)
                
                # Save best model
                mean_eval_reward = eval_metrics.get('mean_reward', -np.inf)
                if mean_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = mean_eval_reward
                    self.save_checkpoint("best_model.pt")
                    logger.success(f"New best model! Reward: {mean_eval_reward:.2f}")
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pt")
        
        # Final evaluation
        logger.info("Training complete! Running final evaluation...")
        final_eval = self.evaluate(self.config.training.eval_episodes)
        self._log_metrics(final_eval, "final_eval")
        self._print_evaluation(final_eval)
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Training summary
        elapsed_time = time.time() - start_time
        self._save_training_summary(elapsed_time)
        
        # Cleanup
        self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        
        logger.success(f"Training completed in {elapsed_time/3600:.2f} hours")
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        """Log metrics to TensorBoard and W&B"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # TensorBoard
                self.writer.add_scalar(f"{prefix}/{key}", value, self.episode)
                
                # Weights & Biases
                if self.use_wandb:
                    self.wandb.log({f"{prefix}/{key}": value, "episode": self.episode})
    
    def _print_progress(self, metrics: Dict[str, float]) -> None:
        """Print training progress"""
        episode_reward = metrics.get('episode_reward', 0)
        episode_length = metrics.get('episode_length', 0)
        loss = metrics.get('loss', 0)
        
        logger.info(
            f"Episode {self.episode:5d} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Length: {episode_length:4d} | "
            f"Loss: {loss:7.4f} | "
            f"Steps: {self.total_steps:7d}"
        )
    
    def _print_evaluation(self, metrics: Dict[str, float]) -> None:
        """Print evaluation results"""
        mean_reward = metrics.get('mean_reward', 0)
        std_reward = metrics.get('std_reward', 0)
        mean_length = metrics.get('mean_length', 0)
        
        logger.info(
            f"Evaluation | Episode {self.episode:5d} | "
            f"Mean Reward: {mean_reward:7.2f} ± {std_reward:5.2f} | "
            f"Mean Length: {mean_length:6.1f}"
        )
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save agent
        agent_path = checkpoint_path.with_suffix('.agent.pt')
        self.agent.save(str(agent_path))
        
        # Save trainer state
        trainer_state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_eval_reward': self.best_eval_reward,
            'episode_rewards': self.episode_rewards[-1000:],  # Keep last 1000
            'episode_lengths': self.episode_lengths[-1000:],
            'eval_rewards': self.eval_rewards,
        }
        
        trainer_path = checkpoint_path.with_suffix('.trainer.pt')
        torch.save(trainer_state, trainer_path)
        
        logger.debug(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        # Load agent
        agent_path = checkpoint_path.with_suffix('.agent.pt')
        self.agent.load(str(agent_path))
        
        # Load trainer state
        trainer_path = checkpoint_path.with_suffix('.trainer.pt')
        if trainer_path.exists():
            trainer_state = torch.load(trainer_path)
            self.episode = trainer_state['episode']
            self.total_steps = trainer_state['total_steps']
            self.best_eval_reward = trainer_state['best_eval_reward']
            self.episode_rewards = trainer_state['episode_rewards']
            self.episode_lengths = trainer_state['episode_lengths']
            self.eval_rewards = trainer_state['eval_rewards']
        
        logger.info(f"Loaded checkpoint: {filename}")
    
    def _save_training_summary(self, elapsed_time: float) -> None:
        """Save training summary to JSON"""
        summary = {
            'experiment_name': self.experiment_name,
            'algorithm': self.config.training.algorithm,
            'total_episodes': self.episode,
            'total_steps': self.total_steps,
            'elapsed_time_hours': elapsed_time / 3600,
            'best_eval_reward': float(self.best_eval_reward),
            'final_eval_rewards': self.eval_rewards[-5:] if self.eval_rewards else [],
            'mean_episode_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            'config': self.config.model_dump()
        }
        
        summary_path = self.log_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved training summary to {summary_path}")