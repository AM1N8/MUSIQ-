"""Hyperparameter tuning with Optuna"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from src.config.base_config import Config, DQNConfig
from src.envs.music_env import MusicEnv
from src.agents.dqn_agent import DQNAgent
from src.trainers.discrete_trainer import DQNTrainer
from loguru import logger


def objective(trial, algorithm="dqn", data_path="data", n_episodes=500):
    """
    Optuna objective function for hyperparameter tuning
    
    Args:
        trial: Optuna trial object
        algorithm: Algorithm to tune
        data_path: Path to data
        n_episodes: Number of training episodes
    
    Returns:
        Mean evaluation reward
    """
    # Suggest hyperparameters
    if algorithm == "dqn":
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        target_update_freq = trial.suggest_int("target_update_freq", 500, 2000)
        hidden_dim1 = trial.suggest_categorical("hidden_dim1", [128, 256, 512])
        hidden_dim2 = trial.suggest_categorical("hidden_dim2", [64, 128, 256])
        
        # Create config
        config = Config()
        config.training.algorithm = "dqn"
        config.training.num_episodes = n_episodes
        config.training.eval_interval = 100
        config.env.data_path = data_path
        
        config.dqn = DQNConfig(
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            target_update_freq=target_update_freq
        )
        config.dqn.network_config.hidden_dims = [hidden_dim1, hidden_dim2, 64]
    
    # Create environment
    env = MusicEnv(data_path=data_path, mode="discrete")
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config.dqn.model_dump(),
        device="cuda"
    )
    
    # Create trainer
    trainer = DQNTrainer(agent, env, config, f"optuna_trial_{trial.number}")
    
    # Train
    try:
        trainer.train()
        
        # Return best evaluation reward
        if trainer.eval_rewards:
            return max(trainer.eval_rewards)
        else:
            return -np.inf
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -np.inf


def tune_hyperparameters(
    algorithm: str = "dqn",
    data_path: str = "data",
    n_trials: int = 50,
    n_episodes: int = 500,
    output_path: str = "hyperparameter_tuning_results.csv"
):
    """
    Run hyperparameter tuning
    
    Args:
        algorithm: Algorithm to tune
        data_path: Path to data
        n_trials: Number of Optuna trials
        n_episodes: Episodes per trial
        output_path: Path to save results
    """
    logger.info(f"Starting hyperparameter tuning for {algorithm}")
    logger.info(f"Trials: {n_trials}, Episodes per trial: {n_episodes}")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, algorithm, data_path, n_episodes),
        n_trials=n_trials
    )
    
    # Print results
    logger.success("=" * 60)
    logger.success("Hyperparameter Tuning Results")
    logger.success("=" * 60)
    logger.success(f"Best trial: {study.best_trial.number}")
    logger.success(f"Best value: {study.best_value:.4f}")
    logger.success("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.success(f"  {key}: {value}")
    logger.success("=" * 60)
    
    # Save results
    df = study.trials_dataframe()
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    
    return study.best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="dqn")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--output", default="tuning_results.csv")
    args = parser.parse_args()
    
    tune_hyperparameters(
        algorithm=args.algorithm,
        data_path=args.data_path,
        n_trials=args.n_trials,
        n_episodes=args.n_episodes,
        output_path=args.output
    )