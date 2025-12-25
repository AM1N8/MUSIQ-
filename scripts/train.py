"""Standalone training script with simplified interface"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.config.base_config import Config, DEFAULT_DQN_CONFIG
from src.envs.music_env import MusicEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.sac_agent import SACAgent
from src.trainers.discrete_trainer import create_discrete_trainer
from src.trainers.continuous_trainer import create_continuous_trainer
from src.utils.utils import setup_logging, set_seed, get_device
from loguru import logger


def train_agent(
    algorithm: str = "dqn",
    data_path: str = "data",
    num_episodes: int = 1000,
    config_path: str = None,
    checkpoint_path: str = None
):
    """
    Train an RL agent
    
    Args:
        algorithm: Algorithm to use (dqn, a2c, sac)
        data_path: Path to data directory
        num_episodes: Number of training episodes
        config_path: Path to config file (optional)
        checkpoint_path: Path to checkpoint to resume from (optional)
    """
    # Load config
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = DEFAULT_DQN_CONFIG
        config.training.algorithm = algorithm
    
    config.env.data_path = data_path
    config.training.num_episodes = num_episodes
    
    # Setup
    setup_logging("logs", "INFO")
    set_seed(config.training.seed)
    device = get_device(config.training.device == "cuda")
    
    # Create environment
    env = MusicEnv(
        data_path=config.env.data_path,
        mode=config.env.mode,
        **config.env.model_dump()
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if config.env.mode == "discrete" else env.action_space.shape[0]
    
    # Create agent
    agent_config = config.get_agent_config()
    if algorithm == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config.model_dump(),
            device=str(device)
        )
    elif algorithm == "a2c":
        agent = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config.model_dump(),
            device=str(device)
        )
    elif algorithm == "sac":
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config.model_dump(),
            device=str(device)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create trainer
    if algorithm in ["dqn", "a2c"]:
        trainer = create_discrete_trainer(algorithm, agent, env, config)
    else:
        trainer = create_continuous_trainer(algorithm, agent, env, config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    logger.info(f"Starting {algorithm.upper()} training...")
    trainer.train()
    logger.success("Training completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "a2c", "sac"])
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    
    train_agent(
        algorithm=args.algorithm,
        num_episodes=args.num_episodes,
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )