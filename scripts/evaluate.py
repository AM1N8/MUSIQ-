"""Detailed evaluation script with metrics analysis"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.base_config import Config
from src.envs.music_env import MusicEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.sac_agent import SACAgent
from loguru import logger


def evaluate_agent(
    config_path: str,
    checkpoint_path: str,
    num_episodes: int = 100,
    output_path: str = None
):
    """
    Evaluate trained agent with detailed metrics
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to agent checkpoint
        num_episodes: Number of evaluation episodes
        output_path: Path to save results (optional)
    """
    # Load config
    config = Config.from_yaml(config_path)
    
    # Create environment
    env = MusicEnv(
        data_path=config.env.data_path,
        mode=config.env.mode,
        **config.env.model_dump()
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if config.env.mode == "discrete" else env.action_space.shape[0]
    
    # Create agent
    algorithm = config.training.algorithm
    agent_config = config.get_agent_config()
    
    if algorithm == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    elif algorithm == "a2c":
        agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    elif algorithm == "sac":
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
    
    # Load checkpoint
    agent.load(checkpoint_path)
    agent.eval_mode()
    
    # Evaluate
    logger.info(f"Evaluating for {num_episodes} episodes...")
    
    results = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        reactions = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if algorithm == "a2c":
                action, _, _ = agent.act(state, training=False)
            else:
                action = agent.act(state, training=False)
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            reactions.append(info['user_reaction'])
        
        skip_rate = reactions.count(0) / len(reactions)
        partial_rate = reactions.count(1) / len(reactions)
        complete_rate = reactions.count(2) / len(reactions)
        
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'skip_rate': skip_rate,
            'partial_rate': partial_rate,
            'complete_rate': complete_rate
        })
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    # Compute statistics
    df = pd.DataFrame(results)
    
    summary = {
        'mean_reward': df['reward'].mean(),
        'std_reward': df['reward'].std(),
        'min_reward': df['reward'].min(),
        'max_reward': df['reward'].max(),
        'mean_length': df['length'].mean(),
        'mean_skip_rate': df['skip_rate'].mean(),
        'mean_partial_rate': df['partial_rate'].mean(),
        'mean_complete_rate': df['complete_rate'].mean()
    }
    
    # Print summary
    logger.success("=" * 60)
    logger.success("Evaluation Summary")
    logger.success("=" * 60)
    for key, value in summary.items():
        logger.success(f"{key}: {value:.4f}")
    logger.success("=" * 60)
    
    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output", default="evaluation_results.csv")
    args = parser.parse_args()
    
    evaluate_agent(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_path=args.output
    )