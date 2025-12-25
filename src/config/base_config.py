"""
Base Configuration Classes using Pydantic
==========================================

Provides validated configuration dataclasses for the RL training pipeline.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml


class EnvConfig(BaseModel):
    """Environment configuration"""
    data_path: str = Field(default="data", description="Path to data directory")
    mode: str = Field(default="discrete", description="Action space mode: discrete or continuous")
    max_session_length: int = Field(default=20, ge=1, le=100)
    num_recent_tracks: int = Field(default=5, ge=1, le=20)
    user_embed_dim: int = Field(default=32, ge=8, le=512)
    song_embed_dim: int = Field(default=8, ge=4, le=128)
    
    # Reward weights
    reward_full_listen: float = Field(default=1.0)
    reward_partial_listen: float = Field(default=0.5)
    reward_skip: float = Field(default=-0.5)
    reward_diversity_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    reward_personalization_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    reward_session_bonus_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # User behavior
    skip_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    partial_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    dropout_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ['discrete', 'continuous']:
            raise ValueError("mode must be 'discrete' or 'continuous'")
        return v


class NetworkConfig(BaseModel):
    """Neural network architecture configuration"""
    hidden_dims: List[int] = Field(default=[256, 128, 64])
    activation: str = Field(default="relu")
    use_layer_norm: bool = Field(default=True)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_attention: bool = Field(default=True)
    attention_heads: int = Field(default=4, ge=1, le=16)
    
    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v: str) -> str:
        valid = ['relu', 'tanh', 'elu', 'gelu']
        if v not in valid:
            raise ValueError(f"activation must be one of {valid}")
        return v


class DQNConfig(BaseModel):
    """DQN agent configuration"""
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0)
    epsilon_decay: int = Field(default=50000, ge=100)
    target_update_freq: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=64, ge=1)
    buffer_size: int = Field(default=100000, ge=1000)
    use_double_dqn: bool = Field(default=True)
    use_dueling: bool = Field(default=True)
    network_config: NetworkConfig = Field(default_factory=NetworkConfig)


class A2CConfig(BaseModel):
    """A2C agent configuration"""
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    value_loss_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    n_steps: int = Field(default=5, ge=1, le=100)
    network_config: NetworkConfig = Field(default_factory=NetworkConfig)


class SACConfig(BaseModel):
    """SAC agent configuration"""
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    tau: float = Field(default=0.005, ge=0.0, le=1.0)
    alpha: float = Field(default=0.2, ge=0.0)
    automatic_entropy_tuning: bool = Field(default=True)
    target_update_interval: int = Field(default=1, ge=1)
    batch_size: int = Field(default=256, ge=1)
    buffer_size: int = Field(default=1000000, ge=1000)
    network_config: NetworkConfig = Field(default_factory=NetworkConfig)


class TrainingConfig(BaseModel):
    """Training configuration"""
    algorithm: str = Field(default="dqn")
    num_episodes: int = Field(default=10000, ge=1)
    eval_interval: int = Field(default=100, ge=1)
    eval_episodes: int = Field(default=10, ge=1)
    save_interval: int = Field(default=1000, ge=1)
    log_interval: int = Field(default=10, ge=1)
    
    # Optimization
    warmup_steps: int = Field(default=1000, ge=0)
    grad_clip: Optional[float] = Field(default=1.0, gt=0.0)
    
    # Paths
    checkpoint_dir: str = Field(default="checkpoints")
    log_dir: str = Field(default="logs")
    tensorboard_dir: str = Field(default="runs")
    
    # Experiment tracking
    experiment_name: Optional[str] = Field(default=None)
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default="music-rl")
    
    # Hardware
    device: str = Field(default="cuda")
    num_workers: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    
    @field_validator('algorithm')
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        valid = ['dqn', 'a2c', 'sac', 'cql', 'rainbow']
        if v not in valid:
            raise ValueError(f"algorithm must be one of {valid}")
        return v


class Config(BaseModel):
    """Master configuration combining all sub-configs"""
    env: EnvConfig = Field(default_factory=EnvConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dqn: Optional[DQNConfig] = Field(default=None)
    a2c: Optional[A2CConfig] = Field(default=None)
    sac: Optional[SACConfig] = Field(default=None)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def get_agent_config(self):
        """Get agent config based on training algorithm"""
        algo = self.training.algorithm
        if algo == "dqn":
            return self.dqn or DQNConfig()
        elif algo == "a2c":
            return self.a2c or A2CConfig()
        elif algo == "sac":
            return self.sac or SACConfig()
        elif algo == "cql":
            # CQL uses SAC config structure with added parameters
            return self.sac or SACConfig()
        else:
            raise ValueError(f"Unknown algorithm: {algo}")


# Default configurations
DEFAULT_DQN_CONFIG = Config(
    training=TrainingConfig(algorithm="dqn"),
    dqn=DQNConfig()
)

DEFAULT_A2C_CONFIG = Config(
    training=TrainingConfig(algorithm="a2c"),
    a2c=A2CConfig()
)

DEFAULT_SAC_CONFIG = Config(
    env=EnvConfig(mode="continuous"),
    training=TrainingConfig(algorithm="sac"),
    sac=SACConfig()
)