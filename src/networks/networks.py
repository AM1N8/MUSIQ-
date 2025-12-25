"""
Neural Network Architectures for RL Agents
===========================================

Implements policy and value networks with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import math
from .sequential_network import SequentialStateEncoder


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for sequence modeling"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, embed_dim)
        """
        B, L, D = x.shape
        
        # Linear projections and split into heads
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        return out


class FeatureExtractor(nn.Module):
    """Extract features from user embedding and listening history"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Feature processing
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Optional attention for sequence modeling
        if use_attention:
            self.attention = MultiHeadAttention(prev_dim, num_heads, dropout)
        
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, output_dim)
        """
        features = self.feature_net(x)
        
        if self.use_attention:
            # Reshape for attention: (batch, 1, dim)
            features = features.unsqueeze(1)
            features = self.attention(features)
            features = features.squeeze(1)
        
        return features


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128, 64],
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            state_dim, hidden_dims, use_attention, num_heads, dropout
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            Q-values: (batch, action_dim)
        """
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for A2C"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128, 64],
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            state_dim, hidden_dims, use_attention, num_heads, dropout
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            logits: (batch, action_dim)
            value: (batch, 1)
        """
        features = self.feature_extractor(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous action spaces (SAC)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            state_dim, hidden_dims, use_attention, num_heads, dropout
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std_head = nn.Linear(feature_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
        """
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        
        # Log probability with change of variables
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob


class QNetwork(nn.Module):
    """Q-network for SAC (twin critics)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.q_net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
        Returns:
            Q-value: (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.q_net(x)


class SequentialQNetwork(nn.Module):
    """Q-network for SAC with sequential state input"""
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Sequential encoder
        self.encoder = SequentialStateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )
        
        # Q-value head (MLP)
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, seq_len, input_dim)
            action: (batch, action_dim)
        """
        # Encode sequence -> vector
        state_encoded = self.encoder(state)
        
        # Concatenate with action
        x = torch.cat([state_encoded, action], dim=1)
        
        return self.q_net(x)


# Factory functions
def create_dqn_network(state_dim: int, action_dim: int, config: dict) -> DuelingQNetwork:
    """Create DQN network from config"""
    return DuelingQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        use_attention=config.get('use_attention', True),
        num_heads=config.get('attention_heads', 4),
        dropout=config.get('dropout', 0.1)
    )


def create_a2c_network(state_dim: int, action_dim: int, config: dict) -> ActorCriticNetwork:
    """Create A2C network from config"""
    return ActorCriticNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        use_attention=config.get('use_attention', True),
        num_heads=config.get('attention_heads', 4),
        dropout=config.get('dropout', 0.1)
    )


def create_sac_networks(
    state_dim: int, 
    action_dim: int, 
    config: dict
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Create SAC networks (policy + twin critics)"""
    
    encoder_type = config.get('encoder_type', 'mlp')
    
    if encoder_type == 'transformer':
        # For transformer, state_dim is the feature dimension per step
        hidden_dim = config.get('hidden_dim', 128)
        
        # Policy with sequential encoder
        class SequentialPolicy(GaussianPolicy):
            def __init__(self, **kwargs):
                super(nn.Module, self).__init__()
                self.log_std_min = -20
                self.log_std_max = 2
                
                # Override feature extractor with sequential one
                self.feature_extractor = SequentialStateEncoder(
                    input_dim=state_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get('num_layers', 2),
                    nhead=config.get('nhead', 4),
                    dropout=config.get('dropout', 0.1)
                )
                feature_dim = self.feature_extractor.output_dim
                
                self.mean_head = nn.Linear(feature_dim, action_dim)
                self.log_std_head = nn.Linear(feature_dim, action_dim)
        
        policy = SequentialPolicy()
        
        q1 = SequentialQNetwork(state_dim, action_dim, hidden_dim)
        q2 = SequentialQNetwork(state_dim, action_dim, hidden_dim)
        
    else:
        # Standard MLP networks
        policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256]),
            use_attention=config.get('use_attention', True),
            num_heads=config.get('attention_heads', 4),
            dropout=config.get('dropout', 0.1)
        )
        
        q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256]),
            dropout=config.get('dropout', 0.1)
        )
        
        q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.get('hidden_dims', [256, 256]),
            dropout=config.get('dropout', 0.1)
        )
    
    return policy, q1, q2