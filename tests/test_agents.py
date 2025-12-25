"""
Unit Tests for RL Agents
=========================

Comprehensive test suite for agents, networks, and buffers.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.sac_agent import SACAgent
from src.replay_buffers.replay_buffer import (
    ReplayBuffer, PrioritizedReplayBuffer, 
    RolloutBuffer, NStepReplayBuffer
)
from src.networks.networks import (
    DuelingQNetwork, ActorCriticNetwork, 
    GaussianPolicy, QNetwork
)


class TestDQNAgent:
    """Tests for DQN agent"""
    
    @pytest.fixture
    def agent(self):
        """Create DQN agent for testing"""
        return DQNAgent(
            state_dim=10,
            action_dim=5,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32,
            device="cpu"
        )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.epsilon == 1.0
        assert len(agent.replay_buffer) == 0
    
    def test_act(self, agent):
        """Test action selection"""
        state = np.random.randn(10)
        
        # Training mode (exploration)
        action_train = agent.act(state, training=True)
        assert isinstance(action_train, int)
        assert 0 <= action_train < 5
        
        # Eval mode (exploitation)
        agent.epsilon = 0.0
        action_eval = agent.act(state, training=False)
        assert isinstance(action_eval, int)
    
    def test_store_transition(self, agent):
        """Test storing transitions"""
        state = np.random.randn(10)
        action = 2
        reward = 1.0
        next_state = np.random.randn(10)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
    
    def test_learn(self, agent):
        """Test learning from experience"""
        # Fill buffer with random transitions
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            agent.store_transition(state, action, reward, next_state, done)
        
        # Learn
        metrics = agent.learn()
        assert 'loss' in metrics
        assert 'epsilon' in metrics
        assert isinstance(metrics['loss'], float)
    
    def test_save_load(self, agent):
        """Test saving and loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "agent.pt"
            
            # Train a bit
            for _ in range(50):
                state = np.random.randn(10)
                action = agent.act(state)
                agent.store_transition(
                    state, action, 1.0, np.random.randn(10), False
                )
            agent.learn()
            
            # Save
            agent.save(str(save_path))
            assert save_path.exists()
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_dim=10, action_dim=5, device="cpu"
            )
            new_agent.load(str(save_path))
            
            # Check state matches
            assert new_agent.training_step == agent.training_step


class TestA2CAgent:
    """Tests for A2C agent"""
    
    @pytest.fixture
    def agent(self):
        """Create A2C agent for testing"""
        return A2CAgent(
            state_dim=10,
            action_dim=5,
            learning_rate=1e-3,
            gamma=0.99,
            n_steps=5,
            device="cpu"
        )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.n_steps == 5
    
    def test_act(self, agent):
        """Test action selection"""
        state = np.random.randn(10)
        action, value, log_prob = agent.act(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
        assert isinstance(value, float)
        assert isinstance(log_prob, float)
    
    def test_store_transition(self, agent):
        """Test storing transitions"""
        state = np.random.randn(10)
        action = 2
        reward = 1.0
        value = 0.5
        log_prob = -1.0
        done = False
        
        agent.store_transition(state, action, reward, value, log_prob, done)
        assert len(agent.rollout_buffer) == 1
    
    def test_learn(self, agent):
        """Test learning"""
        # Fill buffer
        for _ in range(10):
            state = np.random.randn(10)
            action, value, log_prob = agent.act(state)
            agent.store_transition(
                state, action, 1.0, value, log_prob, False
            )
        
        # Learn
        metrics = agent.learn()
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics


class TestSACAgent:
    """Tests for SAC agent"""
    
    @pytest.fixture
    def agent(self):
        """Create SAC agent for testing"""
        return SACAgent(
            state_dim=10,
            action_dim=3,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32,
            device="cpu"
        )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.alpha > 0
    
    def test_act(self, agent):
        """Test action selection"""
        state = np.random.randn(10)
        action = agent.act(state, training=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert np.all(action >= -1) and np.all(action <= 1)
    
    def test_store_transition(self, agent):
        """Test storing transitions"""
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = 1.0
        next_state = np.random.randn(10)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
    
    def test_learn(self, agent):
        """Test learning"""
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(10)
            action = agent.act(state)
            agent.store_transition(
                state, action, 1.0, np.random.randn(10), False
            )
        
        # Learn
        metrics = agent.learn()
        assert 'loss' in metrics
        assert 'q1_loss' in metrics
        assert 'q2_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'alpha' in metrics


class TestReplayBuffers:
    """Tests for replay buffers"""
    
    def test_replay_buffer(self):
        """Test standard replay buffer"""
        buffer = ReplayBuffer(capacity=100, state_dim=10, action_dim=1)
        
        # Add transitions
        for _ in range(50):
            state = np.random.randn(10)
            action = np.array([np.random.randint(0, 5)])
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        assert buffer.is_ready(32)
        
        # Sample batch
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (32, 10)
        assert actions.shape == (32, 1)
        assert rewards.shape == (32, 1)
    
    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer"""
        buffer = PrioritizedReplayBuffer(
            capacity=100, state_dim=10, action_dim=1, alpha=0.6
        )
        
        # Add transitions
        for _ in range(50):
            state = np.random.randn(10)
            action = np.array([np.random.randint(0, 5)])
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            buffer.add(state, action, reward, next_state, done)
        
        # Sample with priorities
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        assert weights.shape == (32, 1)
        assert len(indices) == 32
        
        # Update priorities
        new_priorities = np.random.rand(32) + 0.1
        buffer.update_priorities(indices, new_priorities)
    
    def test_rollout_buffer(self):
        """Test rollout buffer"""
        buffer = RolloutBuffer(capacity=100, state_dim=10, action_dim=1)
        
        # Add transitions
        for _ in range(10):
            state = np.random.randn(10)
            action = np.random.randint(0, 5)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = False
            buffer.add(state, action, reward, value, log_prob, done)
        
        assert len(buffer) == 10
        
        # Get all data
        batch = buffer.get()
        states, actions, rewards, values, log_probs, dones = batch
        
        assert states.shape == (10, 10)
        assert len(actions) == 10
        
        # Clear
        buffer.clear()
        assert len(buffer) == 0


class TestNetworks:
    """Tests for neural networks"""
    
    def test_dueling_q_network(self):
        """Test Dueling Q-Network"""
        net = DuelingQNetwork(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        state = torch.randn(4, 10)
        q_values = net(state)
        
        assert q_values.shape == (4, 5)
    
    def test_actor_critic_network(self):
        """Test Actor-Critic Network"""
        net = ActorCriticNetwork(
            state_dim=10,
            action_dim=5,
            hidden_dims=[64, 32]
        )
        
        state = torch.randn(4, 10)
        logits, value = net(state)
        
        assert logits.shape == (4, 5)
        assert value.shape == (4, 1)
    
    def test_gaussian_policy(self):
        """Test Gaussian Policy"""
        policy = GaussianPolicy(
            state_dim=10,
            action_dim=3,
            hidden_dims=[64, 64]
        )
        
        state = torch.randn(4, 10)
        
        # Forward pass
        mean, log_std = policy(state)
        assert mean.shape == (4, 3)
        assert log_std.shape == (4, 3)
        
        # Sample action
        action, log_prob = policy.sample(state)
        assert action.shape == (4, 3)
        assert log_prob.shape == (4, 1)
    
    def test_q_network(self):
        """Test Q-Network"""
        net = QNetwork(
            state_dim=10,
            action_dim=3,
            hidden_dims=[64, 64]
        )
        
        state = torch.randn(4, 10)
        action = torch.randn(4, 3)
        q_value = net(state, action)
        
        assert q_value.shape == (4, 1)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])