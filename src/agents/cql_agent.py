from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger

from .sac_agent import SACAgent

class CQLAgent(SACAgent):
    """
    Conservative Q-Learning (CQL) Agent
    Extends SAC with conservative Q-function updates for offline RL.
    """
    
    def __init__(
        self,
        cql_weight: float = 1.0,
        temp: float = 1.0,
        min_q_weight: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cql_weight = cql_weight
        self.temp = temp
        self.min_q_weight = min_q_weight
        
        logger.success(f"CQL Agent initialized | Weight: {cql_weight}")
        
    def learn(self, batch: Any = None) -> Dict[str, float]:
        """Update policy and Q-functions with CQL loss"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update Q-functions (standard SAC + CQL)
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_q_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        
        # Standard SAC error
        q1_loss_mse = F.mse_loss(q1_pred, target_q)
        q2_loss_mse = F.mse_loss(q2_pred, target_q)
        
        # === CQL Extra Terms ===
        # We need to compute Q-values for:
        # 1. Random actions
        # 2. Current policy actions
        # 3. Next policy actions (already have q_next, roughly)
        
        random_actions = torch.FloatTensor(self.batch_size, self.action_dim).uniform_(-1, 1).to(self.device)
        curr_actions, curr_log_probs = self.policy.sample(states)
        
        # Q1 terms
        q1_rand = self.q1(states, random_actions)
        q1_curr = self.q1(states, curr_actions)
        
        # Q2 terms
        q2_rand = self.q2(states, random_actions)
        q2_curr = self.q2(states, curr_actions)
        
        # LogSumExp terms
        # cat(q_rand, q_curr, q_next) - but strict CQL often just uses q_rand and q_curr
        # Simple CQL implementation: log(sum(exp(Q(s, a_rand)))) - Q(s, a_data)
        
        cat_q1 = torch.cat([q1_rand, q1_pred, q1_curr], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_pred, q2_curr], dim=1)
        
        cql_loss_q1 = (torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.temp) - q1_pred.mean()
        cql_loss_q2 = (torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.temp) - q2_pred.mean()
        
        q1_loss = q1_loss_mse + self.cql_weight * cql_loss_q1
        q2_loss = q2_loss_mse + self.cql_weight * cql_loss_q2
        
        # Optimization steps
        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy (standard SAC)
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - min_q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update
        if self.training_step % self.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
        
        self.update_training_step()
        
        return {
            'loss': (q1_loss + q2_loss + policy_loss).item(),
            'q1_loss': q1_loss.item(),
            'cql_loss': (cql_loss_q1 + cql_loss_q2).item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha
        }
