"""
Base Policy Network (Stage 2)

MAPPO Actor-Critic for multi-agent navigation
- Actor: Decentralized policy (local observations)
- Critic: Centralized value function (global state)

Reference: MAPPO (Yu et al., 2021), MACPO (Gu et al., AIJ 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .distributions import bounded_sample, bounded_log_prob, bounded_entropy


def init_weights(module, gain=1.0):
    """Orthogonal initialization"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Actor(nn.Module):
    """
    Actor network for decentralized policy
    Outputs: action_mean, action_log_std
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128], 
                 activation='tanh', init_gain=np.sqrt(2), output_gain=0.01,
                 feature_norm=False, layer_norm=False,
                 norm_dim=None, extra_dim=0):
        super().__init__()
        
        self.obs_dim = int(obs_dim)
        self.action_dim = action_dim
        self.feature_norm = bool(feature_norm)
        self.layer_norm = bool(layer_norm)
        # Optional task/value feature appended after the input normalization.
        # ``norm_dim`` must cover the original observation-like features, so a
        # zero-initialized extra column cannot change the LayerNorm statistics
        # of the pre-existing policy.
        self.extra_dim = int(extra_dim)
        self.norm_dim = int(self.obs_dim if norm_dim is None else norm_dim)
        if self.norm_dim + self.extra_dim != self.obs_dim:
            raise ValueError(
                f'norm_dim ({self.norm_dim}) + extra_dim ({self.extra_dim}) '
                f'must equal obs_dim ({self.obs_dim})')
        
        # Activation function
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Optional input normalization.  The upstream MAPPO Ant baseline uses
        # LayerNorm on the observation before the MLP; this option keeps the
        # default nav actor unchanged.  With extra_dim > 0 the norm is applied
        # only to the first norm_dim features.
        self.input_norm = (
            nn.LayerNorm(self.norm_dim) if self.feature_norm
            else nn.Identity())
        
        # Shared feature extractor
        layers = []
        dims = [self.obs_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn())
            if self.layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Action mean head
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
        
        # Action log std (learnable parameter)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=init_gain))
        init_weights(self.action_mean, gain=output_gain)  # Small output
    
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] or [batch_size, num_agents, obs_dim]
        
        Returns:
            action_mean: [batch_size, action_dim]
            action_std: [batch_size, action_dim]
        """
        if self.extra_dim > 0:
            base = obs[..., :self.norm_dim]
            extra = obs[..., self.norm_dim:]
            features = self.feature_extractor(
                torch.cat([self.input_norm(base), extra], dim=-1))
        else:
            features = self.feature_extractor(self.input_norm(obs))
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std.clamp(-5, 2)).expand_as(action_mean)
        
        return action_mean, action_std
     
    def get_action(self, obs, deterministic=False, bounded=True):
        """
        Sample action from policy.
        
        ``bounded=True`` keeps the original tanh-squashed policy used by the
        navigation and V6 Ant h arms.  ``bounded=False`` matches the upstream
        MAPPO Ant Gaussian policy (no tanh squashing), which is needed for a
        faithful pure-task baseline comparison.
        
        Args:
            obs: [batch_size, obs_dim]
            deterministic: bool - if True, return mean action
            bounded: bool - whether to apply the tanh action transform
        
        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size, 1]
        """
        action_mean, action_std = self.forward(obs)
        if bounded:
            return bounded_sample(action_mean, action_std, deterministic)
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs, actions, bounded=True):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: [batch_size, obs_dim]
            actions: [batch_size, action_dim]
            bounded: bool - match the sampling transform
        
        Returns:
            log_prob: [batch_size, 1]
            entropy: [batch_size, 1]
        """
        action_mean, action_std = self.forward(obs)
        if bounded:
            log_prob = bounded_log_prob(action_mean, action_std, actions)
            entropy = bounded_entropy(action_mean, action_std)
        else:
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy

    @staticmethod
    def _squashed_log_prob(dist, raw_action):
        """Log probability after tanh squashing, matching env action bounds."""
        correction = torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """
    Critic network for centralized value function
    Input: global state (all agents' observations)
    Output: state value
    """
    
    def __init__(self, state_dim, hidden_dims=[256, 256], 
                 activation='tanh', use_value_norm=True,
                 feature_norm=False, layer_norm=False):
        super().__init__()
        
        self.state_dim = state_dim
        self.use_value_norm = use_value_norm
        self.feature_norm = bool(feature_norm)
        self.layer_norm = bool(layer_norm)
        
        # Activation function
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Value network
        layers = []
        dims = [state_dim] + hidden_dims + [1]
        if self.feature_norm:
            layers.append(nn.LayerNorm(state_dim))
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(act_fn())
                if self.layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
        
        self.value_net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
        # Value normalization
        if use_value_norm:
            self.value_norm = ValueNorm(1)
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim]
        
        Returns:
            value: [batch_size, 1]
        """
        value = self.value_net(state)
        
        if self.use_value_norm:
            value = self.value_norm.denormalize(value)
        
        return value
    
    @torch.no_grad()
    def update_value_norm(self, values):
        """Update value normalization statistics"""
        if self.use_value_norm:
            old_mean = self.value_norm.running_mean.clone()
            old_std = torch.sqrt(self.value_norm.running_var.clamp_min(1e-4)
                                 + self.value_norm.epsilon)
            self.value_norm.update(values)
            new_std = torch.sqrt(self.value_norm.running_var.clamp_min(1e-4)
                                 + self.value_norm.epsilon)
            # Preserve unnormalized predictions as normalization changes.
            output = self.value_net[-1]
            output.weight.mul_(old_std/new_std)
            output.bias.copy_((old_std*output.bias+old_mean-self.value_norm.running_mean)/new_std)


class ValueNorm(nn.Module):
    """
    Running mean and std normalization for value function
    Reference: MAPPO implementation
    """
    
    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        
        self.epsilon = epsilon
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(1))
    
    @torch.no_grad()
    def update(self, values):
        """Update running statistics"""
        values = values.detach().reshape(-1, self.running_mean.numel())
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.size(0)
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(total_count)
    
    def normalize(self, values):
        """Normalize values"""
        return (values - self.running_mean) / torch.sqrt(self.running_var.clamp_min(1e-4) + self.epsilon)
    
    def denormalize(self, values):
        """Denormalize values"""
        return values * torch.sqrt(self.running_var.clamp_min(1e-4) + self.epsilon) + self.running_mean


class BasePolicyNetwork(nn.Module):
    """
    Complete MAPPO Actor-Critic Network
    
    Usage:
        policy = BasePolicyNetwork(obs_dim, state_dim, action_dim, config)
        action, log_prob = policy.get_action(obs)
        value = policy.get_value(state)
    """
    
    def __init__(self, obs_dim, state_dim, action_dim, config):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor (decentralized)
        self.actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.get('actor_hidden_dims', [256, 256, 128]),
            activation=config.get('actor_activation', 'tanh'),
            init_gain=config.get('actor_init_gain', np.sqrt(2)),
            output_gain=config.get('actor_output_gain', 0.01),
            feature_norm=config.get('actor_feature_norm', False),
            layer_norm=config.get('actor_layer_norm', False),
        )
        
        # Critic (centralized)
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=config.get('critic_hidden_dims', [256, 256]),
            activation=config.get('critic_activation', 'tanh'),
            use_value_norm=config.get('use_value_norm', True),
            feature_norm=config.get('critic_feature_norm', False),
            layer_norm=config.get('critic_layer_norm', False),
        )
    
    def get_action(self, obs, deterministic=False):
        """Get action from actor"""
        return self.actor.get_action(obs, deterministic)
    
    def get_value(self, state):
        """Get value from critic"""
        return self.critic(state)
    
    def evaluate_actions(self, obs, state, actions):
        """Evaluate actions for PPO update"""
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic(state)
        return log_prob, value, entropy
    
    def update_value_norm(self, values):
        """Update value normalization"""
        self.critic.update_value_norm(values)


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: [num_steps] - rewards
        values: [num_steps + 1] - value estimates (including next_value)
        dones: [num_steps] - done flags
        gamma: float - discount factor
        gae_lambda: float - GAE lambda
    
    Returns:
        advantages: [num_steps]
        returns: [num_steps]
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32, device=rewards.device)
    dones = torch.as_tensor(dones, device=rewards.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    
    return advantages, returns


if __name__ == "__main__":
    # Test Base Policy Network
    obs_dim = 14  # self_state(6) + neighbors(2*4)
    state_dim = 32  # 8 agents * 4 states
    action_dim = 2
    
    config = {
        'actor_hidden_dims': [256, 256, 128],
        'critic_hidden_dims': [256, 256],
        'actor_activation': 'tanh',
        'critic_activation': 'tanh',
        'use_value_norm': True
    }
    
    policy = BasePolicyNetwork(obs_dim, state_dim, action_dim, config)
    
    print(f"Actor parameters: {sum(p.numel() for p in policy.actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in policy.critic.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, obs_dim)
    state = torch.randn(batch_size, state_dim)
    
    # Get action
    action, log_prob = policy.get_action(obs, deterministic=False)
    print(f"\nAction shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    # Get value
    value = policy.get_value(state)
    print(f"Value shape: {value.shape}")
    
    # Evaluate actions
    log_prob, value, entropy = policy.evaluate_actions(obs, state, action)
    print(f"\nEvaluate - Log prob: {log_prob.shape}, Value: {value.shape}, Entropy: {entropy.shape}")
    
    # Test GAE
    rewards = torch.randn(10)
    values = torch.randn(11)
    dones = torch.zeros(10)
    advantages, returns = compute_gae(rewards, values, dones)
    print(f"\nGAE - Advantages: {advantages.shape}, Returns: {returns.shape}")
    
    print("\n✓ Base Policy Network test passed!")
