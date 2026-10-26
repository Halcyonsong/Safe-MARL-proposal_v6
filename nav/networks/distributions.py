"""Bounded Gaussian actions with identical sampling/evaluation probabilities."""

import torch
from torch.distributions import Normal


def bounded_sample(mean, std, deterministic=False):
    raw = mean if deterministic else Normal(mean, std).sample()
    action = raw.tanh().clamp(-1+1e-6, 1-1e-6)
    return action, None if deterministic else bounded_log_prob(mean, std, action)


def bounded_log_prob(mean, std, action):
    action = action.clamp(-1+1e-6, 1-1e-6)
    raw = torch.atanh(action)
    return (Normal(mean, std).log_prob(raw) - torch.log1p(-action.square())).sum(-1, keepdim=True)


def bounded_entropy(mean, std):
    """Monte Carlo entropy of the transformed distribution, with gradients."""
    raw = Normal(mean, std).rsample()
    # Stable log sech^2(raw).
    jacobian = 2 * (torch.log(raw.new_tensor(2.)) - raw - torch.nn.functional.softplus(-2*raw))
    return -(Normal(mean, std).log_prob(raw) - jacobian).sum(-1, keepdim=True)
