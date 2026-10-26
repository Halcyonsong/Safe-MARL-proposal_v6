"""Reproducible runtime setup and versioned checkpoint loading."""

import random
import numpy as np
import torch


SCHEMA_VERSION = 2


def setup(config, requested_device='cpu'):
    seed = int(config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('num_threads', 4)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(config.get('deterministic', True), warn_only=True)
    return torch.device(requested_device if torch.cuda.is_available() else 'cpu')


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'{path}: legacy checkpoint; retrain using schema {SCHEMA_VERSION}')
    return checkpoint


def graph_kwargs(env_config):
    return {key: env_config[key] for key in ('sensing_radius', 'agent_radius', 'area_size')}
