"""
Stage 2: Base Policy Training (Standard MAPPO)

Train base policy to solve task WITHOUT considering safety
"""

from .train_base_policy import train_base_policy

__all__ = [
    'train_base_policy',
]
