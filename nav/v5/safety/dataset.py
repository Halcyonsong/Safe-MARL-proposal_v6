"""Offline safety dataset for learned h.

Expected ``.npz`` keys:

- ``states`` [N, state_dim] or ``obs``;
- ``cost_labels`` [N, C] (binary), or ``cost`` [N] for one cost;
- ``terminal`` [N] (optional; defaults to zero);
- ``next_states`` [N, state_dim] (optional, for invariance/direction losses).

The dataset contains safety labels only; it has no task-return field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SafetyDataset(Dataset):
    states: np.ndarray
    cost_labels: np.ndarray
    terminal: np.ndarray
    next_states: Optional[np.ndarray] = None

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=np.float32)
        self.cost_labels = np.asarray(self.cost_labels, dtype=np.float32)
        self.terminal = np.asarray(self.terminal, dtype=np.float32).reshape(-1)
        if self.states.ndim != 2:
            raise ValueError(f'states must be [N,D], got {self.states.shape}')
        if self.cost_labels.ndim == 1:
            self.cost_labels = self.cost_labels[:, None]
        if self.cost_labels.ndim != 2:
            raise ValueError(
                f'cost_labels must be [N,C], got {self.cost_labels.shape}')
        if len(self.states) != len(self.cost_labels) or len(self.states) != len(self.terminal):
            raise ValueError('states / cost_labels / terminal length mismatch')
        if self.next_states is not None:
            self.next_states = np.asarray(self.next_states, dtype=np.float32)
            if self.next_states.shape != self.states.shape:
                raise ValueError('next_states must match states shape')

    @classmethod
    def from_npz(cls, path: str | Path) -> 'SafetyDataset':
        data = np.load(Path(path))
        states = data['states'] if 'states' in data else data['obs']
        cost = data['cost_labels'] if 'cost_labels' in data else data['cost']
        terminal = data['terminal'] if 'terminal' in data else np.zeros(
            len(states), dtype=np.float32)
        nxt = data['next_states'] if 'next_states' in data else None
        return cls(states=states, cost_labels=cost, terminal=terminal,
                   next_states=nxt)

    @property
    def cost_dim(self) -> int:
        return int(self.cost_labels.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    def __len__(self) -> int:
        return int(len(self.states))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(
            state=torch.from_numpy(self.states[index].copy()),
            cost_labels=torch.from_numpy(self.cost_labels[index].copy()),
            terminal=torch.tensor(float(self.terminal[index])),
        )
        if self.next_states is not None:
            item['next_state'] = torch.from_numpy(
                self.next_states[index].copy())
            item['delta_state'] = item['next_state'] - item['state']
        return item

    def loader(self, batch_size: int = 256, shuffle: bool = True) -> Iterator:
        return DataLoader(self, batch_size=int(batch_size), shuffle=bool(shuffle))


__all__ = ['SafetyDataset']
