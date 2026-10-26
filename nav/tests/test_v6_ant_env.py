"""Integration smoke for the V6 Ant adapter (requires MuJoCo/mappo_lag)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

pytest.importorskip('mujoco_py')

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_safety import (  # noqa: E402
    FixedAntSafetyProvider, official_cost_from_geom)


def test_env_reset_step_and_cost_consistency():
    env = Ant2x4Env(
        seed=0, episode_limit=8,
        safety_provider=FixedAntSafetyProvider())
    obs, h, grad, hvec, hs, geom, state = env.reset(seed=3)
    assert obs.shape == (env.num_agents, env.observation_dim)
    assert h.shape == (env.num_agents, 1)
    assert grad.shape == (env.num_agents, 9)
    assert hvec.shape == (4,)
    assert geom.shape == (9,)
    assert state.shape == (env.global_state_dim,)
    assert env.global_state_dim == 31
    assert np.all(h == hs)
    for _ in range(4):
        out = env.step(np.zeros((env.num_agents, env.action_dim)))
        obs, h, grad, hvec, hs, geom, state = out[:7]
        reward, cost, terminated, truncated, info = out[7:]
        assert obs.shape == (env.num_agents, env.observation_dim)
        assert h.shape == (env.num_agents, 1)
        assert grad.shape == (env.num_agents, 9)
        assert np.isfinite(reward)
        assert cost in (0.0, 1.0)
        assert int(cost) == official_cost_from_geom(geom)
        assert np.isfinite(info['h'])
        if terminated or truncated:
            break
    env.close()
