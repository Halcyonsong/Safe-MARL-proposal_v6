"""Smoke test: the V5 adapter delegates to the official GCBF+ environment."""

import sys
from pathlib import Path

import numpy as np

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator
from gcbfplus.env import make_env
import jax.random as jr


def test_official_env_shapes_and_step():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=256, num_obstacles=8,
        n_rays=32, max_neighbors=5, seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (8, env.observation_dim)
    assert obs.dtype == np.float32
    assert env.get_agent_states().shape == (8, 4)
    assert env.get_goal_states().shape == (8, 4)
    assert env.get_lidar().shape == (32 * 8, 4)
    next_obs, reward, terminated, truncated, info = env.step(np.zeros((8, 2)))
    assert next_obs.shape == obs.shape
    assert reward.shape == (8,)
    assert terminated is False and truncated is False
    assert {'safety', 'unsafe', 'reached', 'official_cost'} <= set(info)
    assert env.step_count == 1


def test_adapter_reset_matches_official_env():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=256, num_obstacles=8,
        n_rays=32, max_neighbors=5, seed=0)
    env.reset(seed=0)
    official = make_env('DoubleIntegrator', num_agents=8, area_size=4.0,
                        max_step=256, num_obs=8, n_rays=32)
    graph = official.reset(jr.PRNGKey(0))
    direct_agent = np.asarray(graph.type_states(type_idx=0, n_type=8))
    direct_goal = np.asarray(graph.type_states(type_idx=1, n_type=8))
    direct_lidar = np.asarray(graph.type_states(type_idx=2, n_type=8 * 32))
    assert np.allclose(env.get_agent_states(), direct_agent)
    assert np.allclose(env.get_goal_states(), direct_goal)
    assert np.allclose(env.get_lidar(), direct_lidar)


def test_adapter_step_dynamics_match_official_env():
    import jax.numpy as jnp

    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=256, num_obstacles=8,
        n_rays=32, max_neighbors=5, seed=0)
    env.reset(seed=0)
    official = make_env('DoubleIntegrator', num_agents=8, area_size=4.0,
                        max_step=256, num_obs=8, n_rays=32)
    graph = official.reset(jr.PRNGKey(0))
    action = np.zeros((8, 2), dtype=np.float32)
    action[0] = [0.4, -0.2]
    next_graph, _, direct_cost, _, _ = official.step(
        graph, jnp.asarray(action), get_eval_info=True)
    _, _, _, _, info = env.step(action)
    direct_agent = np.asarray(next_graph.type_states(type_idx=0, n_type=8))
    assert np.allclose(env.get_agent_states(), direct_agent)
    assert np.isclose(info['official_cost'], float(direct_cost))


def test_adapter_step_does_not_call_official_uref_reward():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=256, num_obstacles=8,
        n_rays=32, max_neighbors=5, seed=0)
    env.reset(seed=0)

    def _forbidden(_graph):
        raise AssertionError('adapter step called official u_ref')

    env._env.u_ref = _forbidden
    env.step(np.zeros((8, 2), dtype=np.float32))
