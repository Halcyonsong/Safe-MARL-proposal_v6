"""Adapter around the official GCBF+ DoubleIntegrator environment.

This wrapper intentionally delegates reset/step/dynamics/obstacles/LiDAR to the
vendored JAX implementation. It exposes states and a compact per-agent
observation for the V5 learned policy; it does not introduce a custom
environment or custom safety semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parents[3] / 'third_party' / 'gcbfplus'
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import jax
import jax.numpy as jnp
import jax.random as jr

from gcbfplus.env import make_env
from gcbfplus.env.utils import inside_obstacles


class OfficialDoubleIntegrator:
    """Single-environment adapter for the official GCBF+ DoubleIntegrator.

    Observation (per agent, fixed size):
      own state                 4
      goal relative state       4
      5 nearest agents rel p/v 20
      32 LiDAR hit rel p       64
    total                      92

    This is a V5 policy view of the official graph; the official GraphsTuple is
    still available through ``graph`` for h/graph models.
    """

    def __init__(self, num_agents=8, area_size=4.0, max_step=256,
                 num_obstacles=8, n_rays=32, max_neighbors=5, seed=0):
        self.num_agents = int(num_agents)
        self.max_neighbors = int(max_neighbors)
        self.n_rays = int(n_rays)
        self._env = make_env(
            'DoubleIntegrator', num_agents=self.num_agents,
            area_size=float(area_size), max_step=int(max_step),
            num_obs=int(num_obstacles), n_rays=int(n_rays))
        self.dt = float(self._env.dt)
        self.max_step = int(self._env.max_episode_steps)
        self.area_size = float(self._env.area_size)
        self.comm_radius = float(self._env.params['comm_radius'])
        self.step_count = 0
        self._seed = int(seed)
        self.graph = None

    @property
    def observation_dim(self):
        return 4 + 4 + self.max_neighbors * 4 + self.n_rays * 2

    @property
    def action_dim(self):
        return 2

    def reset(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        key = jr.PRNGKey(self._seed)
        self.graph = self._env.reset(key)
        self.step_count = 0
        return self.observation()

    def _agent_states(self, graph=None):
        graph = self.graph if graph is None else graph
        return np.asarray(
            graph.type_states(type_idx=0, n_type=self.num_agents),
            dtype=np.float32)

    def _goal_states(self, graph=None):
        graph = self.graph if graph is None else graph
        return np.asarray(
            graph.type_states(type_idx=1, n_type=self.num_agents),
            dtype=np.float32)

    def _lidar_states(self, graph=None):
        graph = self.graph if graph is None else graph
        return np.asarray(
            graph.type_states(type_idx=2, n_type=self.n_rays * self.num_agents),
            dtype=np.float32)

    def observation(self):
        """Compact per-agent observation built from official graph states."""
        agent = self._agent_states()
        goal = self._goal_states()
        lidar = self._lidar_states().reshape(self.num_agents, self.n_rays, 4)
        pos = agent[:, :2]
        vel = agent[:, 2:]
        # Relative agent features, nearest (including self) sorted by distance.
        rel = np.concatenate([
            agent[None, :, :] - agent[:, None, :],
        ], axis=-1)
        dist = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=-1)
        order = np.argsort(dist, axis=1)[:, :self.max_neighbors]
        neighbor = np.take_along_axis(rel, order[..., None].repeat(4, axis=-1), axis=1)
        # LiDAR hit positions relative to the observing agent. The official
        # sensor uses an infinite-ray sentinel (1e6) for no-hit beams; the
        # official graph masks those out. Do the same here instead of feeding
        # the sentinel to the policy.
        lidar_rel = lidar[..., :2] - pos[:, None, :]
        hit_dist = np.linalg.norm(lidar_rel, axis=-1)
        active = hit_dist < (self.comm_radius - 1e-1)
        lidar_rel = np.where(active[..., None], lidar_rel, 0.0)
        # Flatten: own state, goal relative state, neighbor rel p/v,
        # lidar rel p. This mirrors the official agent-goal edge feature.
        obs = np.concatenate([
            agent,
            goal - agent,
            neighbor.reshape(self.num_agents, -1),
            lidar_rel.reshape(self.num_agents, -1),
        ], axis=1).astype(np.float32)
        return obs

    def step(self, action):
        """Advance official dynamics without invoking the official reward/u_ref.

        The official ``DoubleIntegrator.step`` computes a reward against an LQR
        ``u_ref``; V5 deliberately does not use that expert reward. This method
        reuses only the official state transition, collision cost, safe/unsafe
        masks and LiDAR graph construction.
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        prev_agent = self._agent_states()
        prev_goal = self._goal_states()
        goal_states = self.graph.type_states(
            type_idx=1, n_type=self.num_agents)
        obstacles = self.graph.env_states.obstacle
        current_cost = self._env.get_cost(self.graph)
        clipped = self._env.clip_action(jnp.asarray(action))
        next_agent = self._env.agent_step_euler(prev_agent, clipped)
        next_state = self._env.EnvState(next_agent, goal_states, obstacles)
        next_graph = self._env.get_graph(next_state)
        self._env._t += 1
        self.graph = next_graph
        self.step_count += 1
        agent = self._agent_states()
        goal = self._goal_states()
        prev_dist = np.linalg.norm(prev_agent[:, :2] - prev_goal[:, :2], axis=1)
        dist = np.linalg.norm(agent[:, :2] - goal[:, :2], axis=1)
        progress = prev_dist - dist
        # Official safety semantics; do not replace them with custom rules.
        # `unsafe_mask` includes a warning zone/heading condition, while
        # `collision_mask` is the actual collision condition used by the
        # official evaluation. Keep both, explicitly distinguished.
        safe_vec = np.asarray(self._env.safe_mask(next_graph), dtype=bool)
        unsafe_vec = np.asarray(self._env.unsafe_mask(next_graph), dtype=bool)
        collision_vec = np.asarray(
            self._env.collision_mask(next_graph), dtype=bool)
        reached = np.asarray(self._env.finish_mask(next_graph), dtype=bool)
        inside = inside_obstacles(
            prev_agent[:, :2], obstacles, r=self._env.params['car_radius'])
        terminated = bool(self.step_count >= self.max_step)
        obs = self.observation()
        info = dict(progress=progress.astype(np.float32),
                    distance=dist.astype(np.float32),
                    safety=float(safe_vec.mean()),
                    unsafe=unsafe_vec.copy(),
                    collision=collision_vec.copy(),
                    reached=reached.copy(),
                    official_cost=float(current_cost),
                    inside_obstacles=np.asarray(inside, dtype=bool))
        return obs, progress.astype(np.float32), terminated, False, info

    def get_graph(self):
        return self.graph

    def get_agent_states(self):
        return self._agent_states()

    def get_goal_states(self):
        return self._goal_states()

    def get_lidar(self):
        return self._lidar_states()
