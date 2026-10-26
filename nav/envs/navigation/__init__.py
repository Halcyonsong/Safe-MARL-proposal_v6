"""Joint-action navigation environment and a synchronous vector wrapper."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DoubleIntegratorEnv(gym.Env):
    """N disc robots with acceleration control; all robots step simultaneously.

    Gymnasium observation/action shapes are [N, D] and [N, 2]. The scalar
    environment reward is the mean task reward; per-agent rewards and safety
    masks are available in info. Collisions are recorded throughout an episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.dt = float(config['dt'])
        self.max_speed = float(config['max_speed'])
        self.max_acceleration = float(config['max_acceleration'])
        self.area_size = float(config['area_size'])
        self.num_agents = int(config['num_agents_train'])
        self.num_obstacles = int(config['num_obstacles_train'])
        self.agent_radius = float(config['agent_radius'])
        self.max_steps = int(config['max_steps'])
        self.success_threshold = float(config['success_threshold'])
        self.max_neighbors = int(config.get('max_neighbors', 5))
        self.sensing_radius = float(config.get('sensing_radius', 3.0))
        if min(self.num_agents, self.max_steps, self.dt, self.agent_radius) <= 0:
            raise ValueError('num_agents, max_steps, dt and agent_radius must be positive')
        self.obs_dim = 6 + self.max_neighbors * 6
        self.state_dim = self.num_agents * 6 + self.num_obstacles * 3
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.num_agents, self.obs_dim), np.float32)
        self.action_space = spaces.Box(-1., 1., (self.num_agents, 2), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self._ended = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        rng = self.np_random
        self.obstacle_radii = rng.uniform(
            self.config['obstacle_radius_min'], self.config['obstacle_radius_max'],
            self.num_obstacles)
        self.obstacles = np.empty((self.num_obstacles, 2))
        for j, radius in enumerate(self.obstacle_radii):
            limit = self.area_size / 2 - radius
            self.obstacles[j] = rng.uniform(-limit, limit, 2)

        def free_points():
            points = []
            limit = self.area_size / 2 - self.agent_radius - 0.01
            for _ in range(self.num_agents):
                for attempt in range(10000):
                    point = rng.uniform(-limit, limit, 2)
                    obstacle_clear = np.all(np.linalg.norm(self.obstacles - point, axis=1)
                                            > self.obstacle_radii + self.agent_radius + .05)
                    agent_clear = all(np.linalg.norm(point - p) > 2 * self.agent_radius + .05
                                      for p in points)
                    if obstacle_clear and agent_clear:
                        points.append(point)
                        break
                else:
                    raise ValueError('Cannot sample a collision-free scene at this density')
            return np.asarray(points)

        self.positions = free_points()
        self.goals = free_points()
        self.velocities = np.zeros_like(self.positions)
        self.reached = np.zeros(self.num_agents, dtype=bool)
        self.collided = np.zeros(self.num_agents, dtype=bool)
        self.path_lengths = np.zeros(self.num_agents)
        self.initial_distances = np.linalg.norm(self.positions - self.goals, axis=1)
        self.step_count = 0
        self._ended = False
        return self._get_obs(), {}

    def integrate(self, states, actions):
        """Semi-implicit Euler step, shared by simulation and data collection."""
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions)
        if states.shape != (self.num_agents, 4) or actions.shape != (self.num_agents, 2):
            raise ValueError('Expected states [N,4] and actions [N,2]')
        velocity = states[:, 2:] + np.clip(actions, -1, 1) * self.max_acceleration * self.dt
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity *= np.minimum(1., self.max_speed / np.maximum(speed, 1e-12))
        return np.concatenate([states[:, :2] + velocity * self.dt, velocity], axis=1)

    def collision_mask(self, states=None):
        positions = self.positions if states is None else np.asarray(states)[:, :2]
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        collision = (distance < 2 * self.agent_radius).any(axis=1)
        if self.num_obstacles:
            distance = np.linalg.norm(positions[:, None] - self.obstacles[None, :], axis=-1)
            collision |= (distance < self.obstacle_radii + self.agent_radius).any(axis=1)
        collision |= (np.abs(positions) + self.agent_radius > self.area_size / 2).any(axis=1)
        return collision

    def clearance_margin(self, states=None):
        """Signed clearance to the nearest collision boundary.

        Positive values are safe, zero is the exact collision boundary and
        negative values are inside a collision. Unlike ``collision_mask`` this
        is a continuous barrier target suitable for regression.
        """
        positions = self.positions if states is None else np.asarray(states)[:, :2]
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        margin = (distance - 2 * self.agent_radius).min(axis=1)
        if self.num_obstacles:
            distance = np.linalg.norm(positions[:, None] - self.obstacles[None, :], axis=-1)
            margin = np.minimum(margin,
                                (distance - self.obstacle_radii - self.agent_radius).min(axis=1))
        # The simulator treats the square boundary as four half-planes.
        wall_margin = self.area_size / 2 - np.abs(positions) - self.agent_radius
        margin = np.minimum(margin, wall_margin.min(axis=1))
        return margin.astype(np.float32)

    def stopping_clearance_margin(self, states=None, barrier_margin=0.0,
                                  speed_coef=1.0):
        """Signed clearance after subtracting a braking-distance margin.

        This is the velocity-aware barrier target used by h v3. For a
        static obstacle/wall the required clearance grows as
        ``v^2 / (2 a_max)``. For agent pairs the relative speed is used with
        the combined braking capability ``2 a_max``. The result is negative
        when the remaining clearance is smaller than the stopping distance
        plus ``barrier_margin``.
        """
        states = self.get_agent_states() if states is None else np.asarray(states, dtype=np.float64)
        positions, velocities = states[:, :2], states[:, 2:]
        speed = np.linalg.norm(velocities, axis=1)
        stopping = float(speed_coef) * speed ** 2 / (2.0 * self.max_acceleration)
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        relative_speed = np.linalg.norm(velocities[:, None] - velocities[None, :], axis=-1)
        relative_stopping = float(speed_coef) * relative_speed ** 2 / (4.0 * self.max_acceleration)
        margin = (distance - 2.0 * self.agent_radius - relative_stopping).min(axis=1)
        if self.num_obstacles:
            distance = np.linalg.norm(positions[:, None] - self.obstacles[None, :], axis=-1)
            margin = np.minimum(
                margin,
                (distance - self.obstacle_radii - self.agent_radius - stopping[:, None]).min(axis=1))
        wall_margin = (self.area_size / 2.0 - np.abs(positions) - self.agent_radius
                       - stopping[:, None])
        margin = np.minimum(margin, wall_margin.min(axis=1))
        return (margin - float(barrier_margin)).astype(np.float32)

    def check_collision(self, states=None):
        return self.collision_mask(states)

    def step(self, actions):
        if self._ended:
            raise RuntimeError('Call reset() before stepping a finished environment')
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, 2) or not np.isfinite(actions).all():
            raise ValueError('Actions must be finite with shape [N,2]')
        old_positions = self.positions.copy()
        old_distance = np.linalg.norm(self.positions - self.goals, axis=1)
        states = self.integrate(self.get_agent_states(), actions)
        states[self.reached, :2] = self.positions[self.reached]
        states[self.reached, 2:] = 0
        self.positions, self.velocities = states[:, :2], states[:, 2:]
        distance = np.linalg.norm(self.positions - self.goals, axis=1)
        newly_reached = ~self.reached & (distance < self.success_threshold)
        task_reward = old_distance - distance + 10 * newly_reached - .01 * ~self.reached
        self.reached |= newly_reached
        self.velocities[self.reached] = 0
        collision = self.collision_mask()
        self.collided |= collision
        self.path_lengths += np.linalg.norm(self.positions - old_positions, axis=1)
        self.step_count += 1
        terminated = bool(self.reached.all())
        if self.config.get('terminate_on_collision', False):
            terminated |= bool(collision.any())
        truncated = self.step_count >= self.max_steps and not terminated
        self._ended = terminated or truncated
        info = {'collision': collision, 'reached': self.reached.copy(),
                'success': self.reached & ~self.collided, 'safe': ~self.collided,
                'newly_reached': newly_reached, 'dist_to_goal': distance,
                'task_reward': task_reward.astype(np.float32),
                'terminated': terminated, 'truncated': truncated}
        return self._get_obs(), float(task_reward.mean()), terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        obs[:, :2], obs[:, 2:4], obs[:, 4:6] = self.positions, self.velocities, self.goals
        for i in range(self.num_agents):
            rows = []
            for j in range(self.num_agents):
                if i != j:
                    rows.append((self.positions[j] - self.positions[i],
                                 self.velocities[j] - self.velocities[i], self.agent_radius, 0.))
            for pos, radius in zip(self.obstacles, self.obstacle_radii):
                rows.append((pos - self.positions[i], -self.velocities[i], radius, 1.))
            # Four wall points are included as local safety observations.
            for axis in range(2):
                for side in (-1, 1):
                    rel = np.zeros(2)
                    rel[axis] = side * self.area_size / 2 - self.positions[i, axis]
                    rows.append((rel, -self.velocities[i], 0., 1.))
            rows.sort(key=lambda row: np.linalg.norm(row[0]) - row[2])
            rows = [row for row in rows if np.linalg.norm(row[0]) - row[2] <= self.sensing_radius]
            for k, (pos, vel, radius, kind) in enumerate(rows[:self.max_neighbors]):
                obs[i, 6 + 6*k:12 + 6*k] = np.r_[pos, vel, radius, kind]
        return obs

    def get_agent_states(self):
        return np.concatenate([self.positions, self.velocities], axis=1).astype(np.float32)

    def get_state(self):
        return np.r_[self.get_agent_states().ravel(), self.goals.ravel(),
                     self.get_obstacles().ravel()].astype(np.float32)

    def get_goal(self):
        return self.goals.astype(np.float32).copy()

    def get_obstacles(self):
        return np.column_stack([self.obstacles, self.obstacle_radii]).astype(np.float32)


class NavigationVecEnv:
    """Return terminal observations; reset_done is called after bootstrapping."""

    def __init__(self, config, num_envs, seed=0):
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        self.envs = [DoubleIntegratorEnv(config) for _ in range(num_envs)]
        self.num_envs, self.num_agents = num_envs, self.envs[0].num_agents
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.state_space = self.envs[0].state_space
        self._needs_reset = np.zeros(num_envs, dtype=bool)
        self.observations = np.stack([env.reset(seed=seed+i)[0] for i, env in enumerate(self.envs)])

    def reset(self):
        self._needs_reset[:] = False
        self.observations = np.stack([env.reset()[0] for env in self.envs])
        return self.observations.copy()

    def step(self, actions):
        if self._needs_reset.any():
            raise RuntimeError('Store terminal transitions and call reset_done() before step()')
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, terminated, truncated, info = zip(*results)
        self._needs_reset = np.asarray(terminated) | np.asarray(truncated)
        self.observations = np.stack(obs)
        return self.observations.copy(), np.asarray(rewards), self._needs_reset.copy(), list(info)

    def reset_done(self, observations=None):
        for i in np.flatnonzero(self._needs_reset):
            self.observations[i] = self.envs[i].reset()[0]
        self._needs_reset[:] = False
        return self.observations.copy()

    def get_state(self):
        return np.stack([env.get_state() for env in self.envs])

    def get_agent_states(self):
        return np.stack([env.get_agent_states() for env in self.envs])

    def get_goal(self):
        return np.stack([env.get_goal() for env in self.envs])

    def get_obstacles(self):
        return np.stack([env.get_obstacles() for env in self.envs])

    def close(self):
        for env in self.envs:
            env.close()


def make_vec_envs(config, num_envs, seed=0):
    if config.get('name', 'DoubleIntegrator') != 'DoubleIntegrator':
        raise NotImplementedError('Only DoubleIntegrator is implemented')
    return NavigationVecEnv(config, num_envs, seed)
