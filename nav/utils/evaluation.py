"""Episode-level metrics using deterministic actions and fixed scene seeds."""

import numpy as np
import torch
from envs.navigation import make_vec_envs
from utils.ppo import actor_action


def evaluate_policy(model, features, env_config, episodes=100, seed=10000):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    was_training = model.training
    model.eval()
    records = []
    for episode in range(episodes):
        envs = make_vec_envs(env_config, 1, seed+episode)
        env = envs.envs[0]
        returns = np.zeros(env.num_agents)
        corrections = []
        done = False
        while not done:
            inputs, _ = features(envs)
            with torch.no_grad():
                action = actor_action(model.actor, inputs, deterministic=True)[0]
            if 'u_ref' in inputs:
                corrections.append(float((action-inputs['u_ref']).norm(dim=-1).mean()))
            _, _, dones, infos = envs.step(action.cpu().numpy())
            info = infos[0]
            returns += info['task_reward']
            done = bool(dones[0])
        success = info['success'].astype(float)
        efficiency = env.initial_distances / np.maximum(np.maximum(env.path_lengths, env.initial_distances), 1e-8)
        records.append(dict(safety_rate=float(info['safe'].mean()),
                            reach_rate=float(info['reached'].mean()),
                            success_rate=float(success.mean()),
                            spl=float((success*efficiency).mean()),
                            task_return=float(returns.mean()), length=env.step_count,
                            avg_correction=float(np.mean(corrections)) if corrections else 0.))
        envs.close()
    model.train(was_training)
    return {key: float(np.mean([r[key] for r in records])) for key in records[0]}
