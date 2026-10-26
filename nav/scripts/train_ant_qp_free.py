"""QP-free V5 smoke train/eval on official Safe MAMuJoCo Ant-2x4.

This is a bridge experiment, not a final Ant baseline.  It uses the official
``MujocoMulti(Ant-v2, 2x4)`` environment, the official binary cost (wall or
topple/height), and a fixed V5 Ant margin.  There is no expert u_ref and no QP.

The task head can be warm-started by behavior cloning a learned MAPPO-Lag
Ant actor (allowed by the proposal).  The correction head and safety losses are
still learned under the V5 objective; the base actor is not used at inference.

Run with the MuJoCo conda environment, after ``source ../stage1/env.sh`` from
the v2 repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.safe_policy.qp_free_ant import QpFreeAntEnsemble  # noqa: E402
from algos.safe_policy.qp_free import (  # noqa: E402
    QpFreePolicy,
    barrier_loss,
    correction_penalty,
    forward_invariance_loss,
    gradient_direction_loss,
)
from envs.mamujoco import Ant2x4Env  # noqa: E402

V2 = Path(os.environ.get(
    'SAFE_MAMUJOCO_ROOT',
    str(ROOT.parent / 'Safe-MARL-proposal_v2'))).expanduser()
PKG = V2 / 'third_party/mappo_lag/MAPPO-Lagrangian'
BASE_MODELS = (
    PKG / 'mappo_lagrangian/scripts/results/mujoco/Ant-v2/'
    'mappo_lagr/mappo_paper/run1/models'
)


@dataclass
class AntRollout:
    obs: torch.Tensor
    h_vec: torch.Tensor
    grad: torch.Tensor
    h: torch.Tensor
    h_next: torch.Tensor
    action_raw: torch.Tensor
    action_env: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    cost: torch.Tensor
    reward: torch.Tensor
    cost_target: torch.Tensor
    official_cost: torch.Tensor
    terminal: torch.Tensor
    done: torch.Tensor
    delta_state: torch.Tensor
    last_value: torch.Tensor
    last_cost: torch.Tensor
    completed: list[dict]

    def flatten(self):
        t, n = self.h.shape
        return dict(
            obs=self.obs.reshape(t * n, -1),
            h_vec=self.h_vec.reshape(t * n, -1),
            grad=self.grad.reshape(t * n, -1),
            h=self.h.reshape(-1),
            h_next=self.h_next.reshape(-1),
            action_raw=self.action_raw.reshape(t * n, -1),
            log_prob=self.log_prob.reshape(-1),
            value=self.value.reshape(-1),
            cost=self.cost.reshape(-1),
            reward=self.reward.reshape(-1),
            cost_target=self.cost_target.reshape(-1),
            official_cost=self.official_cost.reshape(-1),
            terminal=self.terminal.reshape(-1),
            delta_state=self.delta_state.reshape(t * n, -1),
            valid=(~self.done).to(self.obs.dtype)[:, None].expand(t, n).reshape(-1),
        )


def _t(x, device):
    return torch.as_tensor(np.array(x, copy=True), dtype=torch.float32,
                           device=device)


def _gae(rewards, values, dones, last_value, gamma, lam):
    steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in range(steps - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < steps else last_value
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def add_gae(rollout, gamma, lam):
    rollout.advantage, rollout.returns = _gae(
        rollout.reward, rollout.value, rollout.done, rollout.last_value,
        gamma, lam)
    rollout.cost_advantage, rollout.cost_returns = _gae(
        rollout.cost_target, rollout.cost, rollout.done, rollout.last_cost,
        gamma, lam)
    return rollout


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def _actor_args():
    return SimpleNamespace(
        hidden_size=64, layer_N=1, use_ReLU=True, use_orthogonal=True,
        use_feature_normalization=True, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, recurrent_N=1, gain=0.01,
        stacked_frames=1, use_policy_active_masks=True,
        std_x_coef=1.0, std_y_coef=0.5)


def load_base_actors(env, device, ckpt_dir=None):
    from mappo_lagrangian.algorithms.r_mappo.algorithm.r_actor_critic import (
        R_Actor,
    )
    ckpt = Path(ckpt_dir) if ckpt_dir is not None else BASE_MODELS
    actors = []
    for i in range(env.num_agents):
        actor = R_Actor(_actor_args(), env.env.observation_space[i],
                        env.env.action_space[i], device)
        actor.load_state_dict(torch.load(
            ckpt / f'actor_agent{i}.pt', map_location=device))
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        actors.append(actor)
    return actors


def base_action(actors, obs, rnn, masks, device):
    actions = []
    new_rnn = []
    for i, actor in enumerate(actors):
        o = torch.as_tensor(obs[i][None], dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, r = actor(o, rnn[i], masks, None, True)
        actions.append(action.cpu().numpy()[0])
        new_rnn.append(r)
    return np.stack(actions), new_rnn


def warmstart_task_head(env, model, actors, steps, device, seed=50):
    """Behavior-clone the learned base actor into the task head only."""
    obs, h, grad, hv, hs, g = env.reset(seed=seed)
    rnn = [torch.zeros(1, 1, 64, device=device) for _ in range(env.num_agents)]
    masks = torch.ones(1, 1, device=device)
    rows = []
    for _ in range(steps):
        actions, rnn = base_action(actors, obs, rnn, masks, device)
        rows.append((obs.copy(), h.copy(), grad.copy(), actions.copy()))
        obs, h, grad, hv, hs, g, term, trunc, info = env.step(actions)
        if term or trunc:
            obs, h, grad, hv, hs, g = env.reset(seed=seed + len(rows))
            rnn = [torch.zeros(1, 1, 64, device=device)
                   for _ in range(env.num_agents)]
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(20):
        perm = np.random.permutation(len(rows))
        for start in range(0, len(rows), 64):
            batch = [rows[i] for i in perm[start:start + 64]]
            ob = _t(np.stack([r[0] for r in batch]), device)
            hh = _t(np.stack([r[1] for r in batch]), device)
            gg = _t(np.stack([r[2] for r in batch]), device)
            aa = _t(np.stack([r[3] for r in batch]), device)
            out = model(ob, hh, gg)
            loss = (out['task'] - aa).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    # Start with zero correction; the base task head is the initial walker.
    with torch.no_grad():
        model.correction_head.weight.zero_()
        model.correction_head.bias.zero_()
    return float(loss.detach())


def collect(env, model, steps, device, seed, cfg):
    model.eval()
    obs, h, grad, h_vec, h_scalar, g = env.reset(seed=seed)
    prev_geom = env.get_geometry()
    episode = 0
    ep_ret = 0.0
    ep_cost = []
    ep_wall = []
    ep_topple = []
    ep_h = []
    completed = []
    b = {k: [] for k in (
        'obs', 'h_vec', 'grad', 'h', 'h_next', 'action_raw', 'action_env',
        'log_prob', 'value', 'cost', 'reward', 'cost_target', 'official_cost',
        'terminal', 'done', 'delta_state')}
    last_done = False
    for _ in range(steps):
        obs_t = _t(obs, device)
        h_t = _t(h, device)
        grad_t = _t(grad, device)
        with torch.no_grad():
            out = model.sample(obs_t, h_t, grad_t)
        action = out['env_action'].cpu().numpy()
        (next_obs, next_h, next_grad, next_hv, next_hs, next_g,
         term, trunc, info) = env.step(action)
        geom = env.get_geometry()
        delta = geom - prev_geom
        done = bool(term or trunc)
        # Task return is not contaminated by safety penalties.  The safety
        # branch receives its own Lagrangian target below.
        reward = float(cfg['reward_scale']) * float(info['reward'])
        cost_target = (float(cfg['lambda_cost']) * float(info['cost'])
                       + float(cfg['lambda_survival']) * float(term))
        b['obs'].append(obs_t.cpu())
        b['h_vec'].append(h_t.cpu())
        b['grad'].append(grad_t.cpu())
        b['h'].append(torch.full((env.num_agents,), float(h_scalar),
                                 dtype=torch.float32))
        b['h_next'].append(torch.full((env.num_agents,), float(next_hs),
                                     dtype=torch.float32))
        b['action_raw'].append(out['raw_action'].cpu())
        b['action_env'].append(out['env_action'].cpu())
        b['log_prob'].append(out['log_prob'].cpu())
        b['value'].append(out['value'].cpu())
        b['cost'].append(out['cost'].cpu())
        b['reward'].append(torch.full((env.num_agents,), float(reward),
                                     dtype=torch.float32))
        b['cost_target'].append(torch.full((env.num_agents,), float(cost_target),
                                          dtype=torch.float32))
        b['official_cost'].append(torch.full(
            (env.num_agents,), float(info['cost']), dtype=torch.float32))
        b['terminal'].append(torch.full(
            (env.num_agents,), float(term), dtype=torch.float32))
        b['done'].append(torch.tensor(done, dtype=torch.bool))
        b['delta_state'].append(_t(np.repeat(delta[None, :], env.num_agents, axis=0),
                                   device).cpu())
        ep_ret += reward
        ep_cost.append(float(info['cost']))
        ep_wall.append(float(info['wall']))
        ep_topple.append(float(info['topple']))
        ep_h.append(float(h_scalar))
        last_done = done
        if done:
            completed.append(dict(
                return_=ep_ret,
                cost_rate=float(np.mean(ep_cost)),
                wall_rate=float(np.mean(ep_wall)),
                topple_rate=float(np.mean(ep_topple)),
                safe_episode=float(np.max(ep_cost) == 0),
                survived=float(trunc),
                length=len(ep_cost),
                h_mean=float(np.mean(ep_h)),
            ))
            episode += 1
            obs, h, grad, h_vec, h_scalar, g = env.reset(seed=seed + episode)
            prev_geom = env.get_geometry()
            ep_ret = 0.0
            ep_cost.clear(); ep_wall.clear(); ep_topple.clear(); ep_h.clear()
        else:
            obs, h, grad, h_vec, h_scalar, g = (
                next_obs, next_h, next_grad, next_hv, next_hs, next_g)
            prev_geom = geom
    if last_done:
        last_value = torch.zeros(env.num_agents, dtype=torch.float32)
        last_cost = torch.zeros_like(last_value)
    else:
        with torch.no_grad():
            out = model(_t(obs, device), _t(h, device), _t(grad, device))
        last_value = out['value'].cpu()
        last_cost = out['cost'].cpu()
    return AntRollout(
        obs=torch.stack(b['obs']).float(),
        h_vec=torch.stack(b['h_vec']).float(),
        grad=torch.stack(b['grad']).float(),
        h=torch.stack(b['h']).float(),
        h_next=torch.stack(b['h_next']).float(),
        action_raw=torch.stack(b['action_raw']).float(),
        action_env=torch.stack(b['action_env']).float(),
        log_prob=torch.stack(b['log_prob']).float(),
        value=torch.stack(b['value']).float(),
        cost=torch.stack(b['cost']).float(),
        reward=torch.stack(b['reward']).float(),
        cost_target=torch.stack(b['cost_target']).float(),
        official_cost=torch.stack(b['official_cost']).float(),
        terminal=torch.stack(b['terminal']).float(),
        done=torch.stack(b['done']),
        delta_state=torch.stack(b['delta_state']).float(),
        last_value=last_value.float(),
        last_cost=last_cost.float(),
        completed=completed,
    )


def _set_branch_modes(sub, task_trainable, safety_trainable):
    task_modules = [sub.task_encoder, sub.task_head, sub.task_value_head]
    for module in task_modules:
        for p in module.parameters():
            p.requires_grad = bool(task_trainable)
    for p in [sub.log_std]:
        p.requires_grad = bool(task_trainable)
    safety_modules = [
        sub.safety_encoder, sub.correction_head, sub.feasibility_head,
        sub.value_head, sub.cost_head,
    ]
    for module in safety_modules:
        for p in module.parameters():
            p.requires_grad = bool(safety_trainable)


def task_update(model, opt, rollout, cfg, device):
    """Return-only task update.  Safety parameters are frozen."""
    t, n = rollout.h.shape
    logs = []
    model.train()
    for agent in range(n):
        sub = model.models[agent]
        _set_branch_modes(sub, task_trainable=True, safety_trainable=False)
        obs = rollout.obs[:, agent].to(device)
        h_vec = rollout.h_vec[:, agent].to(device)
        grad = rollout.grad[:, agent].to(device)
        raw = rollout.action_raw[:, agent].to(device)
        old_logp = rollout.log_prob[:, agent].to(device)
        old_value = rollout.value[:, agent].to(device)
        adv = _normalize(rollout.advantage[:, agent].to(device))
        ret = rollout.returns[:, agent].to(device)
        bs = min(int(cfg['minibatch_size']), t)
        for _ in range(int(cfg['epochs'])):
            perm = torch.randperm(t, device=device)
            for start in range(0, t, bs):
                idx = perm[start:start + bs]
                out = sub.evaluate_actions(
                    obs[idx], h_vec[idx], grad[idx], raw[idx])
                ratio = torch.exp(out['log_prob'] - old_logp[idx])
                clipped = torch.clamp(
                    ratio, 1 - cfg['clip_range'], 1 + cfg['clip_range'])
                policy = -torch.min(
                    ratio * adv[idx], clipped * adv[idx]).mean()
                vclip = old_value[idx] + torch.clamp(
                    out['value'] - old_value[idx],
                    -cfg['clip_range'], cfg['clip_range'])
                vloss = 0.5 * torch.maximum(
                    (out['value'] - ret[idx]).square(),
                    (vclip - ret[idx]).square()).mean()
                ent = out['entropy'].mean()
                loss = (policy + cfg['task_value_coef'] * vloss
                        - cfg['entropy_coef'] * ent)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite task loss {loss.item()}')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sub.parameters() if p.requires_grad],
                    cfg['max_grad_norm'])
                opt.step()
                logs.append(dict(
                    task_loss=float(loss.detach()),
                    task_policy=float(policy.detach()),
                    task_value=float(vloss.detach()),
                    task_entropy=float(ent.detach())))
    model.eval()
    return {k: float(np.mean([r[k] for r in logs])) for k in logs[0]}


def safety_update(model, opt, rollout, cfg, device):
    """Safety-only update: cost/survival/h.  Return is not in this loss."""
    t, n = rollout.h.shape
    logs = []
    model.train()
    for agent in range(n):
        sub = model.models[agent]
        _set_branch_modes(sub, task_trainable=False, safety_trainable=True)
        obs = rollout.obs[:, agent].to(device)
        h_vec = rollout.h_vec[:, agent].to(device)
        grad = rollout.grad[:, agent].to(device)
        h = rollout.h[:, agent].to(device)
        h_next = rollout.h_next[:, agent].to(device)
        raw = rollout.action_raw[:, agent].to(device)
        old_logp = rollout.log_prob[:, agent].to(device)
        delta = rollout.delta_state[:, agent].to(device)
        cadv = _normalize(rollout.cost_advantage[:, agent].to(device))
        cret = rollout.cost_returns[:, agent].to(device)
        bs = min(int(cfg['minibatch_size']), t)
        for _ in range(int(cfg['epochs'])):
            perm = torch.randperm(t, device=device)
            for start in range(0, t, bs):
                idx = perm[start:start + bs]
                out = sub.evaluate_actions(
                    obs[idx], h_vec[idx], grad[idx], raw[idx])
                ratio = torch.exp(out['log_prob'] - old_logp[idx])
                closs = 0.5 * (out['cost'] - cret[idx]).square().mean()
                sv_loss = 0.5 * (out['safety_value'] - cret[idx]).square().mean()
                cost_actor = (
                    torch.clamp(ratio, 0.0, 2.0) * cadv[idx]).mean()
                inv = forward_invariance_loss(
                    h[idx], h_next[idx], cfg['alpha'])
                gl = gradient_direction_loss(
                    h[idx], grad[idx], delta[idx],
                    margin=cfg['gradient_margin'])
                bar = barrier_loss(h[idx], margin=cfg['barrier_margin'])
                corr = correction_penalty(out['delta'])
                loss = (cfg['cost_actor_coef'] * cost_actor
                        + cfg['cost_coef'] * closs
                        + cfg['safety_value_coef'] * sv_loss
                        + cfg['invariance_coef'] * inv
                        + cfg['gradient_coef'] * gl
                        + cfg['barrier_coef'] * bar
                        + cfg['correction_coef'] * corr)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite safety loss {loss.item()}')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sub.parameters() if p.requires_grad],
                    cfg['max_grad_norm'])
                opt.step()
                logs.append(dict(
                    safety_loss=float(loss.detach()),
                    safety_actor=float(cost_actor.detach()),
                    safety_cost_value=float(closs.detach()),
                    safety_value=float(sv_loss.detach()),
                    safety_invariance=float(inv.detach()),
                    safety_gradient=float(gl.detach()),
                    safety_barrier=float(bar.detach()),
                    safety_correction=float(corr.detach())))
    model.eval()
    return {k: float(np.mean([r[k] for r in logs])) for k in logs[0]}


@torch.no_grad()
def evaluate(env, model, seeds, device, cfg):
    model.eval()
    rows = []
    for seed in seeds:
        obs, h, grad, hv, hs, g = env.reset(seed=seed)
        ep_ret = 0.0
        costs, walls, topples, h_values = [], [], [], []
        h_vecs = []
        survival_risks = []
        length = 0
        x_max = float(env.get_torso_xyz()[0])
        x_final = x_max
        for _ in range(env.max_step):
            out = model(_t(obs, device), _t(h, device), _t(grad, device))
            action = out['action'].cpu().numpy()
            obs, h, grad, hv, h_scalar_now, g, term, trunc, info = env.step(action)
            ep_ret += float(info['reward'])
            costs.append(float(info['cost']))
            walls.append(float(info['wall']))
            topples.append(float(info['topple']))
            h_values.append(float(info['h']))
            h_vecs.append(np.asarray(info['h_vec'], dtype=np.float64).copy())
            survival_risks.append(float(info.get('survival_risk', 0.0)))
            x_final = float(info.get('x', x_final))
            x_max = max(x_max, x_final)
            length += 1
            if term or trunc:
                break
        hv_arr = np.asarray(h_vecs, dtype=np.float64).reshape(-1, 4)
        cost_arr = np.asarray(costs, dtype=np.float64)
        h_neg = hv_arr.min(axis=1) < 0.0
        rows.append(dict(
            return_=ep_ret,
            cost_rate=float(np.mean(costs)) if costs else 0.0,
            wall_rate=float(np.mean(walls)) if walls else 0.0,
            topple_rate=float(np.mean(topples)) if topples else 0.0,
            safe_episode=float(np.max(costs) == 0) if costs else 0.0,
            wall_episode=float(np.max(walls) > 0) if walls else 1.0,
            topple_episode=float(np.max(topples) > 0) if topples else 1.0,
            survived=float(length >= env.max_step),
            length=length,
            h_mean=float(np.mean(h_values)) if h_values else 0.0,
            h_min=float(np.min(h_values)) if h_values else 0.0,
            h_neg_frac=float(np.mean(h_neg)) if h_values else 0.0,
            h_wall_neg_frac=float(np.mean(hv_arr[:, 0] < 0.0)) if h_values else 0.0,
            h_floor_neg_frac=float(np.mean(hv_arr[:, 1] < 0.0)) if h_values else 0.0,
            h_ceil_neg_frac=float(np.mean(hv_arr[:, 2] < 0.0)) if h_values else 0.0,
            h_upright_neg_frac=float(np.mean(hv_arr[:, 3] < 0.0)) if h_values else 0.0,
            h_false_alarm=float(np.mean(h_neg & (cost_arr == 0.0))) if h_values else 0.0,
            h_miss=float(np.mean((~h_neg) & (cost_arr > 0.0))) if h_values else 0.0,
            survival_risk_mean=float(np.mean(survival_risks)) if survival_risks else 0.0,
            x_final=x_final,
            x_max=x_max,
        ))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(ROOT / 'outputs/v5/ant/smoke'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warmstart-steps', type=int, default=2000)
    ap.add_argument('--rollout-steps', type=int, default=1024)
    ap.add_argument('--updates', type=int, default=2)
    ap.add_argument('--eval-episodes', type=int, default=3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--unsafe-penalty', type=float, default=0.1)
    ap.add_argument('--reward-scale', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--task-lr', type=float, default=1e-5)
    ap.add_argument('--ckpt', default=None, help='upstream MAPPO actor directory; default is pure MAPPO 10M')
    ap.add_argument('--correction-radius', type=float, default=0.05)
    ap.add_argument('--gate-margin', type=float, default=0.1)
    ap.add_argument('--gate-tau', type=float, default=0.05)
    ap.add_argument('--initial-log-std', type=float, default=-3.0)
    ap.add_argument('--gate-on-wall', action='store_true', help='gate correction on h_wall instead of min(h)')
    ap.add_argument('--h-kappa', type=float, default=0.0, help='predictive wall-approach margin coefficient')
    ap.add_argument('--h-kappa-z', type=float, default=0.0, help='predictive vertical margin coefficient')
    ap.add_argument('--safety-h-checkpoint', default=None,
                    help='official learned-h checkpoint; if set, replaces analytic Ant h')
    ap.add_argument('--cost-limit', type=float, default=0.02)
    ap.add_argument('--survival-limit', type=float, default=0.95)
    ap.add_argument('--lambda-lr', type=float, default=0.1)
    ap.add_argument('--eval-interval', type=int, default=20)
    ap.add_argument('--selection-episodes', type=int, default=3)
    ap.add_argument('--selection-seed', type=int, default=20000)
    ap.add_argument('--survival-penalty', type=float, default=0.5)
    ap.add_argument('--barrier-coef', type=float, default=0.5)
    ap.add_argument('--gradient-coef', type=float, default=0.2)
    ap.add_argument('--invariance-coef', type=float, default=0.5)
    ap.add_argument('--cost-actor-coef', type=float, default=0.2)
    ap.add_argument('--cost-coef', type=float, default=0.5)
    ap.add_argument('--task-value-coef', type=float, default=0.5)
    ap.add_argument('--safety-value-coef', type=float, default=0.2)
    ap.add_argument('--return-tolerance', type=float, default=0.10,
                    help='validation return guardrail relative to the task prior')
    ap.add_argument('--base-val-episodes', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--freeze-task', action='store_true', help='freeze the warm-started task branch; otherwise task-lr is used')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(gamma=0.99, lam=0.95, clip_range=0.2,
               minibatch_size=256, value_coef=0.5, cost_coef=args.cost_coef,
               task_value_coef=args.task_value_coef,
               safety_value_coef=args.safety_value_coef,
               cost_actor_coef=args.cost_actor_coef,
               invariance_coef=args.invariance_coef,
               gradient_coef=args.gradient_coef,
               barrier_coef=args.barrier_coef, barrier_margin=0.05,
               correction_coef=1e-3, entropy_coef=0.001, alpha=0.1,
               gradient_margin=0.1, max_grad_norm=0.5,
               unsafe_penalty=args.unsafe_penalty,
               survival_penalty=args.survival_penalty,
               reward_scale=args.reward_scale, epochs=args.epochs,
               lambda_cost=args.unsafe_penalty,
               lambda_survival=args.survival_penalty,
               cost_limit=args.cost_limit,
               survival_limit=args.survival_limit,
               lambda_lr=args.lambda_lr)
    device = torch.device(args.device)
    safety_provider = None
    if args.safety_h_checkpoint:
        from v5.safety.checkpoint import LearnedSafetyProvider
        h_path = Path(args.safety_h_checkpoint).expanduser()
        if not h_path.is_absolute():
            h_path = (Path.cwd() / h_path).resolve()
        args.safety_h_checkpoint = str(h_path)
        safety_provider = LearnedSafetyProvider.from_checkpoint(
            h_path, device=device)
    env = Ant2x4Env(seed=args.seed, episode_limit=1000,
                    h_kappa=args.h_kappa, h_kappa_z=args.h_kappa_z,
                    safety_provider=safety_provider)
    model = QpFreeAntEnsemble(env.num_agents, obs_dim=env.observation_dim,
                              hidden=128, correction_radius=args.correction_radius,
                              action_dim=env.action_dim, h_dim=4,
                              grad_dim=9, gate_margin=args.gate_margin,
                              gate_tau=args.gate_tau,
                              gate_on_wall=args.gate_on_wall,
                              initial_log_std=args.initial_log_std).to(device)
    # Load only for warm-start; it is not used by the V5 actor at inference.
    actors = load_base_actors(env, device, args.ckpt)
    # Exact task-branch initialization from the learned MAPPO actor; no
    # BC approximation and no expert u_ref.  The correction branch starts at
    # zero, so the initial policy is the base actor.
    model.load_base_actors_task(actors)
    bc = 0.0
    task_params = []
    safety_params = []
    for m in model.models:
        task_params += list(m.task_encoder.parameters())
        task_params += list(m.task_head.parameters())
        task_params += list(m.task_value_head.parameters())
        task_params += [m.log_std]
        safety_params += [p for n, p in m.named_parameters()
                          if not n.startswith('task_encoder.')
                          and not n.startswith('task_head.')
                          and not n.startswith('task_value_head.')
                          and n != 'log_std']
    if args.freeze_task:
        for p in task_params:
            p.requires_grad = False
        task_opt = None
    else:
        task_opt = torch.optim.Adam(task_params, lr=args.task_lr)
    safety_opt = torch.optim.Adam(safety_params, lr=args.lr)
    history = []
    best_state = None
    best_metrics = None
    best_score = -float('inf')
    # Task-prior validation is used only as a return guardrail.  It is not an
    # h objective and is not used to update the safety branch.
    base_val = evaluate(
        env, model,
        list(range(args.selection_seed,
                   args.selection_seed + args.base_val_episodes)),
        device, cfg)
    return_floor = base_val['return_'] * (1.0 - args.return_tolerance)
    for u in range(args.updates):
        rollout = collect(env, model, args.rollout_steps, device,
                          seed=args.seed + 1000 * u, cfg=cfg)
        add_gae(rollout, cfg['gamma'], cfg['lam'])
        logs = {}
        if task_opt is not None:
            logs.update(task_update(model, task_opt, rollout, cfg, device))
        logs.update(safety_update(model, safety_opt, rollout, cfg, device))
        logs['update'] = u
        logs['bc_loss'] = bc
        logs['completed'] = rollout.completed[-3:]
        history.append(logs)
        # Update constrained-objective multipliers from the rollout.
        mean_cost = float(rollout.official_cost.mean().item())
        if rollout.completed:
            survival = float(np.mean([c['survived'] for c in rollout.completed]))
        else:
            survival = 1.0
        cfg['lambda_cost'] = max(
            0.0, cfg['lambda_cost'] + cfg['lambda_lr'] * (mean_cost - cfg['cost_limit']))
        cfg['lambda_survival'] = max(
            0.0, cfg['lambda_survival'] + cfg['lambda_lr'] *
            (cfg['survival_limit'] - survival))
        logs['mean_cost'] = mean_cost
        logs['survival'] = survival
        logs['lambda_cost'] = cfg['lambda_cost']
        logs['lambda_survival'] = cfg['lambda_survival']
        # Validation-based feasible selection.  Return is a task guardrail, not
        # an h/safety objective; h parameters receive no return gradient.
        if args.eval_interval > 0 and (
                (u + 1) % args.eval_interval == 0 or u + 1 == args.updates):
            val = evaluate(
                env, model,
                list(range(args.selection_seed,
                           args.selection_seed + args.selection_episodes)),
                device, cfg)
            feasible = (val['cost_rate'] <= args.cost_limit + 1e-9 and
                        val['survived'] >= args.survival_limit - 1e-9 and
                        val['return_'] >= return_floor - 1e-9)
            score = (val['return_']
                     - 1000.0 * max(0.0, val['cost_rate'] - args.cost_limit)
                     - 1000.0 * max(0.0, args.survival_limit - val['survived'])
                     - 1000.0 * max(0.0, return_floor - val['return_'])
                     / max(abs(base_val['return_']), 1.0))
            if (best_state is None or (feasible and not best_metrics.get('_feasible', False))
                    or (feasible == best_metrics.get('_feasible', False) and score > best_score)):
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = dict(val, _feasible=feasible)
                best_score = score
            logs['selection'] = dict(val=val, feasible=feasible, score=score)
        print(json.dumps(logs), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_metrics = evaluate(env, model, list(range(args.seed, args.seed + args.eval_episodes)),
                            device, cfg)
    payload = dict(
        config=vars(args), training=cfg, bc_loss=bc,
        eval=eval_metrics, history=history,
        base_validation=base_val, return_floor=return_floor,
        best_selection=best_metrics,
        best_score=best_score,
        note=('QP-free Ant V5: task return branch and safety-only h/cost/survival '
              'branch are updated separately; return is a task guardrail, not '
              'an h objective'))
    (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    torch.save(dict(schema='v5-ant-v5', model=model.state_dict(),
                    config=payload, obs_dim=env.observation_dim,
                    action_dim=env.action_dim), out / 'policy_smoke.pth')
    if best_state is not None:
        torch.save(dict(schema='v5-ant-v5-best', model=best_state,
                        config=payload, obs_dim=env.observation_dim,
                        action_dim=env.action_dim), out / 'policy_best.pth')
    print(json.dumps(dict(eval=eval_metrics), indent=2))
    env.close()


if __name__ == '__main__':
    main()
