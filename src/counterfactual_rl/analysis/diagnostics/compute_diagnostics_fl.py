"""
Offline CCE diagnostics for FrozenLake — compute stage (GPU/CPU, run on SLURM).

Same idea as the Connect Four diagnostic, but the ground-truth oracle is EXACT:
FrozenLake is a small known MDP, so we get optimal Q*(s,a) by value iteration
(value_iteration.py) instead of an MCTS approximation. The oracle depends only on
the env, so it's computed once and shared across all checkpoints/seeds.

Per checkpoint we record, for ~1000 visited transitions:
  cce_score      consequence score (reuses _compiled_rollout_fn + compute_consequence_metric)
  td_error       standalone Bellman error (online vs target net)
  truth_qvalues  exact Q*(s,·)           (4,)
  truth_spread   max_a Q* - min_a Q*     (ground-truth stakes)
  truth_regret   max_a Q* - Q*(s,taken)

Usage:
    python -m counterfactual_rl.analysis.diagnostics.compute_diagnostics_fl \
        --runs-root <.../frozen_lake/runs> \
        --run-ids 255495 255496 255497 \
        --n-chunks 10 --n-transitions 1000 --epsilon 0.05 \
        --out <.../docs/figures/diagnostics_fl/diagnostics.npz>
"""

import argparse
import glob
import os
import pickle
import re
import time

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.consequence_dqn_vectorized import (
    FrozenLakeConsequenceDQNVectorized,
)
from counterfactual_rl.analysis.metrics import compute_consequence_metric
from counterfactual_rl.analysis.diagnostics.value_iteration import compute_qstar, stakes_from_qstar

ALL_ACTIONS = jnp.arange(4, dtype=jnp.int32)

# Fixed 8x8 slippery map (envs/frozen_lake.py). Stored in the npz for the D8 grid render.
MAP_8x8 = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
           "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]


def build_td_fn(network, gamma):
    def td(params, target_params, states, next_states, actions, rewards, dones):
        q = jax.vmap(network.apply, in_axes=(None, 0))(params, states)             # (B,4)
        q_taken = jnp.take_along_axis(q, actions[:, None], axis=1)[:, 0]           # (B,)
        next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
        max_next_q = next_q.max(axis=1)
        targets = rewards + gamma * max_next_q * (1.0 - dones)
        return targets - q_taken
    return jax.jit(td)


def generate_transitions(agent, n_transitions, epsilon):
    n_envs = agent.config.get('n_envs', 256)
    collect_steps = agent.config.get('collect_steps', 128)
    agent._key, ik, ck, sub = jax.random.split(agent._key, 4)
    init_keys = jax.random.split(ik, n_envs)
    current = jax.vmap(lambda k: agent.env.reset(k)[1])(init_keys)
    step_keys = jax.random.split(ck, n_envs * collect_steps).reshape(n_envs, collect_steps, 2)
    current, outputs = agent._collect_fn(agent.params, jnp.float32(epsilon), current, step_keys)
    jax.block_until_ready(outputs)
    state, action, reward, next_state, done = (np.asarray(outputs[i]).reshape(-1) for i in range(5))
    total = state.shape[0]
    idx = np.asarray(jax.random.choice(sub, total, shape=(min(n_transitions, total),), replace=False))
    return {
        'state':      state[idx].astype(np.int32),
        'action':     action[idx].astype(np.int32),
        'reward':     reward[idx].astype(np.float32),
        'next_state': next_state[idx].astype(np.int32),
        'done':       done[idx].astype(np.float32),
    }


def cce_scores(agent, states, actions, cce_batch=256):
    B = states.shape[0]
    N = agent.cf_n_rollouts
    returns = np.zeros((B, 4, N), dtype=np.float32)
    for lo in range(0, B, cce_batch):
        hi = min(lo + cce_batch, B)
        sa = jnp.asarray(states[lo:hi], dtype=jnp.int32)
        agent._key, sk = jax.random.split(agent._key)
        keys = jax.random.split(sk, (hi - lo) * 4 * N).reshape(hi - lo, 4, N, 2)
        out = agent._compiled_rollout_fn(agent.params, sa, ALL_ACTIONS, keys)
        returns[lo:hi] = np.asarray(jax.block_until_ready(out))
    scores = np.zeros(B, dtype=np.float32)
    for i in range(B):
        dists = {(a,): returns[i, a] for a in range(4)}
        scores[i] = compute_consequence_metric(
            (int(actions[i]),), dists,
            metric=agent.consequence_metric, aggregation=agent.consequence_aggregation,
        )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def process_checkpoint(agent, td_fn, qstar, stakes, ckpt_path, n_transitions, epsilon):
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    agent.params = jax.tree.map(jnp.array, ck['params'])
    agent.target_params = jax.tree.map(jnp.array, ck['target_params'])

    b = generate_transitions(agent, n_transitions, epsilon)
    s, a = b['state'], b['action']

    cce = cce_scores(agent, s, a)
    td = np.asarray(td_fn(
        agent.params, agent.target_params,
        jnp.asarray(s), jnp.asarray(b['next_state']),
        jnp.asarray(a), jnp.asarray(b['reward']), jnp.asarray(b['done']),
    )).astype(np.float32)

    qv = qstar[s]                                   # (B,4) exact optimal action-values
    return {
        'state':         s,
        'taken_action':  a,
        'reward':        b['reward'],
        'cce_score':     cce,
        'td_error':      td,
        'truth_qvalues': qv,
        'truth_spread':  stakes[s],
        'truth_regret':  (qv.max(axis=1) - qv[np.arange(len(s)), a]).astype(np.float32),
    }


def select_checkpoints(run_dir, n_chunks):
    files = glob.glob(os.path.join(run_dir, 'checkpoints', 'ckpt_*.pkl'))
    parsed = sorted((int(re.search(r'ckpt_(\d+)\.pkl', f).group(1)), f) for f in files)
    if not parsed:
        return []
    if len(parsed) <= n_chunks:
        return parsed
    idx = np.linspace(0, len(parsed) - 1, n_chunks).round().astype(int)
    return [parsed[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--run-ids', nargs='+', required=True)
    ap.add_argument('--n-chunks', type=int, default=10)
    ap.add_argument('--n-transitions', type=int, default=1000)
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    first = select_checkpoints(os.path.join(args.runs_root, args.run_ids[0]), args.n_chunks)[0][1]
    with open(first, 'rb') as f:
        cfg = pickle.load(f)['config']
    print(f"Config: map={cfg.get('map_name')} slippery={cfg.get('is_slippery')} "
          f"cf_horizon={cfg.get('cf_horizon')} cf_n_rollouts={cfg.get('cf_n_rollouts')} "
          f"metric={cfg.get('consequence_metric')}", flush=True)

    agent = FrozenLakeConsequenceDQNVectorized(config=cfg)
    agent._build_rollout_fn()
    agent._build_collect_fn()
    td_fn = build_td_fn(agent.network, agent.gamma)

    # Exact ground-truth oracle — computed ONCE (depends only on the env).
    qstar = compute_qstar(agent.env, gamma=agent.gamma)     # (S,4)
    stakes = stakes_from_qstar(qstar)                       # (S,)
    print(f"Q* oracle: stakes mean={stakes.mean():.3f} max={stakes.max():.3f} "
          f"({int((stakes>0.05).sum())}/{stakes.size} states with stakes>0.05)", flush=True)

    rows = {k: [] for k in ('seed', 'chunk', 'state', 'taken_action', 'reward',
                            'cce_score', 'td_error', 'truth_qvalues', 'truth_spread', 'truth_regret')}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def dump():
        merged = {k: np.concatenate(v, axis=0) for k, v in rows.items() if v}
        merged['fl_map'] = np.array(MAP_8x8)
        merged['fl_ncols'] = np.int64(8)
        np.savez_compressed(args.out, **merged)
        return merged

    for run_id in args.run_ids:
        for episode, ckpt in select_checkpoints(os.path.join(args.runs_root, run_id), args.n_chunks):
            t0 = time.time()
            out = process_checkpoint(agent, td_fn, qstar, stakes, ckpt, args.n_transitions, args.epsilon)
            n = out['cce_score'].shape[0]
            rows['seed'].append(np.full(n, int(run_id), dtype=np.int64))
            rows['chunk'].append(np.full(n, episode, dtype=np.int64))
            for k, v in out.items():
                rows[k].append(v)
            merged = dump()
            print(f"  run {run_id} ep {episode}: {n} trans  "
                  f"CCE[mean={out['cce_score'].mean():.3f} max={out['cce_score'].max():.3f}]  "
                  f"({time.time()-t0:.0f}s)  [{merged['cce_score'].shape[0]} rows]", flush=True)

    print(f"\nDone — {merged['cce_score'].shape[0]} rows → {args.out}", flush=True)


if __name__ == '__main__':
    main()
