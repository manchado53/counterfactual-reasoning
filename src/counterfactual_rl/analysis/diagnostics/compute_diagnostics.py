"""
Offline CCE diagnostics — compute stage (GPU, run on SLURM).

For each trained checkpoint we:
  1. Load it and generate fresh transitions by rolling the loaded policy vs the
     training opponent (MCTS-32), keeping the pgx state of each decision.
  2. CCE-score each transition (reuses the exact training scoring path:
     _build_batched_rollout_fn + _compiled_batched_fn + compute_consequence_metric).
  3. Compute the standalone TD error for each (online vs target net, dqn.py:142).
  4. Compute a strong-MCTS (n_sims=200) value per action as ground truth
     (policy_output.search_tree.summary().qvalues).

All three numbers are aligned per transition and dumped to one .npz, tagged by
(seed, chunk), so the plot stage can answer Q1/Q2/Q3.

Usage:
    python -m counterfactual_rl.analysis.diagnostics.compute_diagnostics \
        --runs-root <.../agents/shared/runs> \
        --run-ids 259281 259282 259283 \
        --chunks 10 20 30 40 50 60 70 80 90 100 \
        --n-transitions 1000 --epsilon 0.05 --mcts-sims 200 \
        --out <.../docs/figures/diagnostics/diagnostics.npz>
"""

import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import mctx

from counterfactual_rl.agents.connect_four.consequence_dqn import Connect4ConsequenceDQN
from counterfactual_rl.agents.connect_four.opponent_mcts import _recurrent_fn, C4_ACTIONS
from counterfactual_rl.utils.action_selection import beam_search_top_k_joint_actions
from counterfactual_rl.analysis.metrics import compute_consequence_metric

ENV_INFO = {'obs_dim': 84, 'num_agents': 1, 'actions_per_agent': 7}


# ── Ground truth: strong-MCTS per-action Q-values ─────────────────────────────

def _build_mcts_values_fn(n_sims):
    """Return a jitted fn mapping a batch of pgx states → (B, 7) root Q-values.

    Reuses opponent_mcts._recurrent_fn so the search dynamics match the training
    opponent exactly; only the simulation budget differs (n_sims vs 32).
    """
    def mcts_values_single(state, rng_key):
        batched = jax.tree.map(lambda x: x[None], state)
        root = mctx.RootFnOutput(
            prior_logits=jnp.zeros((1, C4_ACTIONS)),
            value=jnp.zeros(1),
            embedding=batched,
        )
        out = mctx.gumbel_muzero_policy(
            params=None, rng_key=rng_key, root=root,
            recurrent_fn=jax.vmap(_recurrent_fn, in_axes=(None, None, 0, 0)),
            num_simulations=n_sims,
            invalid_actions=~state.legal_action_mask[None],
            max_depth=42,
            max_num_considered_actions=C4_ACTIONS,
        )
        return out.search_tree.summary().qvalues[0]  # (7,)

    return jax.jit(jax.vmap(mcts_values_single, in_axes=(0, 0)))


# ── Standalone TD error (mirrors dqn.py update_step loss_fn) ───────────────────

def _build_td_error_fn(network, gamma):
    def td_error(params, target_params, obs, next_obs, actions, rewards, dones, next_masks):
        q = jax.vmap(network.apply, in_axes=(None, 0))(params, obs)          # (B,1,7)
        q_taken = jnp.take_along_axis(q, actions[:, None, None], axis=-1)     # (B,1,1)
        q_taken = q_taken.reshape(-1)                                         # (B,)

        next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_obs)[:, 0, :]  # (B,7)
        next_q = jnp.where(next_masks, next_q, -jnp.inf)
        max_next_q = next_q.max(axis=-1)                                     # (B,)
        targets = rewards + gamma * jnp.where(dones > 0.5, jnp.float32(0.0), max_next_q)
        return targets - q_taken                                            # (B,)

    return jax.jit(td_error)


# ── Transition generation (reuses the trained collect fn) ──────────────────────

def generate_transitions(agent, n_transitions, epsilon):
    """Roll the loaded policy vs the config opponent; return a flat, subsampled batch."""
    agent._key, collect_key, sub_key = jax.random.split(agent._key, 3)
    env_keys = jax.random.split(collect_key, agent.n_envs)
    init_states = jax.vmap(agent.pgx_env.init)(env_keys)
    outputs = agent._collect_fn(agent.params, jnp.float32(epsilon), init_states, env_keys)
    jax.block_until_ready(outputs)

    actions, rewards, dones, obs, next_obs, masks, next_masks, saved_state = outputs[:8]
    total = agent.n_envs * agent.collect_steps
    flat = lambda x: np.asarray(x).reshape(total, *x.shape[2:])

    idx = np.asarray(jax.random.choice(sub_key, total, shape=(min(n_transitions, total),),
                                       replace=False))
    batch = {
        'action':     flat(actions)[idx].astype(np.int32),
        'reward':     flat(rewards)[idx].astype(np.float32),
        'done':       flat(dones)[idx].astype(np.float32),
        'obs':        flat(obs)[idx].astype(np.float32),
        'next_obs':   flat(next_obs)[idx].astype(np.float32),
        'next_mask':  flat(next_masks)[idx].astype(bool),
    }
    # saved_state is a pytree with leaves (n_envs, T, ...) → flatten then subsample axis 0.
    flat_state = jax.tree.map(lambda x: x.reshape(total, *x.shape[2:])[idx], saved_state)
    return batch, flat_state


# ── CCE scoring (faithful replica of _score_buffer_transitions, no buffer) ─────

def cce_scores(agent, flat_state, actions_taken, cce_batch=100):
    """Score each transition's consequence using the agent's compiled rollout fn."""
    legal_masks = np.asarray(flat_state.legal_action_mask)  # (B,7)
    B = legal_masks.shape[0]
    K, N = agent.cf_top_k, agent.cf_n_rollouts

    all_actions, all_action_probs, all_actual = [], [], []
    for i in range(B):
        actual = (int(actions_taken[i]),)
        valid_wrapped = [[j for j, v in enumerate(legal_masks[i]) if v]]
        to_eval, probs = beam_search_top_k_joint_actions(valid_wrapped, k=K, return_probs=True)
        if actual not in to_eval:
            to_eval = [actual] + to_eval[:K - 1]
            if actual not in probs:
                probs[actual] = (min(probs.values()) if probs else 0.01) * 0.5
        while len(to_eval) < K:
            to_eval.append(actual)
        all_actions.append(to_eval)
        all_action_probs.append(probs)
        all_actual.append(actual)

    actions_array = jnp.array([[a[0] for a in row] for row in all_actions], dtype=jnp.int32)  # (B,K)

    # Rollouts in memory-bounded sub-batches (B*K*N too large for the T4 at once).
    returns = np.zeros((B, K, N), dtype=np.float32)
    for lo in range(0, B, cce_batch):
        hi = min(lo + cce_batch, B)
        sub_state = jax.tree.map(lambda x: x[lo:hi], flat_state)
        agent._key, subkey = jax.random.split(agent._key)
        keys = jax.random.split(subkey, (hi - lo) * K * N).reshape(hi - lo, K, N, 2)
        out = agent._compiled_batched_fn(agent.params, sub_state, actions_array[lo:hi], keys)
        returns[lo:hi] = np.asarray(jax.block_until_ready(out))

    scores = np.zeros(B, dtype=np.float32)
    for i in range(B):
        dists = {}
        for j, atup in enumerate(all_actions[i]):
            dists.setdefault(atup, returns[i, j])
        scores[i] = compute_consequence_metric(
            all_actual[i], dists,
            metric=agent.consequence_metric,
            action_probs=all_action_probs[i],
            aggregation=agent.consequence_aggregation,
        )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


# ── Per-checkpoint driver ─────────────────────────────────────────────────────

def process_checkpoint(agent, mcts_fn, td_fn, ckpt_path, n_transitions, epsilon,
                       mcts_batch=100):
    """Load params, generate transitions, compute CCE + TD + MCTS-truth; return dict of arrays."""
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    agent.params = jax.tree.map(jnp.array, ck['params'])
    agent.target_params = jax.tree.map(jnp.array, ck['target_params'])

    batch, flat_state = generate_transitions(agent, n_transitions, epsilon)
    B = batch['action'].shape[0]

    cce = cce_scores(agent, flat_state, batch['action'])

    td = np.asarray(td_fn(
        agent.params, agent.target_params,
        jnp.asarray(batch['obs']), jnp.asarray(batch['next_obs']),
        jnp.asarray(batch['action']), jnp.asarray(batch['reward']),
        jnp.asarray(batch['done']), jnp.asarray(batch['next_mask']),
    ))

    qvals = np.zeros((B, C4_ACTIONS), dtype=np.float32)
    for lo in range(0, B, mcts_batch):
        hi = min(lo + mcts_batch, B)
        sub_state = jax.tree.map(lambda x: x[lo:hi], flat_state)
        agent._key, subkey = jax.random.split(agent._key)
        keys = jax.random.split(subkey, hi - lo)
        qvals[lo:hi] = np.asarray(jax.block_until_ready(mcts_fn(sub_state, keys)))

    legal = np.asarray(flat_state.legal_action_mask)  # (B,7)
    q_legal = np.where(legal, qvals, np.nan)
    q_max = np.nanmax(q_legal, axis=1)
    q_min = np.nanmin(q_legal, axis=1)
    taken = batch['action']
    q_taken = qvals[np.arange(B), taken]

    return {
        'taken_action': taken,
        'reward':       batch['reward'],
        'obs':          batch['obs'],
        'legal_mask':   legal,
        'cce_score':    cce,
        'td_error':     td.astype(np.float32),
        'mcts_qvalues': qvals,
        'mcts_spread':  (q_max - q_min).astype(np.float32),
        'mcts_regret':  (q_max - q_taken).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--run-ids', nargs='+', required=True)
    ap.add_argument('--chunks', nargs='+', type=int,
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument('--n-transitions', type=int, default=1000)
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--mcts-sims', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Build the agent once from the first checkpoint's config; reuse across all
    # (run, chunk) pairs — the compiled rollout/MCTS/TD fns take params as args,
    # so swapping params never triggers a recompile.
    first_ckpt = os.path.join(args.runs_root, args.run_ids[0], 'checkpoints',
                              f'ckpt_{args.chunks[0]:07d}.pkl')
    with open(first_ckpt, 'rb') as f:
        cfg = pickle.load(f)['config']
    print(f"Config: cf_horizon={cfg.get('cf_horizon')} cf_n_rollouts={cfg.get('cf_n_rollouts')} "
          f"metric={cfg.get('consequence_metric')} opponent={cfg.get('opponent')}", flush=True)

    agent = Connect4ConsequenceDQN(ENV_INFO, config=cfg)
    agent._build_batched_rollout_fn()
    mcts_fn = _build_mcts_values_fn(args.mcts_sims)
    td_fn = _build_td_error_fn(agent.network, agent.gamma)

    rows = {k: [] for k in ('seed', 'chunk', 'taken_action', 'reward', 'obs', 'legal_mask',
                            'cce_score', 'td_error', 'mcts_qvalues', 'mcts_spread', 'mcts_regret')}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def dump():
        merged = {k: np.concatenate(v, axis=0) for k, v in rows.items() if v}
        np.savez_compressed(args.out, **merged)
        return merged

    for run_id in args.run_ids:
        for chunk in args.chunks:
            ckpt = os.path.join(args.runs_root, run_id, 'checkpoints', f'ckpt_{chunk:07d}.pkl')
            if not os.path.exists(ckpt):
                print(f"  SKIP missing {ckpt}", flush=True)
                continue
            t0 = time.time()
            out = process_checkpoint(agent, mcts_fn, td_fn, ckpt, args.n_transitions, args.epsilon)
            n = out['cce_score'].shape[0]
            rows['seed'].append(np.full(n, int(run_id), dtype=np.int64))
            rows['chunk'].append(np.full(n, chunk, dtype=np.int64))
            for k, v in out.items():
                rows[k].append(v)
            # Write incrementally so a timeout preserves completed checkpoints.
            merged = dump()
            print(f"  run {run_id} chunk {chunk}: {n} transitions  "
                  f"CCE[mean={out['cce_score'].mean():.3f} max={out['cce_score'].max():.3f}]  "
                  f"({time.time()-t0:.0f}s)  [{merged['cce_score'].shape[0]} rows saved]", flush=True)

    print(f"\nDone — {merged['cce_score'].shape[0]} rows → {args.out}", flush=True)


if __name__ == '__main__':
    main()
