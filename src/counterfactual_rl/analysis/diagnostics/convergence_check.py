"""
MCTS ground-truth convergence check.

Question: is the "stakes" referee (MCTS per-action value spread) horizon-limited?
If MCTS-200 can't see deep cliffs (no value net, only 200 sims), then bumping the
simulation budget should make new high-stakes states appear. If stakes are stable
across budgets, MCTS-200 is converged and we can trust it.

Method: take ONE fixed sample of states (from a trained checkpoint, same generation
path as compute_diagnostics), then compute per-action MCTS value spread at several
simulation budgets on those SAME states. Compare.

Usage:
    python -m counterfactual_rl.analysis.diagnostics.convergence_check \
        --runs-root <.../runs> --run-id 259281 --chunk 100 \
        --n-states 300 --budgets 50 200 800 3200 \
        --out <.../docs/figures/diagnostics/convergence.npz>
"""

import argparse
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr

from counterfactual_rl.agents.connect_four.consequence_dqn import Connect4ConsequenceDQN
from counterfactual_rl.analysis.diagnostics.compute_diagnostics import (
    ENV_INFO, _build_mcts_values_fn, generate_transitions,
)


def stakes_at_budget(mcts_fn, flat_state, legal, batch=100):
    """Per-state value spread (max-min over legal actions) at one sim budget."""
    B = legal.shape[0]
    qvals = np.zeros((B, legal.shape[1]), dtype=np.float32)
    key = jax.random.PRNGKey(0)
    for lo in range(0, B, batch):
        hi = min(lo + batch, B)
        sub = jax.tree.map(lambda x: x[lo:hi], flat_state)
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, hi - lo)
        qvals[lo:hi] = np.asarray(jax.block_until_ready(mcts_fn(sub, keys)))
    q_legal = np.where(legal, qvals, np.nan)
    return (np.nanmax(q_legal, axis=1) - np.nanmin(q_legal, axis=1)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--n-states', type=int, default=300)
    ap.add_argument('--epsilon', type=float, default=0.05)
    ap.add_argument('--budgets', nargs='+', type=int, default=[50, 200, 800, 3200])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ckpt = os.path.join(args.runs_root, args.run_id, 'checkpoints', f'ckpt_{args.chunk:07d}.pkl')
    with open(ckpt, 'rb') as f:
        cfg = pickle.load(f)['config']
    print(f"Checkpoint: {ckpt}", flush=True)
    print(f"Config opponent={cfg.get('opponent')} mcts_n_sims(train)={cfg.get('mcts_n_sims')}", flush=True)

    agent = Connect4ConsequenceDQN(ENV_INFO, config=cfg)
    agent._build_batched_rollout_fn()
    with open(ckpt, 'rb') as f:
        ck = pickle.load(f)
    agent.params = jax.tree.map(jnp.array, ck['params'])
    agent.target_params = jax.tree.map(jnp.array, ck['target_params'])

    # ONE fixed sample of states, reused across all budgets.
    _batch, flat_state = generate_transitions(agent, args.n_states, args.epsilon)
    legal = np.asarray(flat_state.legal_action_mask)
    B = legal.shape[0]
    print(f"Sampled {B} fixed states.\n", flush=True)

    results = {}
    for nb in args.budgets:
        t0 = time.time()
        mcts_fn = _build_mcts_values_fn(nb)
        s = stakes_at_budget(mcts_fn, flat_state, legal)
        results[nb] = s
        print(f"  n_sims={nb:5d}: stakes mean={s.mean():.3f}  "
              f"%>0.05={100*np.mean(s>0.05):4.1f}  %>0.5={100*np.mean(s>0.5):4.1f}  "
              f"%>1.0={100*np.mean(s>1.0):4.1f}  ({time.time()-t0:.0f}s)", flush=True)

    budgets = sorted(results)
    lo_b, hi_b = budgets[0], budgets[-1]
    lo, hi = results[lo_b], results[hi_b]

    print(f"\n=== Convergence: {lo_b} -> {hi_b} sims (same states) ===", flush=True)
    print(f"rank stability spearman(stakes_{lo_b}, stakes_{hi_b}) = "
          f"{spearmanr(lo, hi).correlation:+.3f}", flush=True)
    # Cliffs invisible at low budget that appear at high budget:
    invisible_lo = lo < 0.05
    appeared = invisible_lo & (hi > 0.5)
    print(f"states with stakes<0.05 at {lo_b} sims:        {invisible_lo.sum():4d} "
          f"({100*invisible_lo.mean():.1f}%)", flush=True)
    print(f"  ...of those, became >0.5 at {hi_b} sims:     {appeared.sum():4d} "
          f"({100*appeared.sum()/max(1,invisible_lo.sum()):.1f}% of the invisibles)", flush=True)
    # And the reverse: stable zeros (truly boring at both)
    stable_zero = invisible_lo & (hi < 0.05)
    print(f"  ...stayed <0.05 at {hi_b} sims (truly boring):{stable_zero.sum():4d} "
          f"({100*stable_zero.sum()/max(1,invisible_lo.sum()):.1f}% of the invisibles)", flush=True)

    print("\nmean stakes by budget:", flush=True)
    for nb in budgets:
        print(f"  {nb:5d}: {results[nb].mean():.3f}", flush=True)
    growth = results[hi_b].mean() / max(1e-9, results[lo_b].mean())
    print(f"\nmean-stakes growth {lo_b}->{hi_b}: {growth:.2f}x", flush=True)
    print("VERDICT:", "MCTS-200 is HORIZON-LIMITED (stakes grow with budget) -> referee undercounts cliffs"
          if results[200].mean() < 0.9 * results[hi_b].mean()
          else "MCTS-200 looks CONVERGED (stakes stable) -> referee is trustworthy", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, legal=legal,
                        **{f'stakes_{nb}': results[nb] for nb in budgets},
                        budgets=np.array(budgets))
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
