"""
Gate 2 -- is the CCE score worth anything on THEIR environment?

    python -m counterfactual_rl.agents.cvrp_neural.gate2_signal \
        --checkpoint <run>/checkpoints/ckpt_0125000.pkl [--n-states 512]

The plan's question, and the reviewer's first one: if a trivial one-step heuristic
already reproduces the score, the rollout is decoration. On the old routing env a
cf_horizon=1 baseline -- which runs NO rollout at all, because the scan length is
horizon-1 -- reached rho ~0.53 and was FLAT across training (+0.035), while the real
rollout rose +0.204. The claim that survives is the rise, not the level.

There is no enumerable exact oracle here: instances are drawn fresh and CVRP is
NP-hard, so "the optimal completion cost from each action" is not available. The
reference used instead is the FULL-HORIZON score -- a rollout run to termination,
which under a deterministic env and a greedy continuation is the exact return of the
current policy, with no truncation and no bootstrap. Every truncated horizon is then
scored by how well it recovers that. This answers both live questions at once:

    horizon=1  vs full   -- does the no-rollout baseline already get the signal?
    horizon=10 vs full   -- is the cheap horizon we plan to ship good enough?

Reported per horizon: Spearman rho against the full-horizon score, plus the health
numbers the plan asks for -- distinct values, fraction exactly zero, gini, and the
realized ess_frac the sampler would actually see.
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def gini(x):
    """0 = every transition equally prioritised, 1 = one transition takes everything."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-states', type=int, default=512)
    ap.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 5, 10, 25])
    ap.add_argument('--chunks', type=int, default=4,
                    help='collection chunks used to fill the buffer before scoring')
    args = ap.parse_args()

    from .consequence_dqn_vectorized import CVRPNeuralConsequenceDQNVectorized

    with open(args.checkpoint, 'rb') as f:
        cfg = dict(pickle.load(f)['config'])
    # Score with CCE machinery regardless of which arm produced the checkpoint --
    # a PER checkpoint is in fact the cleaner control, since the policy being scored
    # was not itself shaped by the score.
    cfg.update(algorithm='consequence-dqn', cf_rollout_temperature=0.0,
               cf_n_rollouts=1, consequence_metric='wasserstein')
    agent = CVRPNeuralConsequenceDQNVectorized.from_checkpoint(args.checkpoint, cfg)
    agent._current_episode = 0

    # Fill the buffer with on-policy transitions from this checkpoint.
    agent._build_collect_fn()
    n_envs = agent.config['n_envs']
    key = jax.random.PRNGKey(12345)
    key, ik = jax.random.split(key)
    states = jax.vmap(lambda k: agent.env.reset(k)[1])(jax.random.split(ik, n_envs))
    for _ in range(args.chunks):
        key, ck = jax.random.split(key)
        sk = jax.random.split(ck, n_envs * agent.config['collect_steps']).reshape(
            n_envs, agent.config['collect_steps'], 2)
        states, outputs = agent._collect_fn(agent.params, jnp.float32(0.02), states, sk)
        agent._add_chunk_to_buffer(outputs)
    print(f"buffer: {len(agent.buffer)} transitions\n")

    # Score the SAME states at every horizon, or the horizons are not comparable.
    #
    # Two RNGs decide that and BOTH must be pinned. agent._key drives the rollout
    # keys, but ConsequenceReplayBuffer.sample_uniform draws its sample with
    # np.random.choice -- pinning only the JAX key leaves every horizon scoring a
    # DIFFERENT random subset. Comparing those then pairs "scored" against
    # "not scored" and manufactures a large spurious correlation that is identical
    # at every horizon. (Observed: a flat rho = -0.79 from horizon 1 through 25.)
    SAMPLE_SEED = 7
    agent.n_score_sample = min(args.n_states, len(agent.buffer))
    rng_state = agent._key

    # Reproduce exactly the indices sample_uniform will draw, so scores can be read
    # back positionally and a legitimately-zero score is not mistaken for "unscored".
    np.random.seed(SAMPLE_SEED)
    idx = np.random.choice(len(agent.buffer), size=agent.n_score_sample, replace=False)

    full_h = agent.env.max_steps  # to termination: no truncation, no bootstrap

    def score_at(h, bootstrap):
        agent._key = rng_state
        np.random.seed(SAMPLE_SEED)            # same states every horizon
        agent.cf_horizon = h
        agent.cf_bootstrap = bootstrap
        agent._compiled_rollout_fn = None
        agent.buffer.consequence_scores[:] = 0.0
        agent._score_buffer_transitions()
        return np.array(agent.buffer.consequence_scores)[idx]

    # Measure each truncated horizon BOTH ways. bootstrap=False is the raw
    # truncation bias; bootstrap=True is what the sweep actually ships, and it
    # exists precisely to repair short horizons (issue #7). Reporting only the
    # former would judge a setting we do not use -- which is how the h=3 dip
    # below h=1 first showed up.
    results, results_bs = {}, {}
    for h in list(args.horizons) + [full_h]:
        s = score_at(h, False)
        results[h] = s
        diag = agent.buffer.priority_diagnostics()
        if h != full_h:
            results_bs[h] = score_at(h, True)
        print(f"horizon {h:>3}{'  (FULL, reference)' if h == full_h else '':<20}"
              f" distinct {len(np.unique(s)):>5}"
              f"  zero {float(np.mean(s == 0.0)):.3f}"
              f"  mean {s.mean():.5f}  std {s.std():.5f}"
              f"  gini {gini(s):.3f}"
              f"  ess_frac {diag.get('ess_frac', float('nan')):.3f}")

    from scipy.stats import spearmanr
    ref = results[full_h]
    print("\nSpearman rho against the full-horizon (exact-return) score:")
    print(f"  {'horizon':>9} {'bootstrap OFF':>15} {'bootstrap ON':>14}   {'delta':>7}")
    for h in args.horizons:
        off = spearmanr(results[h], ref).correlation
        on = spearmanr(results_bs[h], ref).correlation
        tag = '   <- no rollout at all' if h == 1 else ''
        print(f"  {h:>9} {off:>15.4f} {on:>14.4f}   {on-off:>+7.4f}{tag}")
    # Sanity: the reference against itself must be exactly 1. If this is not 1.0000
    # the two passes did not score the same states and every rho above is garbage.
    print(f"  {'full':>10}   rho {spearmanr(ref, ref).correlation: .4f}"
          f"   <- self-check, must be 1.0000")

    print("\nRead this as: whatever horizon=1 scores is the floor a reviewer will")
    print("point at. The number that matters for the paper is how far horizon=10")
    print("sits above it, and whether that gap GROWS across training checkpoints.")


if __name__ == '__main__':
    main()
