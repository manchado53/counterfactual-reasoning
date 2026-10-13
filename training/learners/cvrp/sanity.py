"""
SANITY GATE for the routing env — the pre-flight checklist's "plain DQN must beat the
problem BEFORE testing CCE" rule.

Nothing downstream (Claim 1 correlations, Claim 2 sweeps) is meaningful unless a plain
DQN can actually learn to route. This trains DQN-uniform on a small instance and checks
the greedy plan approaches the exact DP optimum.

Run:
    python -m counterfactual_rl.agents.cvrp.sanity [--instance small] [--episodes 3000]

Passes when the final greedy plan reaches >= --threshold of optimal (default 0.95).
"""

import argparse
import sys

import numpy as np

from .dqn import CVRPDQN, build_env
from .config import DEFAULT_CONFIG


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--instance', default='small')
    p.add_argument('--capacity', type=int, default=None,
                   help="load limit; omit for the instance default, -1 for TSP mode")
    p.add_argument('--episodes', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--algorithm', default='dqn-uniform')
    p.add_argument('--threshold', type=float, default=0.95)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        'instance': args.instance,
        'algorithm': args.algorithm,
        'seed': args.seed,
        'n_episodes': args.episodes,
        'eval_interval': max(50, args.episodes // 20),
        'epsilon_decay_episodes': max(1, int(args.episodes * 0.5)),
        'n_checkpoints': 0,
        'save_every': 10 ** 9,
    })
    if args.capacity is not None:
        cfg['capacity'] = None if args.capacity < 0 else args.capacity
    else:
        cfg.pop('capacity', None)   # fall back to the instance's own capacity

    env = build_env(cfg)
    mode = f"CVRP capacity={env.capacity}" if env.is_capacitated else "TSP"
    print("=" * 68)
    print(f"SANITY GATE — plain {args.algorithm} on {mode}")
    print(f"  instance '{args.instance}': {env.n_customers} customers, "
          f"{env.n_states} states, {env.min_loads()} load(s) minimum")
    print("=" * 68)

    agent = CVRPDQN(cfg)
    print(f"  exact optimal length : {agent.optimal_length:.4f}")
    print(f"  optimal plan         : {agent._optimal_tour}")

    # Reference points: a random legal policy, and greedy nearest-neighbour.
    rng = np.random.default_rng(args.seed)
    masks = np.asarray(env.action_masks)
    rand_lengths = []
    for _ in range(200):
        s, path = env.start_states[0], [0]
        for _ in range(200):
            legal = np.flatnonzero(masks[s])
            if legal.size == 0:
                break
            a = int(rng.choice(legal))
            s = int(np.asarray(env.next_states)[s, a, 0])
            path.append(int(env.state_current_np[s]))
        rand_lengths.append(env.tour_length(path))
    random_ratio = agent.optimal_length / float(np.mean(rand_lengths))
    print(f"  random-policy baseline: {random_ratio:.4f} of optimal")

    agent.learn(n_episodes=args.episodes, verbose=not args.quiet)

    final = agent.evaluate(5)
    path, _ = agent.rollout_greedy()
    achieved = env.tour_length(path)

    print("\n" + "=" * 68)
    print(f"  learned plan    : {path}")
    print(f"  learned length  : {achieved:.4f}   (optimal {agent.optimal_length:.4f})")
    print(f"  fraction optimal: {final['opt_ratio']:.4f}"
          f"   (random was {random_ratio:.4f})")
    ok = final['opt_ratio'] >= args.threshold
    print(f"  GATE: {'PASS' if ok else 'FAIL'} "
          f"(need >= {args.threshold} of optimal)")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
