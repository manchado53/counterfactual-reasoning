"""
Gate 1 -- greedy-decode a checkpoint on RP-DQN's own held-out test set.

    python -m counterfactual_rl.agents.cvrp_neural.eval_testset \
        --checkpoint <run>/best.pkl [--testset data/rpdqn/vrp20_test_seed1234.pkl]

Why this needs no solver. RP-DQN evaluate on the exact 10,000-instance test set
from Nazari et al., and Table 1 / Table 3 report the MEAN TOUR LENGTH on it:

    n=20    Gurobi (optimal) 6.10    LKH3 6.14    RP-DQN greedy 6.36  (gap 4.26%)
    n=50    LKH3 10.38                            RP-DQN greedy 10.92 (gap 5.20%)

So running the same instances and comparing our mean directly to 6.36 is a
complete check -- no LKH3, no Gurobi, no OR-Tools installation.

The file ships in their repo at data/CVRP/. Each record is
``(depot[2], loc[n,2], demand[n] ints 1..9, capacity)``, which is exactly the
distribution ``envs/cvrp_neural.py`` samples, so the env needs no special casing --
only the instance injected into a state pytree instead of drawn from a key.
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from counterfactual_rl.envs.cvrp_neural import CVRPNeuralEnv, CVRPState

# Published reference points, from the paper's Table 1 (n -> (optimal, their greedy)).
REFERENCE = {
    20:  (6.10, 6.36),
    50:  (10.38, 10.92),
    100: (15.65, 16.59),
}


def load_testset(path):
    """-> (coords (M,n+1,2), demands_raw (M,n), capacity float)"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    depots = np.array([np.asarray(d[0], dtype=np.float64) for d in data])       # (M,2)
    locs = np.array([np.asarray(d[1], dtype=np.float64) for d in data])         # (M,n,2)
    demands = np.array([np.asarray(d[2], dtype=np.float64) for d in data])      # (M,n)
    caps = np.array([float(d[3]) for d in data])
    assert np.all(caps == caps[0]), "test set mixes capacities"
    coords = np.concatenate([depots[:, None, :], locs], axis=1)                 # (M,n+1,2)
    return coords, demands, float(caps[0])


def states_from_testset(coords, demands_raw, capacity):
    """Pack instances into a batched CVRPState, matching env.reset's conventions."""
    M, n_nodes, _ = coords.shape
    demands = np.zeros((M, n_nodes), dtype=np.float32)
    demands[:, 1:] = demands_raw / capacity      # env stores demand normalised by capacity
    return CVRPState(
        coords=jnp.asarray(coords, dtype=jnp.float32),
        demands=jnp.asarray(demands),
        visited=jnp.zeros((M, n_nodes), dtype=bool),
        current=jnp.zeros((M,), dtype=jnp.int32),
        load=jnp.ones((M,), dtype=jnp.float32),
        step=jnp.zeros((M,), dtype=jnp.int32),
        done=jnp.zeros((M,), dtype=bool),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--testset', default='data/rpdqn/vrp20_test_seed1234.pkl')
    ap.add_argument('--chunk', type=int, default=1000,
                    help='instances per batch (T4 memory)')
    ap.add_argument('--limit', type=int, default=None,
                    help='evaluate only the first N instances (smoke checks)')
    args = ap.parse_args()

    coords, demands_raw, capacity = load_testset(args.testset)
    if args.limit:
        coords, demands_raw = coords[:args.limit], demands_raw[:args.limit]
    M, n_nodes, _ = coords.shape
    n_customers = n_nodes - 1
    print(f"test set: {M} instances, n={n_customers}, capacity={capacity}")

    from .consequence_dqn_vectorized import CVRPNeuralConsequenceDQNVectorized
    from .dqn_vectorized import CVRPNeuralDQNVectorized

    with open(args.checkpoint, 'rb') as f:
        cfg = pickle.load(f)['config']
    Agent = (CVRPNeuralConsequenceDQNVectorized
             if cfg.get('algorithm') == 'consequence-dqn' else CVRPNeuralDQNVectorized)
    agent = Agent.from_checkpoint(args.checkpoint)

    if agent.env.n_customers != n_customers:
        raise SystemExit(
            f"checkpoint is n={agent.env.n_customers} but the test set is n={n_customers}"
        )
    if abs(agent.env.capacity - capacity) > 1e-6:
        raise SystemExit(
            f"capacity mismatch: env {agent.env.capacity} vs test set {capacity}"
        )

    tours, solved = [], []
    for lo in range(0, M, args.chunk):
        hi = min(lo + args.chunk, M)
        states = states_from_testset(coords[lo:hi], demands_raw[lo:hi], capacity)
        m = agent.evaluate(init_states=states)
        # evaluate() returns means; recover the sum so chunks of unequal size combine.
        tours.append(m['mean_tour'] * (hi - lo))
        solved.append(m['solved_frac'] * (hi - lo))
        print(f"  {hi}/{M}  running mean tour {sum(tours)/hi:.4f}", flush=True)

    mean_tour = sum(tours) / M
    solved_frac = sum(solved) / M

    print("\n" + "=" * 58)
    print(f"  instances          {M}")
    print(f"  feasible solutions {solved_frac:.4%}")
    print(f"  MEAN TOUR LENGTH   {mean_tour:.4f}")
    if n_customers in REFERENCE:
        opt, theirs = REFERENCE[n_customers]
        print(f"  gap to optimal     {100 * (mean_tour - opt) / opt:.2f}%   "
              f"(RP-DQN greedy: {100 * (theirs - opt) / opt:.2f}%)")
        print(f"  vs RP-DQN greedy   {mean_tour:.4f} vs {theirs:.4f} "
              f"({'BETTER' if mean_tour < theirs else 'worse'})")
    print("=" * 58)
    if solved_frac < 0.999:
        print("WARNING: some instances did not finish -- the mean tour length is not "
              "comparable to their number until every instance is feasible.")


if __name__ == '__main__':
    main()
