"""
Correctness tests for the exact routing oracle (analysis/claim1/cvrp/oracle.py).

The critical tests are the brute-force ones: on small instances we compute the true
optimum by exhaustive enumeration and check the DP matches exactly. If those fail, the
Claim-1 ground truth is wrong and nothing downstream means anything.

    TSP  brute force: every permutation of customers.
    CVRP brute force: every permutation x optimal split into loads (exact for CVRP —
                      any solution's routes, concatenated, form some permutation, and
                      the split DP finds the best partition of that permutation).

Runnable two ways:
    pytest tests/test_cvrp_oracle.py
    python tests/test_cvrp_oracle.py     # self-contained runner, no pytest needed
"""

from itertools import permutations

import numpy as np

from counterfactual_rl.envs.cvrp import CVRPEnv, DEPOT
from counterfactual_rl.analysis.claim1.cvrp.oracle import compute_oracle, optimal_tour

# depot + 5 customers -> 120 permutations, brute-forceable
SMALL_XY = np.array(
    [[0.50, 0.50], [0.10, 0.20], [0.85, 0.15], [0.90, 0.70], [0.40, 0.95], [0.05, 0.65]],
    dtype=np.float32,
)
CAP_DEMAND = np.array([0, 3, 3, 3, 3, 3], dtype=np.int32)   # 15 total
CAP_CAPACITY = 6                                             # -> 3 loads needed


def _tsp():
    return CVRPEnv(node_xy=SMALL_XY, capacity=None)


def _cvrp():
    return CVRPEnv(node_xy=SMALL_XY, demand=CAP_DEMAND, capacity=CAP_CAPACITY)


def _brute_tsp(env):
    best = float("inf")
    for perm in permutations(range(1, env.n_nodes)):
        best = min(best, env.tour_length([DEPOT] + list(perm) + [DEPOT]))
    return best


def _brute_cvrp(env):
    """Exhaustive: min over customer orderings of the optimal split into feasible loads."""
    d, dist, cap = env.demand, env.dist, env.capacity
    best = float("inf")
    for perm in permutations(range(1, env.n_nodes)):
        n = len(perm)
        # cost[i] = cheapest way to serve the first i customers of this ordering
        cost = [0.0] + [float("inf")] * n
        for i in range(n):
            if not np.isfinite(cost[i]):
                continue
            load, leg = 0, 0.0
            for j in range(i, n):
                load += int(d[perm[j]])
                if load > cap:
                    break
                leg = leg if j == i else leg + float(dist[perm[j - 1], perm[j]])
                route = float(dist[DEPOT, perm[i]]) + leg + float(dist[perm[j], DEPOT])
                cost[j + 1] = min(cost[j + 1], cost[i] + route)
        best = min(best, cost[n])
    return best


def test_tsp_dp_matches_brute_force():
    env = _tsp()
    brute = _brute_tsp(env)
    Q, _, _ = compute_oracle(env, gamma=1.0, min_legal=1)
    dp = float(np.nanmax(Q[env.start_states[0]]))
    assert np.isclose(-dp, brute, atol=1e-5), (-dp, brute)
    tour, length = optimal_tour(env, gamma=1.0)
    assert np.isclose(length, brute, atol=1e-5)
    assert sorted(tour[1:-1]) == list(range(1, env.n_nodes))


def test_cvrp_dp_matches_brute_force():
    """THE correctness check for the capacitated problem."""
    env = _cvrp()
    brute = _brute_cvrp(env)
    Q, _, _ = compute_oracle(env, gamma=1.0, min_legal=1)
    dp = float(np.nanmax(Q[env.start_states[0]]))
    assert np.isclose(-dp, brute, atol=1e-5), (-dp, brute)

    tour, length = optimal_tour(env, gamma=1.0)
    assert np.isclose(length, brute, atol=1e-5), (length, brute)
    # the optimal plan serves everyone and reloads at the depot
    assert tour[0] == DEPOT and tour[-1] == DEPOT
    assert sorted(p for p in tour if p != DEPOT) == list(range(1, env.n_nodes))
    assert tour.count(DEPOT) >= env.min_loads() + 1


def test_bellman_residual_selfcheck_runs():
    """compute_oracle raises if its state ordering is not topological — exercise both modes."""
    for env in (_tsp(), _cvrp()):
        compute_oracle(env, gamma=0.99)  # must not raise


def test_illegal_actions_are_nan():
    for env in (_tsp(), _cvrp()):
        Q, _, _ = compute_oracle(env, gamma=0.99)
        masks = np.asarray(env.action_masks)
        is_terminal = ~masks.any(axis=1)
        for s in range(env.n_states):
            if is_terminal[s]:
                continue
            assert np.isfinite(Q[s, masks[s]]).all()
            assert np.isnan(Q[s, ~masks[s]]).all()


def test_oracle_labels_are_nonnegative_and_finite():
    for env in (_tsp(), _cvrp()):
        _, oracle, decision = compute_oracle(env, gamma=0.99)
        assert len(oracle) == len(decision) > 0
        vals = np.array(list(oracle.values()))
        assert np.isfinite(vals).all()
        assert (vals >= -1e-9).all()


def test_forced_states_excluded():
    env = _cvrp()
    _, oracle, decision = compute_oracle(env, gamma=0.99, min_legal=2)
    n_legal = np.asarray(env.action_masks).sum(axis=1)
    for s in decision:
        assert n_legal[s] >= 2
    for s in np.where(n_legal == 1)[0]:
        assert int(s) not in oracle


def test_default_cvrp_instance_has_spread():
    env = CVRPEnv()
    _, oracle, decision = compute_oracle(env, gamma=0.99)
    vals = np.array(list(oracle.values()))
    assert len(decision) > 100
    assert np.isfinite(vals).all()
    assert vals.max() - vals.min() > 1e-3


TESTS = [
    test_tsp_dp_matches_brute_force,
    test_cvrp_dp_matches_brute_force,
    test_bellman_residual_selfcheck_runs,
    test_illegal_actions_are_nan,
    test_oracle_labels_are_nonnegative_and_finite,
    test_forced_states_excluded,
    test_default_cvrp_instance_has_spread,
]


if __name__ == "__main__":
    import traceback

    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()

    env = CVRPEnv()
    tour, length = optimal_tour(env, gamma=1.0)
    _, oracle, decision = compute_oracle(env, gamma=0.99)
    vals = np.array(list(oracle.values()))
    print(f"\ndefault CVRP instance: {env.n_states} states, {len(decision)} decision states")
    print(f"optimal plan   : {tour}")
    print(f"optimal length : {length:.4f}  ({tour.count(DEPOT) - 1} loads)")
    print(f"oracle stakes  : min={vals.min():.4f} median={np.median(vals):.4f} max={vals.max():.4f}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
