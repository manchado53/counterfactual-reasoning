"""Tests for the budget-constrained (orienteering) routing env and its exact oracle."""

import numpy as np
import pytest

from counterfactual_rl.envs.routing_budget import (
    BudgetRoutingEnv, INSTANCES, optimal_closed_tour_units, quantize)
from counterfactual_rl.analysis.claim1.cvrp.budget_oracle import (
    brute_force_served, compute_oracle, optimal_plan, optimal_served, stakes)

SMALL = INSTANCES["small"]


def small_env(mult, scale=10, **kw):
    return BudgetRoutingEnv(node_xy=SMALL["xy"], demand=SMALL["demand"],
                            capacity=SMALL["capacity"], budget_mult=mult,
                            dist_scale=scale, **kw)


# ── env invariants ────────────────────────────────────────────────────────────

def test_budget_is_never_exceeded_and_home_is_always_reachable():
    env = small_env(0.8)
    D, B = env.D, env.budget
    for s, si in env._index.items():
        cur, mask, cap, spent = s
        assert spent <= B, f"state {s} already over budget"
        # The vehicle must always be able to get home from any reachable state.
        assert spent + D[cur, 0] <= B, f"state {s} is stranded"


def test_only_legal_actions_are_unmasked_and_masks_match_python_dynamics():
    env = small_env(0.9)
    for s, si in env._index.items():
        expected = set(env._legal_actions(s)) if not env._is_terminal(s) else set()
        got = set(np.flatnonzero(env.action_masks_np[si]).tolist())
        assert got == expected, f"mask mismatch at {s}"


def test_reward_is_one_per_new_customer_and_zero_for_depot_legs():
    env = small_env(0.9)
    for s, si in env._index.items():
        if env._is_terminal(s):
            continue
        for a in env._legal_actions(s):
            r = float(env.rewards_np[si, a])
            assert r == (0.0 if a == 0 else 1.0)


def test_terminal_states_are_at_the_depot():
    env = small_env(0.7)
    for s, si in env._index.items():
        if not env.action_masks_np[si].any():
            assert s[0] == 0, f"terminal state {s} is not at the depot"


def test_features_expose_remaining_fuel():
    env = small_env(0.9)
    feats = np.asarray(env.state_features)
    assert feats.shape == (env.n_states, env.feature_dim)
    fuel = feats[:, -1]
    assert np.isclose(fuel[env.start_states[0]], 1.0)
    assert fuel.min() >= -1e-6 and fuel.max() <= 1.0 + 1e-6
    # fuel must actually vary, else the state is not observable
    assert fuel.std() > 0.0


# ── the dial behaves ──────────────────────────────────────────────────────────

def test_generous_budget_serves_everyone_and_tight_budget_does_not():
    loose = small_env(1.5)
    assert optimal_served(loose) == loose.n_customers
    tight = small_env(0.6)
    assert optimal_served(tight) < tight.n_customers


def test_optimal_served_is_monotone_in_the_budget():
    served = [optimal_served(small_env(m)) for m in (0.5, 0.7, 0.9, 1.1, 1.4)]
    assert served == sorted(served), served


# ── oracle correctness ────────────────────────────────────────────────────────

@pytest.mark.parametrize("mult", [0.5, 0.6, 0.75, 0.9, 1.0, 1.2])
def test_oracle_matches_brute_force_on_the_small_instance(mult):
    env = small_env(mult)
    assert optimal_served(env) == brute_force_served(env)


def test_oracle_self_check_runs_on_the_default_instance():
    env = BudgetRoutingEnv(budget_mult=0.75, dist_scale=10)
    V, Q = compute_oracle(env, check=True)          # raises if the DP order is wrong
    assert V[env.start_states[0]] == optimal_served(env)
    assert V.max() <= env.n_customers


def test_optimal_plan_is_feasible_and_achieves_the_optimum():
    env = small_env(0.9)
    path, served = optimal_plan(env)
    assert path[0] == 0 and path[-1] == 0
    assert env.tour_units(path) <= env.budget
    assert served == optimal_served(env)
    visited = [p for p in path if p != 0]
    assert len(visited) == len(set(visited)) == served


def test_stakes_are_zero_where_the_choice_cannot_matter():
    env = small_env(0.8)
    _, Q = compute_oracle(env)
    st = stakes(env, Q)
    assert st.min() >= 0.0
    single = env.action_masks_np.sum(axis=1) <= 1
    assert np.all(st[single] == 0.0)


# ── quantization ──────────────────────────────────────────────────────────────

def test_quantized_distances_are_symmetric_positive_integers():
    D = quantize(SMALL["xy"], 10)
    assert (D == D.T).all()
    assert (np.diag(D) == 0).all()
    assert (D[~np.eye(len(D), dtype=bool)] >= 1).all()


def test_all_customer_optimum_matches_brute_force_permutations():
    from itertools import permutations
    xy, dem, cap = SMALL["xy"], SMALL["demand"], SMALL["capacity"]
    D = quantize(xy, 10)
    n_cust = len(xy) - 1

    def best_with_splits(perm):
        # optimal insertion of depot reloads into a fixed customer order
        best = None
        for bits in range(1 << (n_cust - 1)):
            tour, load, ok = [0], cap, True
            for i, c in enumerate(perm):
                if dem[c] > load or (i > 0 and (bits >> (i - 1)) & 1 and tour[-1] != 0):
                    tour.append(0); load = cap
                if dem[c] > load:
                    ok = False; break
                tour.append(c); load -= dem[c]
            if not ok:
                continue
            tour.append(0)
            cost = sum(D[tour[k], tour[k + 1]] for k in range(len(tour) - 1))
            best = cost if best is None else min(best, cost)
        return best

    brute = min(best_with_splits(p) for p in permutations(range(1, n_cust + 1)))
    assert optimal_closed_tour_units(D, np.asarray(dem), cap) == brute
