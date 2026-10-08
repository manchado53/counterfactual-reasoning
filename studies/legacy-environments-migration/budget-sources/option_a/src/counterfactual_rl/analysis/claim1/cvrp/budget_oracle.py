"""
Exact oracle for the budget-constrained (orienteering) routing env.

Ground truth = the MAXIMUM NUMBER OF CUSTOMERS still servable from a state, on a closed
tour that stays within the travel budget:

    V*(s) = max customers reachable from s        (integer, 0..N)
    Q*(s, a) = r(s, a) + V*(s')                   (illegal actions -> -inf)
    stakes(s) = max_a Q*(s, a) - min_a Q*(s, a)   (Claim-1 ground-truth importance)

`stakes` is the analogue of FrozenLake's Q*-spread: how much the ACTION CHOICE changes
the outcome. It is exactly what CCE tries to estimate from rollouts, so Spearman(CCE,
stakes) is the Claim-1 measurement.

WHY THIS IS EXACT AND CHEAP
---------------------------
Every leg costs at least one integer unit, so `budget_spent` strictly increases on every
transition: the reachable state graph is a DAG, topologically ordered by spend. One
backward pass over spend levels solves it — no iteration, no discounting, no
approximation. (Contrast the plain CVRP oracle, where customer -> depot reload edges
leave the mask unchanged and a popcount-only order is NOT topological. That subtlety
produced a real bug there; here spend removes the hazard entirely.)

A Bellman-residual self-check runs on every call and raises if the order was wrong.
"""

from typing import Optional, Tuple

import numpy as np

from counterfactual_rl.envs.routing_budget import DEPOT

NEG_INF = -np.inf


def compute_oracle(env, check: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact V* and Q* by backward induction over budget-spend levels.

    Returns
        V : (n_states,)            max customers still servable
        Q : (n_states, n_actions)  per-action optimum; illegal actions are -inf
    """
    n, n_a = env.n_states, env.n_actions
    nxt = env.next_states_np                     # (n, n_a)
    rew = env.rewards_np.astype(np.float64)      # (n, n_a)
    legal = env.action_masks_np                  # (n, n_a) bool
    spent = env.state_spent

    V = np.zeros(n, dtype=np.float64)
    Q = np.full((n, n_a), NEG_INF, dtype=np.float64)

    # Descending spend: every successor has strictly greater spend, so it is already done.
    for lvl in np.unique(spent)[::-1]:
        idx = np.flatnonzero(spent == lvl)
        q = rew[idx] + V[nxt[idx]]
        q = np.where(legal[idx], q, NEG_INF)
        Q[idx] = q
        has_action = legal[idx].any(axis=1)
        # Terminal states (no legal action) are worth 0 — nothing left to serve.
        V[idx] = np.where(has_action, np.where(has_action, q.max(axis=1), 0.0), 0.0)

    if check:
        residual = _bellman_residual(env, V, Q)
        if residual > 1e-9:
            raise RuntimeError(
                f"budget oracle failed its Bellman self-check (residual {residual:.3e}); "
                "the spend-level ordering is not topological for this env")
    return V, Q


def _bellman_residual(env, V: np.ndarray, Q: np.ndarray) -> float:
    """max |V(s) - max_a Q(s,a)| over non-terminal states, and Q vs r + V(s') everywhere."""
    legal = env.action_masks_np
    has_action = legal.any(axis=1)
    q_best = np.where(has_action, np.where(legal, Q, NEG_INF).max(axis=1), 0.0)
    r1 = np.abs(V - q_best).max()

    recomputed = env.rewards_np.astype(np.float64) + V[env.next_states_np]
    diff = np.where(legal, np.abs(Q - recomputed), 0.0)
    r2 = diff.max() if diff.size else 0.0
    return float(max(r1, r2))


def stakes(env, Q: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ground-truth per-state importance: spread of Q* over the LEGAL actions.

    0 means the choice does not matter at all (every legal action leads to the same
    number of customers served); larger means the action choice decides the outcome.
    Terminal / single-action states get 0.
    """
    if Q is None:
        _, Q = compute_oracle(env)
    legal = env.action_masks_np
    hi = np.where(legal, Q, NEG_INF).max(axis=1)
    lo = np.where(legal, Q, np.inf).min(axis=1)
    out = np.where(legal.sum(axis=1) >= 2, hi - lo, 0.0)
    return np.asarray(out, dtype=np.float64)


def optimal_served(env) -> int:
    """Max customers servable from the start state within the budget."""
    V, _ = compute_oracle(env)
    return int(round(V[env.start_states[0]]))


def optimal_plan(env) -> Tuple[list, int]:
    """A greedy-on-Q* walk from the start: (node path, customers served)."""
    V, Q = compute_oracle(env)
    s = env.start_states[0]
    path = [int(env.state_current_np[s])]
    served = 0
    while env.action_masks_np[s].any():
        a = int(np.argmax(np.where(env.action_masks_np[s], Q[s], NEG_INF)))
        served += int(round(float(env.rewards_np[s, a])))
        s = int(env.next_states_np[s, a])
        path.append(int(env.state_current_np[s]))
    return path, served


def brute_force_served(env) -> float:
    """
    Reference implementation: depth-first search over every feasible run.

    Returns the best achievable RETURN, which is what V*(start) means — so this stays a valid
    check under every reward shape, not just the original +1-per-customer one.

    Two details that matter for correctness:
      * the score is recorded only at TERMINAL states, never mid-path. The DP has no option to
        stop early, so a comparison that credits a prefix would not be comparing like with like
        (it matters as soon as a reward can be negative, e.g. the stranding penalty).
      * a stranded run terminates with whatever it has banked, which under 'terminal' shaping
        is zero — that is the cliff, and the brute force has to see it the same way.

    Exponential; small instances only.
    """
    best = -float('inf')
    start = (DEPOT, 0, env._initial_cap(), 0)

    def rec(s, total):
        nonlocal best
        legal = env._legal_actions(s)
        if not legal:                       # terminal: success at the depot, or stranded
            best = max(best, total)
            return
        for a in legal:
            ns, r, _ = env._next(s, a)
            rec(ns, total + float(r))

    rec(start, 0.0)
    return best
