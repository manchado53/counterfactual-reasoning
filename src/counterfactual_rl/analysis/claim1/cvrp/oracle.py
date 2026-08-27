"""
Exact Q* oracle for the routing env (TSP mode) via backward induction on env.P.

This is the Claim-1 ground truth: computed by dynamic programming on the known
transition table, with no learner and no CCE rollouts in the loop.

Why not value iteration (as in the FrozenLake oracle)?
    The routing state graph is a DAG — the visited mask only ever grows — so exact
    backward induction in one sweep (states ordered by decreasing popcount) gives the
    exact optimum with no iteration or tolerance. This is Held-Karp: O(2^N · N^2).

Oracle label per DECISION state (mirrors the paper's Eq. 7, restricted to legal actions):
    Oracle(s) = mean_{a in legal(s), a != a*} [ Q*(s, a*) - Q*(s, a) ],  a* = argmax_a Q*(s,a)

Two routing-specific departures from the FrozenLake oracle, both necessary:
  1. The mean is over LEGAL actions only. Most actions in a routing state are illegal
     (already-served stops); they carry a large fixed penalty, and averaging over them
     would swamp the real decision gap with a constant.
  2. Only states with >= 2 legal actions are labelled. With one legal action there is no
     decision to make (the tour is forced), so a "gap" is undefined — those states are
     excluded from the Claim-1 correlation rather than scored as zero.

High score = choosing wrong here costs a lot = a genuinely consequential decision.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from counterfactual_rl.envs.cvrp import CVRPEnv


def compute_oracle(
    env: Optional[CVRPEnv] = None,
    gamma: float = 0.99,
    min_legal: int = 2,
) -> Tuple[np.ndarray, Dict[int, float], List[int]]:
    """
    Exact Q* and per-state oracle importance for the routing env.

    Parameters
    ----------
    env       : CVRPEnv (default: the standard fixed instance, TSP mode)
    gamma     : discount. Use the SAME value as training (default 0.99) so the oracle
                grades what the learner is actually optimizing. gamma=1.0 makes V(start)
                exactly the negative optimal tour length (used by the tests).
    min_legal : minimum number of legal actions for a state to count as a decision state.

    Returns
    -------
    Q               : np.ndarray (n_states, n_actions) — exact Q*, NaN on illegal actions
    oracle          : dict {state_idx: float} — one label per decision state
    decision_states : list[int] — states with >= min_legal legal actions
    """
    if env is None:
        env = CVRPEnv()

    n_s, n_a = env.n_states, env.n_actions
    masks = np.asarray(env.action_masks)              # (n_states, n_actions) bool
    next_s = np.asarray(env.next_states)[:, :, 0]     # (n_states, n_actions) int
    rew = np.asarray(env.rewards)[:, :, 0]            # (n_states, n_actions) float
    dones = np.asarray(env.dones)[:, :, 0]            # (n_states, n_actions) bool

    # Terminal (absorbing) states have no legal actions by construction.
    is_terminal = ~masks.any(axis=1)

    # Topological order for backward induction.
    #
    # The visited mask never shrinks, so popcount DESCENDING is the primary key. But in
    # CVRP a reload edge (customer -> depot) keeps the mask unchanged, so within one
    # popcount level the depot state must be evaluated BEFORE the customer states that
    # can reload into it. Secondary key `current != depot` enforces that. (In TSP the
    # only same-popcount edge lands on the terminal state, which is pinned at 0, so this
    # secondary key is harmless there.) The Bellman-residual check below verifies the
    # resulting order really was topological.
    popcount = np.array([bin(int(m)).count("1") for m in env.state_mask], dtype=np.int32)
    cur_node = np.asarray(env.state_current_np)
    order = np.lexsort((cur_node != 0, -popcount))

    V = np.zeros(n_s, dtype=np.float64)
    for s in order:
        if is_terminal[s]:
            V[s] = 0.0
            continue
        legal = np.where(masks[s])[0]
        # Deterministic env: one outcome per (s, a). Successor value is 0 if it terminates.
        vals = rew[s, legal] + gamma * np.where(dones[s, legal], 0.0, V[next_s[s, legal]])
        V[s] = vals.max()

    # Self-check: a correct backward induction leaves zero Bellman residual everywhere.
    # This is what catches a wrong topological order (which would otherwise silently
    # produce a plausible-but-incorrect ground truth).
    resid = 0.0
    for s in range(n_s):
        if is_terminal[s]:
            continue
        legal = np.where(masks[s])[0]
        vals = rew[s, legal] + gamma * np.where(dones[s, legal], 0.0, V[next_s[s, legal]])
        resid = max(resid, abs(vals.max() - V[s]))
    if resid > 1e-9:
        raise RuntimeError(
            f"backward induction did not converge (max Bellman residual {resid:.3e}); "
            "the state ordering is not topological for this env."
        )

    # Q*(s, a) — NaN on illegal actions so downstream code can never average them in.
    Q = np.full((n_s, n_a), np.nan, dtype=np.float64)
    for s in range(n_s):
        if is_terminal[s]:
            continue
        legal = np.where(masks[s])[0]
        Q[s, legal] = rew[s, legal] + gamma * np.where(
            dones[s, legal], 0.0, V[next_s[s, legal]]
        )

    # Oracle label: mean suboptimality gap across the OTHER legal actions.
    oracle: Dict[int, float] = {}
    decision_states: List[int] = []
    for s in range(n_s):
        if is_terminal[s]:
            continue
        legal = np.where(masks[s])[0]
        if legal.size < min_legal:
            continue  # forced move — no decision to grade
        decision_states.append(int(s))
        q = Q[s, legal]
        a_star = int(np.argmax(q))
        gaps = q[a_star] - np.delete(q, a_star)
        # gaps is empty only when min_legal=1 and this state has a single forced move;
        # with no alternative there is no consequence to the choice -> 0.0 (not NaN).
        oracle[int(s)] = float(np.mean(gaps)) if gaps.size else 0.0

    return Q, oracle, decision_states


def optimal_tour(env: Optional[CVRPEnv] = None, gamma: float = 1.0) -> Tuple[List[int], float]:
    """
    Follow argmax Q* from the start to recover the optimal tour.

    Returns (node sequence starting and ending at the depot, its euclidean length).
    With gamma=1.0 this is the true shortest tour for the instance.
    """
    if env is None:
        env = CVRPEnv()
    Q, _, _ = compute_oracle(env, gamma=gamma, min_legal=1)
    masks = np.asarray(env.action_masks)
    next_s = np.asarray(env.next_states)[:, :, 0]

    s = env.start_states[0]
    tour = [int(env.state_current[s])]
    while masks[s].any():
        legal = np.where(masks[s])[0]
        a = int(legal[int(np.argmax(Q[s, legal]))])
        s = int(next_s[s, a])
        tour.append(int(env.state_current[s]))
    return tour, env.tour_length(tour)
