"""
Exact Q* oracle for DoorKey via value iteration on env.P.

Mirrors the FrozenLake oracle (analysis/claim1/frozen_lake/oracle.py). DoorKey is
deterministic, so each P[s][a] has a single outcome and value iteration is a plain
max-backup over the enumerated (cell, dir, has_key, door) state set.

Oracle label per non-terminal state:
    mean_{a != a*} |Q*(s, a*) - Q*(s, a)|   where a* = argmax_a Q*(s, a)

High score = the optimal action is substantially better than the alternatives = a
consequential (pivotal) state. In DoorKey the pivotal states are the gating decisions:
picking up the key, and toggling the door open — exactly what CCE should surface.

Note: reward is +1 on reaching the goal and gamma < 1, so V*(s) = gamma^(optimal steps
to goal). The action-gap therefore measures how many extra discounted steps a wrong
action costs — including the (large) cost of an action that fails to progress through
a gate.
"""

import numpy as np

from counterfactual_rl.envs.doorkey import DoorKeyEnv


def compute_oracle(layout_name='6x6', slip_prob=0.2, gamma=0.99, tol=1e-12):
    """
    Exact oracle for the DoorKey MDP at the given slip level.

    For Claim 1 use the SAME slip_prob the agent is trained/scored on (default 0.2),
    exactly as FrozenLake's Claim-1 oracle is computed on the slippery map — CCE's
    total-variation signal is only non-degenerate under stochastic dynamics.

    Returns
    -------
    Q            : np.ndarray (n_states, n_actions)
    oracle       : dict {state_idx: float}  — one label per non-terminal state
    non_terminal : list[int]                — reachable non-goal states
    env          : DoorKeyEnv               — for geometry / downstream scoring
    """
    env = DoorKeyEnv(layout_name=layout_name, slip_prob=slip_prob)

    non_terminal = list(env.non_terminal)
    terminal = set(range(env.n_states)) - set(non_terminal)

    # Value iteration over env.P (deterministic single-outcome transitions).
    V = np.zeros(env.n_states)
    while True:
        delta = 0.0
        for s in range(env.n_states):
            if s in terminal:
                continue
            v = max(
                sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                for a in range(env.n_actions)
            )
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break

    # Q*(s, a)
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Q[s, a] = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])

    # Oracle label: mean gap between the optimal and each suboptimal action.
    oracle = {}
    for s in non_terminal:
        a_star = int(np.argmax(Q[s]))
        oracle[s] = float(np.mean([
            abs(Q[s, a_star] - Q[s, a])
            for a in range(env.n_actions) if a != a_star
        ]))

    return Q, oracle, non_terminal, env
