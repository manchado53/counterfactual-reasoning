"""
Exact Q* oracle for FrozenLake via value iteration on env.P.

Oracle label per non-terminal state:
    mean_{a ≠ a*} |Q*(s, a*) - Q*(s, a)|   where a* = argmax_a Q*(s, a)

High score = optimal action is substantially better than alternatives = consequential state.
"""

import numpy as np
from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv


def compute_oracle(map_name='8x8', is_slippery=True, gamma=0.99, tol=1e-10):
    """
    Returns
    -------
    Q            : np.ndarray (n_states, n_actions)
    oracle       : dict {state_idx: float}  — one label per non-terminal state
    non_terminal : list[int]                — 53 states for 8×8
    """
    env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)

    desc = [c for row in env.desc for c in row]
    terminal = {s for s, c in enumerate(desc) if c in ('H', 'G')}
    non_terminal = [s for s in range(env.n_states) if s not in terminal]

    # Value iteration
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

    # Oracle label: mean gap between optimal and suboptimal actions
    oracle = {}
    for s in non_terminal:
        a_star = int(np.argmax(Q[s]))
        oracle[s] = float(np.mean([
            abs(Q[s, a_star] - Q[s, a])
            for a in range(env.n_actions) if a != a_star
        ]))

    return Q, oracle, non_terminal
