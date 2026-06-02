"""
Exact optimal Q*(s,a) for the custom FrozenLake MDP via value iteration.

FrozenLake here is a small known MDP, so we can compute the TRUE optimal
action-values exactly (no MCTS/rollout approximation). The env exposes, for each
(state, action), three equiprobable slip outcomes as arrays:
    env.next_states (S, 4, 3) int
    env.rewards     (S, 4, 3) float   (+1 only on the transition into the goal)
    env.dones       (S, 4, 3) bool    (terminal: holes + goal self-loop)

Q*(s,a) = mean_o [ R[s,a,o] + gamma * (0 if done else V*(next[s,a,o])) ]
V*(s)   = max_a Q*(s,a)

Ground-truth "stakes" of a state = max_a Q*(s,a) - min_a Q*(s,a):
how much the action choice changes the optimal expected outcome.
"""

import numpy as np


def compute_qstar(env, gamma=0.99, tol=1e-12, max_iter=100_000):
    """Return Q* of shape (S, 4) by exact value iteration on the FrozenLake MDP."""
    ns = np.asarray(env.next_states)          # (S,4,3) int
    rw = np.asarray(env.rewards, dtype=np.float64)   # (S,4,3)
    dn = np.asarray(env.dones).astype(np.float64)    # (S,4,3) 1.0 if terminal transition
    S = ns.shape[0]

    V = np.zeros(S, dtype=np.float64)
    for _ in range(max_iter):
        # bootstrap value of each outcome, zeroed when that transition is terminal
        boot = (1.0 - dn) * V[ns]                     # (S,4,3)
        Q = (rw + gamma * boot).mean(axis=2)          # (S,4) — mean over 3 equiprob slips
        newV = Q.max(axis=1)
        if np.max(np.abs(newV - V)) < tol:
            V = newV
            break
        V = newV

    boot = (1.0 - dn) * V[ns]
    Qstar = (rw + gamma * boot).mean(axis=2)          # (S,4)
    return Qstar.astype(np.float32)


def stakes_from_qstar(Qstar):
    """Per-state ground-truth stakes = spread of optimal action-values (max - min)."""
    return (Qstar.max(axis=1) - Qstar.min(axis=1)).astype(np.float32)
