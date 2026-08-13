"""Env adapter for the suitability pipeline.

Thin seam so FrozenLake works now and Connect Four / SMAX can slot in later (they just
return qstar_spread=None → GAIN-fidelity becomes n/a). For FrozenLake we compute the EXACT
ground-truth stakes by probability-weighted value iteration (handles legacy is_slippery AND
the new slip_probability), generalizing analysis/diagnostics/value_iteration.py:compute_qstar.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SuitabilityAdapter:
    name: str                         # "FL-det", "FL-stoch", ...
    agent: object                     # loaded FrozenLakeConsequenceDQNVectorized
    states: np.ndarray                # (S_eval,) non-terminal state ids to score
    qstar_spread: Optional[np.ndarray]  # (S_full,) exact stakes, or None (no oracle)
    n_actions: int


def non_terminal_states(env) -> np.ndarray:
    """State ids that are not terminal (drop H/G tiles — rollouts from them are degenerate)."""
    dn = np.asarray(env.dones)                       # (S,4,3) bool
    terminal = dn.all(axis=(1, 2))
    return np.where(~terminal)[0].astype(np.int32)


def outcome_probs(env) -> np.ndarray:
    """The 3-slot outcome probabilities [(a-1)%4, a, (a+1)%4].

    slip_probability set → env.outcome_probs; legacy slippery/deterministic → uniform (correct
    because the deterministic table stores the same outcome in all 3 slots)."""
    op = getattr(env, "outcome_probs", None)
    if op is not None:
        return np.asarray(op, dtype=np.float64)
    return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)


def qstar_spread_exact(env, gamma: float, tol: float = 1e-12, max_iter: int = 100_000) -> np.ndarray:
    """Exact per-state stakes = max_a Q*(s,a) − min_a Q*(s,a), via probability-weighted VI.

    Generalizes value_iteration.compute_qstar (which assumes equiprobable slips) to arbitrary
    outcome probabilities, so it is correct for is_slippery True/False AND any slip_probability."""
    ns = np.asarray(env.next_states)                 # (S,4,3) int
    rw = np.asarray(env.rewards, dtype=np.float64)   # (S,4,3)
    dn = np.asarray(env.dones).astype(np.float64)    # (S,4,3)
    probs = outcome_probs(env)                        # (3,)
    S = ns.shape[0]

    V = np.zeros(S, dtype=np.float64)
    for _ in range(max_iter):
        boot = (1.0 - dn) * V[ns]                     # (S,4,3)
        Q = (probs * (rw + gamma * boot)).sum(axis=2)  # (S,4) probability-weighted
        newV = Q.max(axis=1)
        if np.max(np.abs(newV - V)) < tol:
            V = newV
            break
        V = newV
    boot = (1.0 - dn) * V[ns]
    Q = (probs * (rw + gamma * boot)).sum(axis=2)
    return (Q.max(axis=1) - Q.min(axis=1)).astype(np.float32)


def make_frozenlake_adapter(agent, name: str, exact_truth: bool = True) -> SuitabilityAdapter:
    """Build a suitability adapter from a loaded FrozenLake agent."""
    states = non_terminal_states(agent.env)
    qss = qstar_spread_exact(agent.env, agent.gamma) if exact_truth else None
    return SuitabilityAdapter(name=name, agent=agent, states=states,
                              qstar_spread=qss, n_actions=agent.n_actions)
