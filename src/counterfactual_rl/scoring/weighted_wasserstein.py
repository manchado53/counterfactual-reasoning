"""Shared weighted-Wasserstein consequence score for current CCE learners."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def weighted_wasserstein_scores(
    returns,
    action_probabilities,
    taken_actions,
    *,
    legal_actions=None,
):
    """Score each taken action against its legal alternatives.

    ``returns`` is ``(batch, actions)`` for one rollout or
    ``(batch, actions, rollouts)`` for many. Alternative distances are weighted
    by the supplied action probabilities. The taken action is always excluded.
    A zero-weight row falls back to the plain mean over legal alternatives.
    """
    values = np.asarray(returns, dtype=np.float64)
    probabilities = np.asarray(action_probabilities)
    taken = np.asarray(taken_actions, dtype=np.int64)
    if values.ndim not in (2, 3):
        raise ValueError("returns must have shape (batch, actions[, rollouts])")
    if probabilities.shape != values.shape[:2]:
        raise ValueError("action probabilities must match return batch and actions")
    if taken.shape != (values.shape[0],):
        raise ValueError("taken actions must have shape (batch,)")
    if np.any((taken < 0) | (taken >= values.shape[1])):
        raise ValueError("taken action is outside the action space")
    legal = (
        np.ones(values.shape[:2], dtype=bool)
        if legal_actions is None
        else np.asarray(legal_actions, dtype=bool)
    )
    if legal.shape != values.shape[:2]:
        raise ValueError("legal-action mask must match return batch and actions")
    if np.any(~legal[np.arange(values.shape[0]), taken]):
        raise ValueError("taken action must be legal")

    batch_size, n_actions = probabilities.shape
    if values.ndim == 2 or values.shape[2] == 1:
        scalar_returns = values if values.ndim == 2 else values[:, :, 0]
        reference = scalar_returns[np.arange(batch_size), taken]
        numerator = np.zeros(batch_size, dtype=np.float64)
        denominator = np.zeros(batch_size, dtype=np.float64)
        for action in range(n_actions):
            alternative = legal[:, action] & (taken != action)
            weight = probabilities[:, action].astype(np.float64)
            distance = np.abs(reference - scalar_returns[:, action])
            numerator = np.where(
                alternative, numerator + weight * distance, numerator
            )
            denominator = np.where(alternative, denominator + weight, denominator)
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        for row in np.flatnonzero(denominator <= 0):
            alternatives = legal[row].copy()
            alternatives[taken[row]] = False
            distances = np.abs(reference[row] - scalar_returns[row, alternatives])
            scores[row] = float(np.mean(distances)) if len(distances) else 0.0
        return scores

    scores = np.zeros(batch_size, dtype=np.float64)
    for row in range(batch_size):
        distances = np.zeros(n_actions, dtype=np.float64)
        weights = probabilities[row].astype(np.float64).copy()
        weights[~legal[row]] = 0.0
        weights[taken[row]] = 0.0
        reference = values[row, taken[row]]
        for action in range(n_actions):
            if action == taken[row] or not legal[row, action]:
                continue
            distances[action] = wasserstein_distance(reference, values[row, action])
        alternatives = legal[row].copy()
        alternatives[taken[row]] = False
        if not alternatives.any():
            continue
        total_weight = float(weights.sum())
        scores[row] = (
            float(np.dot(weights, distances) / total_weight)
            if total_weight > 0
            else float(np.mean(distances[alternatives]))
        )
    return scores
