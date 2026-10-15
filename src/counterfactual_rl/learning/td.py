"""One TD definition for all current DQN learners.

Environment adapters produce Q arrays, rewards, terminal flags, and optional
legal-action masks. This module owns action selection, bootstrapping, TD error,
and the importance-weighted squared loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def q_learning_targets(
    rewards,
    terminated,
    online_next_q,
    target_next_q,
    gamma: float,
    *,
    double_dqn: bool,
    legal_actions=None,
):
    """Return stopped Q-learning targets.

    ``terminated`` means a true MDP terminal. Collection timeouts must be false
    when timeout bootstrapping is enabled. Every nonterminal row must have at
    least one legal action when ``legal_actions`` is supplied.
    """
    selection_q = online_next_q if double_dqn else target_next_q
    if legal_actions is not None:
        selection_q = jnp.where(legal_actions, selection_q, -jnp.inf)
    next_actions = jnp.argmax(jax.lax.stop_gradient(selection_q), axis=-1)
    next_values = jnp.take_along_axis(
        target_next_q, next_actions[..., None], axis=-1
    )[..., 0]
    if legal_actions is not None:
        has_legal_action = jnp.any(legal_actions, axis=-1)
        next_values = jnp.where(has_legal_action, next_values, jnp.nan)
    targets = rewards + gamma * jnp.where(terminated, 0.0, next_values)
    return jax.lax.stop_gradient(targets)


def weighted_td_loss(q_values, actions, targets, weights):
    """Return weighted squared loss, signed TD errors, and chosen Q values."""
    chosen_q = jnp.take_along_axis(q_values, actions[..., None], axis=-1)[..., 0]
    td_errors = targets - chosen_q
    loss = jnp.mean(weights * td_errors**2)
    return loss, td_errors, chosen_q
