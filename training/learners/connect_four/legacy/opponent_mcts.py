"""
MCTS opponent for Connect Four via google-deepmind/mctx.

Exports a single function matching the rule_based_action signature:
    mcts_action(state, rng_key, n_sims=32) -> jnp.int32

Works on a single pgx State (batch_size=1 internally).
When wrapped in jax.vmap by the caller, JAX vectorizes it across N_ENVS.

Uses gumbel_muzero_policy (Sequential Halving) with:
  - uniform prior logits (no value network needed)
  - zero value estimates
  - pgx env.step as the recurrent function
"""

import jax
import jax.numpy as jnp
import pgx
import mctx

C4_ACTIONS = 7
_env = pgx.make('connect_four')


def _recurrent_fn(params, rng_key, action, state):
    """One env step for mctx tree expansion. Called inside mctx's internal vmap."""
    current_player = state.current_player
    state = _env.step(state, action)
    reward   = state.rewards[current_player]
    discount = jnp.where(state.terminated, jnp.float32(0.0), jnp.float32(-1.0))
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=jnp.zeros(C4_ACTIONS),
        value=jnp.float32(0.0),
    ), state


def mcts_action(state, rng_key, n_sims=32):
    """
    Select a Connect Four action using MCTS.

    Args:
        state:    single pgx Connect Four State (not batched)
        rng_key:  JAX PRNGKey
        n_sims:   number of MCTS simulations (default 32)

    Returns:
        int32 scalar — chosen column (0–6)
    """
    # mctx requires a batch dimension — add it here (batch_size=1).
    # When this function is called inside jax.vmap over N_ENVS, JAX fuses
    # all batch=1 calls into a single batch=N_ENVS computation at the XLA level.
    batched_state = jax.tree.map(lambda x: x[None], state)

    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros((1, C4_ACTIONS)),
        value=jnp.zeros(1),
        embedding=batched_state,
    )

    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=jax.vmap(_recurrent_fn, in_axes=(None, None, 0, 0)),
        num_simulations=n_sims,
        invalid_actions=~state.legal_action_mask[None],  # (1, 7)
        max_depth=42,
        max_num_considered_actions=C4_ACTIONS,
    )

    return policy_output.action[0]  # squeeze batch dim → scalar int32
