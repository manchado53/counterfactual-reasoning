"""
Shared counterfactual rollout factory (repo issue #12).

The CCE rollout was copy-pasted across five agents (frozen_lake, smax, chess,
connect_four, jax_nav) with no shared helper, and the copies have already drifted
apart enough to cause defects. This module is the single implementation.

It is lifted verbatim from ``agents/jax_nav/consequence_dqn.py::_build_rollout_fn``
-- the only copy that threads the observation through the scan carry, supports a
stochastic continuation policy, takes ``target_params``, and has a terminal-value
(bootstrap) hook. ``tests/test_cf_rollout_parity.py`` asserts the output here is
bit-identical to that copy on a fixed seed.

NOTE: jax_nav has NOT been ported onto this module yet -- another session is
mid-flight in that file. Port it once that work lands; until then jax_nav keeps
its own copy and the parity test is what keeps the two honest.

The plan sketched a separate ``obs_fn`` argument. It is not needed: every
observation the rollout consumes already falls out of ``step_fn``, and the first
one comes from stepping the stored state. Passing an ``obs_fn`` would only invite
a second, disagreeing source of observations.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp


def build_rollout_fn(
    step_fn: Callable,
    q_apply: Callable,
    *,
    horizon: int,
    gamma: float,
    temperature: float = 0.0,
    bootstrap: bool = False,
    jit: bool = True,
):
    """Build a triple-vmapped counterfactual rollout.

    Args:
        step_fn:     ``(key, state, action) -> (obs, next_state, reward, done, info)``.
                     Must be the RAW transition -- no auto-reset wrapper, or a
                     counterfactual branch gets silently corrupted at a terminal.
        q_apply:     ``(params, obs) -> (n_actions,)`` Q-values for one observation.
        horizon:     total env steps per rollout (the forced first action plus
                     ``horizon - 1`` more). ``horizon=1`` runs NO rollout at all --
                     the scan length is zero and the score is just the spread of
                     immediate rewards. That is a useful one-step baseline, not a bug.
        gamma:       rollout discount. Separate from the training gamma on purpose.
        temperature: continuation policy. ``0`` (default) is greedy argmax, which is
                     exact for a deterministic env and lets ``cf_n_rollouts=1``.
                     ``> 0`` samples from ``softmax(q / temperature)``, which is what
                     spreads the N rollouts when a bounded metric would otherwise
                     collapse to 0/1.
        bootstrap:   add ``gamma**horizon * V(obs_H)`` for rollouts truncated by the
                     horizon rather than by a real terminal (repo issue #7). Scoring a
                     truncated rollout as if it ended with zero value is a bias, and
                     it is worst exactly at the short horizons that make scoring cheap.
        jit:         wrap the result in ``jax.jit``. Off is for debugging only.

    Returns:
        ``f(params, target_params, states, first_actions, keys) -> (B, A, N)``
        cumulative discounted returns, where ``states`` is a state pytree batched on
        a leading axis of B, ``first_actions`` has shape ``(A,)`` and ``keys`` has
        shape ``(B, A, N, 2)``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    def single_rollout(params, target_params, state, first_action, rng_key):
        rng_key, step_key = jax.random.split(rng_key)
        next_obs, next_state, reward, done, _ = step_fn(step_key, state, first_action)
        init_carry = (next_state, next_obs, rng_key, reward, jnp.float32(gamma), done)

        def scan_step(carry, _):
            s, obs, key, cum_ret, disc, done_flag = carry
            q = q_apply(params, obs)
            key, ak, sk = jax.random.split(key, 3)
            if temperature and temperature > 0:
                # stochastic continuation -> spreads the N rollouts (deterministic env)
                action = jax.random.categorical(ak, q / temperature)
            else:
                action = jnp.argmax(q)
            n_obs, ns, r, nd, _ = step_fn(sk, s, action)
            masked_r = jnp.where(done_flag, 0.0, r)
            new_cum = cum_ret + disc * masked_r
            new_disc = jnp.where(done_flag, disc, disc * gamma)
            new_done = jnp.logical_or(done_flag, nd)
            return (ns, n_obs, key, new_cum, new_disc, new_done), None

        final, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=horizon - 1)
        _, obs_H, _, cum, disc, done = final
        if not bootstrap:
            return cum  # cumulative discounted return, truncated at the horizon

        # The scan ran `horizon` env steps in total (the first action plus
        # horizon-1 more), and `disc` is frozen at gamma**horizon, so it is
        # exactly the discount this bootstrap term belongs at. `done` is only
        # true for a REAL terminal, and those are correctly worth 0, so the
        # term is masked off for them.
        q_H = q_apply(target_params, obs_H)
        if temperature and temperature > 0:
            # Value under the policy the rollout actually followed. A max()
            # bootstrap would assume greedy continuation and systematically
            # overestimate every truncated rollout.
            v_H = jnp.sum(jax.nn.softmax(q_H / temperature) * q_H)
        else:
            v_H = jnp.max(q_H)
        return cum + jnp.where(done, 0.0, disc * v_H)

    over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, None, 0))  # N keys
    over_actions  = jax.vmap(over_rollouts,  in_axes=(None, None, None, 0, 0))     # A actions
    over_states   = jax.vmap(over_actions,   in_axes=(None, None, 0, None, 0))     # B states
    return jax.jit(over_states) if jit else over_states
