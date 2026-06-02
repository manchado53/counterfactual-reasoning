"""
CCE scorer for all non-terminal FrozenLake states.

Loads a checkpoint, builds a vmapped rollout function (identical to the one used
during training in consequence_dqn.py), and scores every non-terminal state.
"""

import pickle

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN
from counterfactual_rl.analysis.metrics import compute_consequence_metric


def _build_rollout_fn(env, network, horizon, gamma):
    """
    Triple-vmapped JIT rollout function — identical to consequence_dqn._build_rollout_fn.

    Signature: fn(params, states(B,), actions(4,), keys(B,4,N,2)) → returns(B,4,N)
    """
    def single_rollout(params, state_idx, first_action, rng_key):
        rng_key, step_key = jax.random.split(rng_key)
        _, next_state, reward, done, _ = env.step(step_key, state_idx, first_action)
        init_carry = (next_state, rng_key, reward, jnp.float32(gamma), done)

        def scan_step(carry, _):
            s, key, cum_ret, disc, done_flag = carry
            q = network.apply(params, s)
            action = jnp.argmax(q)
            key, sk = jax.random.split(key)
            _, ns, r, nd, _ = env.step(sk, s, action)
            masked_r = jnp.where(done_flag, 0.0, r)
            new_cum = cum_ret + disc * masked_r
            new_disc = jnp.where(done_flag, disc, disc * gamma)
            new_done = jnp.logical_or(done_flag, nd)
            return (ns, key, new_cum, new_disc, new_done), None

        final, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=horizon - 1)
        return final[2]

    over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, 0))
    over_actions  = jax.vmap(over_rollouts,  in_axes=(None, None, 0,    0))
    over_states   = jax.vmap(over_actions,   in_axes=(None, 0,    None, 0))
    return jax.jit(over_states)


def score_all_states(ckpt_path, non_terminal, n_rollouts=20, horizon=10,
                     gamma=0.99, metric='total_variation', seed=42):
    """
    Load checkpoint and CCE-score every non-terminal state.

    Parameters
    ----------
    ckpt_path    : path to .pkl checkpoint (FrozenLakeDQN format)
    non_terminal : list of state indices to score (53 for 8×8)
    n_rollouts   : rollouts per (state, action) pair
    horizon      : rollout length in steps
    gamma        : discount factor
    metric       : divergence metric passed to compute_consequence_metric
    seed         : JAX PRNG seed for rollouts

    Returns
    -------
    dict {state_idx: float}
    """
    agent = FrozenLakeDQN()
    agent.load(ckpt_path)

    rollout_fn = _build_rollout_fn(agent.env, agent.network, horizon, gamma)

    B = len(non_terminal)
    states_arr  = jnp.array(non_terminal, dtype=jnp.int32)       # (B,)
    actions_arr = jnp.arange(4, dtype=jnp.int32)                  # (4,)
    keys = jax.random.split(
        jax.random.PRNGKey(seed), B * 4 * n_rollouts
    ).reshape(B, 4, n_rollouts, 2)

    returns = np.array(
        rollout_fn(agent.params, states_arr, actions_arr, keys)
    )  # (B, 4, N)

    scores = {}
    for i, s in enumerate(non_terminal):
        greedy_a = int(jnp.argmax(agent.network.apply(agent.params, jnp.int32(s))))
        scores[s] = compute_consequence_metric(
            action=(greedy_a,),
            return_distributions={(a,): returns[i, a] for a in range(4)},
            metric=metric,
            aggregation='weighted_mean',
        )

    return scores
