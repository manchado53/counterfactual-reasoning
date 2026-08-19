"""
CCE scorer for routing (CVRP / TSP) decision states.

Loads a checkpoint, builds a vmapped rollout function (identical in shape to the one
used during training), and scores states by the total-variation divergence among the
per-action return distributions — the same signal as every other env.

Adapted from analysis/claim1/frozen_lake/score_states.py, with three routing changes:

1. The ROLLOUT POLICY is masked. Rollouts continue under the current greedy policy; if
   that argmax were unmasked it would pick already-served stops and the whole return
   distribution would be garbage.
2. Only LEGAL actions enter the consequence metric. Counterfactual rollouts are computed
   for all n_actions (a fixed shape keeps the vmap simple), but illegal ones self-loop
   with a penalty and are dropped before the divergence is taken — otherwise every state
   would look consequential purely because of the illegal-action penalty.
3. States are scored in CHUNKS. Routing has tens of thousands of states (vs FrozenLake's
   53), so scoring them in one vmapped call would exhaust memory.
"""

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.cvrp.dqn import CVRPDQN, MASK_FILL
from counterfactual_rl.analysis.metrics import compute_consequence_metric


def _build_rollout_fn(env, network, horizon, gamma):
    """
    Triple-vmapped JIT rollout function.

    Signature: fn(params, states(B,), actions(A,), keys(B,A,N,2)) -> returns(B,A,N)
    """
    features = env.state_features
    masks = env.action_masks

    def single_rollout(params, state_idx, first_action, rng_key):
        rng_key, step_key = jax.random.split(rng_key)
        _, next_state, reward, done, _ = env.step(step_key, state_idx, first_action)
        init_carry = (next_state, rng_key, reward, jnp.float32(gamma), done)

        def scan_step(carry, _):
            s, key, cum_ret, disc, done_flag = carry
            q = network.apply(params, features[s])
            # Masked greedy — the rollout policy must respect the legal set.
            q = jnp.where(masks[s], q, MASK_FILL)
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


def score_states(ckpt_path, states, config=None, n_rollouts=20, horizon=30,
                 gamma=0.99, metric='total_variation', seed=42, chunk_size=256,
                 agent=None, verbose=False):
    """
    Load a checkpoint and CCE-score the given routing states.

    Parameters
    ----------
    ckpt_path  : path to a .pkl checkpoint (CVRPDQN format); ignored if `agent` is given
    states     : iterable of state indices to score (typically the oracle's decision states)
    config     : config dict used to rebuild the env (must match the checkpoint's instance)
    n_rollouts : rollouts per (state, action) pair
    horizon    : rollout length in steps — must cover a full plan
    gamma      : discount for the counterfactual return
    metric     : divergence passed to compute_consequence_metric
    seed       : JAX PRNG seed
    chunk_size : states scored per vmapped call (memory control)
    agent      : optional pre-loaded CVRPDQN (skips reloading for repeated calls)

    Returns
    -------
    dict {state_idx: float}
    """
    if agent is None:
        agent = CVRPDQN(config or {})
        agent.load(ckpt_path)

    env = agent.env
    masks_np = np.asarray(env.action_masks)
    n_actions = env.n_actions

    rollout_fn = _build_rollout_fn(env, agent.network, horizon, gamma)
    actions_arr = jnp.arange(n_actions, dtype=jnp.int32)

    states = list(states)
    scores = {}
    key = jax.random.PRNGKey(seed)

    for start in range(0, len(states), chunk_size):
        chunk = states[start:start + chunk_size]
        B = len(chunk)
        states_arr = jnp.array(chunk, dtype=jnp.int32)
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, B * n_actions * n_rollouts).reshape(
            B, n_actions, n_rollouts, 2
        )
        returns = np.array(rollout_fn(agent.params, states_arr, actions_arr, keys))

        # Greedy (masked) action per state — the "taken" action for the divergence.
        feats = env.state_features[states_arr]
        q = jax.vmap(agent.network.apply, in_axes=(None, 0))(agent.params, feats)
        q = jnp.where(env.action_masks[states_arr], q, MASK_FILL)
        greedy = np.array(jnp.argmax(q, axis=-1))

        for i, s in enumerate(chunk):
            legal = np.flatnonzero(masks_np[s])
            if legal.size < 2:
                continue  # forced move: no alternatives, so no consequence to measure
            scores[int(s)] = compute_consequence_metric(
                action=(int(greedy[i]),),
                return_distributions={(int(a),): returns[i, a] for a in legal},
                metric=metric,
                aggregation='weighted_mean',
            )

        if verbose:
            print(f"  scored {min(start + chunk_size, len(states))}/{len(states)} states")

    return scores
