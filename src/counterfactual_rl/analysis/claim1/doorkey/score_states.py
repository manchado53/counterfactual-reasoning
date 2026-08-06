"""
CCE scorer for all reachable non-terminal DoorKey states.

Loads a checkpoint, builds a vmapped rollout function (identical in structure to the
one used during training in agents/doorkey/consequence_dqn.py), and scores every
reachable non-terminal state. Mirrors analysis/claim1/frozen_lake/score_states.py but
generalised to DoorKey's 7-action space and the slippery (slip_prob>0) dynamics that
make the total-variation signal non-degenerate.
"""

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.doorkey.dqn import DoorKeyDQN
from counterfactual_rl.analysis.metrics import compute_consequence_metric


def _build_rollout_fn(env, network, horizon, gamma, n_actions):
    """
    Triple-vmapped JIT rollout function.

    Signature: fn(params, states(B,), actions(A,), keys(B,A,N,2)) -> returns(B,A,N)
    Forces `first_action` then rolls out greedily under the network for horizon-1 steps.
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


def score_all_states(ckpt_path, non_terminal, n_rollouts=100, horizon=60,
                     gamma=0.99, metric='total_variation', seed=42,
                     layout_name='6x6', slip_prob=0.2):
    """
    Load a DoorKey checkpoint and CCE-score every reachable non-terminal state.

    IMPORTANT: `agent.load()` restores only the network weights, not the environment
    the checkpoint was trained under — DoorKeyDQN() with no config would silently
    default to slip_prob=0.0 (DEFAULT_CONFIG, tuned for Claim 2). Claim 1 checkpoints
    are trained under slip_prob>0 (see agents/doorkey/run_experiments.py
    DOORKEY_CLAIM1) because CCE's total-variation signal is degenerate under
    deterministic dynamics (rollouts from a fixed policy are then delta functions).
    `slip_prob` here MUST match the training run's slip_prob and the oracle's.

    Returns dict {state_idx: float}. Uses `action_probs=None` (weighted_mean falls
    through to MAX over alternatives) to match the FrozenLake Claim-1 convention.
    """
    agent = DoorKeyDQN({'layout_name': layout_name, 'slip_prob': slip_prob})
    agent.load(ckpt_path)

    n_actions = agent.env.n_actions
    rollout_fn = _build_rollout_fn(agent.env, agent.network, horizon, gamma, n_actions)

    B = len(non_terminal)
    states_arr  = jnp.array(non_terminal, dtype=jnp.int32)                 # (B,)
    actions_arr = jnp.arange(n_actions, dtype=jnp.int32)                    # (A,)
    keys = jax.random.split(
        jax.random.PRNGKey(seed), B * n_actions * n_rollouts
    ).reshape(B, n_actions, n_rollouts, 2)

    returns = np.array(
        rollout_fn(agent.params, states_arr, actions_arr, keys)
    )  # (B, A, N)

    scores = {}
    for i, s in enumerate(non_terminal):
        greedy_a = int(jnp.argmax(agent.network.apply(agent.params, jnp.int32(s))))
        scores[s] = compute_consequence_metric(
            action=(greedy_a,),
            return_distributions={(a,): returns[i, a] for a in range(n_actions)},
            metric=metric,
            aggregation='weighted_mean',
        )

    return scores
