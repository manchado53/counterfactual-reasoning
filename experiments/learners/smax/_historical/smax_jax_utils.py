"""
JIT-compatible utility functions for SMAX counterfactual analysis.

These are JAX-traceable versions of functions in smax_utils.py,
designed to work inside jax.jit, jax.vmap, and jax.lax.scan.

Key differences from smax_utils.py:
- Returns JAX arrays instead of Python floats/ints
- Uses jax.random.categorical instead of variable-length indexing
- All functions are safe to call inside JIT-compiled code
"""

from typing import Dict, List, Callable
import jax
import jax.numpy as jnp


def jax_actions_array_to_dict(
    actions_array: jnp.ndarray,
    agent_names: List[str]
) -> Dict[str, jnp.ndarray]:
    """
    Convert a JAX array of actions to an SMAX action dict.

    Unlike tuple_to_action_dict, this keeps values as JAX arrays
    (not Python ints), making it compatible with jit/vmap.

    Args:
        actions_array: Shape (n_agents,) JAX integer array
        agent_names: List of agent name strings (static during tracing)

    Returns:
        Dict mapping agent name -> JAX scalar action
    """
    return {agent: actions_array[i] for i, agent in enumerate(agent_names)}


def jax_sum_rewards(
    rewards: Dict[str, jnp.ndarray],
    agent_names: List[str]
) -> jnp.ndarray:
    """
    Sum rewards across agents, returning a JAX scalar.

    JIT-compatible version of sum_rewards from smax_utils.py.

    Args:
        rewards: Dict mapping agent names to scalar JAX rewards
        agent_names: Agent name list (static during tracing)

    Returns:
        JAX scalar: total team reward
    """
    return jnp.sum(jnp.array([rewards[agent] for agent in agent_names]))


def make_jax_random_policy(agent_names: List[str]) -> Callable:
    """
    Create a JIT-compatible random policy for SMAX.

    Uses jax.random.categorical with masked logits instead of
    jnp.where(mask==1)[0] + jax.random.choice, which produces
    variable-length arrays and breaks JIT compilation.

    Args:
        agent_names: List of agent name strings

    Returns:
        Function (key, obs, avail_actions) -> action_dict
        where all values are JAX arrays (JIT-compatible)
    """
    def jax_random_policy(key, obs, avail_actions):
        keys = jax.random.split(key, len(agent_names))
        actions = {}
        for i, agent in enumerate(agent_names):
            mask = avail_actions[agent]
            # Large negative logits for invalid actions -> near-zero probability
            logits = jnp.where(mask == 1, 0.0, -1e9)
            action = jax.random.categorical(keys[i], logits)
            actions[agent] = action
        return actions

    return jax_random_policy
