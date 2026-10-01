"""Collection boundaries must not change a trajectory with frozen Q and keys."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from counterfactual_rl.envs.jax_nav import JaxNavEnv
from counterfactual_rl.agents.jax_nav.dqn_vectorized import JaxNavDQNVectorized
from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized


class StationaryNetwork:
    def apply(self, params, obs):
        # Zero linear speed. Timeout is deterministic, independent of obstacles.
        return jnp.arange(15, dtype=jnp.float32) * 0 - jnp.arange(15, dtype=jnp.float32)


@pytest.mark.parametrize('parent', [JaxNavDQNVectorized, JaxNavConsequenceDQNVectorized])
@pytest.mark.parametrize('horizon', [3, 4, 5, 12])
def test_collection_boundary_preserves_episode(parent, horizon):
    env = JaxNavEnv(map_size=(8, 8), fill=.3, max_steps=horizon,
                    sparse_reward=False, goal_rew=4., coll_rew=-4.)
    agent = SimpleNamespace(env=env, network=StationaryNetwork(), n_actions=15)
    parent._build_collect_fn(agent)
    states = jax.vmap(lambda k: env.reset(k)[1])(jax.random.split(jax.random.PRNGKey(421), 2))
    keys = jax.random.split(jax.random.PRNGKey(81), 32).reshape(2, 16, 2)
    # Public collector contract: initialize once; carry the result between blocks.
    if hasattr(JaxNavDQNVectorized, '_initial_collect_carry'):
        initial = JaxNavDQNVectorized._initial_collect_carry(agent, states)
    else:
        initial = states
    final, whole = agent._collect_fn(None, jnp.float32(0), initial, keys)
    carry, left = agent._collect_fn(None, jnp.float32(0), initial, keys[:, :4])
    carry, right = agent._collect_fn(None, jnp.float32(0), carry, keys[:, 4:])
    joined = jax.tree.map(lambda a, b: np.concatenate([a, b], axis=1), left, right)
    for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(joined)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(carry)):
        np.testing.assert_array_equal(a, b)
    ended = np.asarray(whole[7])
    np.testing.assert_array_equal(np.asarray(whole[6])[ended], horizon)
    if len(whole) == 9:
        np.testing.assert_allclose(jax.vmap(jax.vmap(env.get_obs))(whole[8]), whole[0], rtol=1e-6, atol=1e-6)
        assert not np.asarray(whole[8].done).any()


@pytest.mark.parametrize('cce', [False, True])
def test_temporal_hold_carries_across_32_steps(cce):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis/claim2'))
    from train_carry_pilot import temporal_class
    parent = JaxNavConsequenceDQNVectorized if cce else JaxNavDQNVectorized
    cls = temporal_class(Path(__file__).resolve().parents[1] / 'analysis/claim2/train_temporal_replay.py', parent, cce)
    env = JaxNavEnv(map_size=(8, 8), fill=.3, max_steps=32, sparse_reward=False)
    agent = SimpleNamespace(env=env, network=StationaryNetwork(), n_actions=15,
                            config={'seed': 0, 'explore_hold_steps': 20})
    cls._build_collect_fn(agent)
    states = jax.vmap(lambda k: env.reset(k)[1])(jax.random.split(jax.random.PRNGKey(311), 2))
    initial = JaxNavDQNVectorized._initial_collect_carry(agent, states)
    keys = jax.random.split(jax.random.PRNGKey(93), 128).reshape(2, 64, 2)
    end, whole = agent._collect_fn(None, jnp.float32(0), initial, keys)
    carry, left = agent._collect_fn(None, jnp.float32(0), initial, keys[:, :32])
    carry, right = agent._collect_fn(None, jnp.float32(0), carry, keys[:, 32:])
    joined = jax.tree.map(lambda a, b: np.concatenate([a, b], axis=1), left, right)
    for a, b in zip(jax.tree.leaves((end, whole)), jax.tree.leaves((carry, joined))):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.asarray(whole[6])[np.asarray(whole[7])], 32)
