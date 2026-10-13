"""Historical hold-exploration collector, extracted verbatim from the sealed suite."""
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .dqn_vectorized_legacy import _tree_where


def temporal_class(Parent, cce):
    prefix = 9 if cce else 8
    class TemporalAgent(Parent):
        def _build_collect_fn(self):
            env, net, seed, hold = self.env, self.network, self.config['seed'], self.config['explore_hold_steps']

            def collect(params, epsilon, init_state, step_keys):
                def step(carry, key):
                    state, obs, ended, total, length = carry
                    kr, _, _, ke = jax.random.split(key, 4)
                    new_obs, new_state = env.reset(kr)
                    state = _tree_where(ended, new_state, state)
                    obs = jnp.where(ended, new_obs, obs)
                    total = jnp.where(ended, jnp.float32(0), total)
                    length = jnp.where(ended, jnp.int32(0), length)
                    # Exact random-key rule of uniform array 278653. The draw is
                    # fixed inside a block; epsilon can change at chunk boundaries.
                    bits = jax.lax.bitcast_convert_type(state.goal[0], jnp.uint32)
                    k = jax.random.PRNGKey(seed)
                    k = jax.random.fold_in(jax.random.fold_in(k, bits[0]), bits[1])
                    ka, kx = jax.random.split(jax.random.fold_in(k, state.step // hold))
                    draw = jax.random.uniform(kx)
                    random_action = jax.random.randint(ka, (), 0, self.n_actions)
                    explore = draw < epsilon
                    action = jnp.where(explore, random_action, jnp.argmax(net.apply(params, obs)))
                    ob, ns, reward, done, _ = env.step(ke, state, action)
                    total = total + reward
                    length = length + jnp.int32(1)
                    standard = (obs, action, reward, ob, done,
                                jnp.where(done, total, 0.), jnp.where(done, length, 0), done)
                    if cce:
                        standard += (state,)  # Parent CCE consumes slot 8 unchanged.
                    return (ns, ob, done, total, length), standard + (explore, state.goal[0], state.step, draw)

                carry = (init_state, env.get_obs(init_state), jnp.bool_(False), jnp.float32(0), jnp.int32(0))
                (state, _, _, _, _), outputs = jax.lax.scan(step, carry, step_keys)
                return state, outputs
            self._collect_fn = jax.jit(jax.vmap(collect, in_axes=(None, None, 0, 0)))

        def _add_chunk_to_buffer(self, outputs):
            start = time.monotonic()
            super()._add_chunk_to_buffer(outputs[:prefix])
            explore = np.asarray(outputs[prefix])
            self._explore_n = getattr(self, '_explore_n', 0) + int(explore.sum())
            self._collect_n = getattr(self, '_collect_n', 0) + explore.size
            self._chunk_n = getattr(self, '_chunk_n', 0) + 1
            row = dict(episode=self._current_episode, chunk=self._chunk_n, epsilon=float(self.epsilon),
                       hold=self.config['explore_hold_steps'], explore_fraction=float(explore.mean()),
                       cumulative_explore_n=self._explore_n, cumulative_transition_n=self._collect_n,
                       scoring_insert_seconds=time.monotonic() - start)
            if cce:
                n = len(self.buffer)
                row['scored_frac'] = float(np.mean(self.buffer.scored_at[:n] >= 0))
                assert row['scored_frac'] == 1., 'CCE lost score-on-insert coverage'
            with (Path(self.metrics_logger.dir) / 'exploration.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
    return TemporalAgent
