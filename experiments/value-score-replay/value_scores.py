"""Only intervention: target-Q gaps replace simulated return differences."""
import time
import jax
import jax.numpy as jnp
import numpy as np


def action_gaps(q, taken, weights=None):
    q = np.asarray(q, dtype=np.float64)
    taken = np.asarray(taken, dtype=int)
    gaps = abs(q - q[np.arange(len(q)), taken, None])
    if weights is None:
        return gaps.max(axis=1)
    weights = np.asarray(weights, dtype=np.float64).copy()
    weights[np.arange(len(q)), taken] = 0
    # Match historical left-to-right accumulation, including zero-weight fallback.
    numerator, denominator = np.zeros(len(q)), np.zeros(len(q))
    for action in range(q.shape[1]):
        numerator += weights[:, action] * gaps[:, action]
        denominator += weights[:, action]
    result = gaps.sum(axis=1) / (q.shape[1] - 1)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


class ValueBase:
    def _build_rollout_fn(self):
        raise AssertionError('Value arm must never construct or invoke scoring rollouts')

    def _values(self, observations):
        # Keep one parent key split per valid scoring call, even though Q is deterministic.
        # Splits of the discarded subkey never changed the parent's key in the old scorer.
        self._key, _ = jax.random.split(self._key)
        if not hasattr(self, '_value_apply'):
            self._value_apply = jax.jit(jax.vmap(self.network.apply, in_axes=(None, 0)))
        return np.asarray(self._value_apply(self.target_params, observations))


class FrozenLakeValue(ValueBase):
    def _score_buffer_transitions(self):
        n = min(self.n_score_sample, len(self.buffer))
        if n == 0:
            return
        started = time.monotonic()
        transitions, indices = self.buffer.sample_uniform(n)
        valid = [(i, self.buffer.get_jax_state(idx), int(t['a']))
                 for i, (t, idx) in enumerate(zip(transitions, indices))
                 if self.buffer.get_jax_state(idx) is not None]
        if not valid:
            return
        slots, states, actions = zip(*valid)
        q = self._values(jnp.asarray(states, dtype=jnp.int32))
        scores = action_gaps(q, actions)
        assert np.isfinite(scores).all()
        self.buffer.update_consequence_scores(indices[np.asarray(slots)], scores)
        self.value_score_calls = getattr(self, 'value_score_calls', 0) + 1
        self.value_score_seconds = getattr(self, 'value_score_seconds', 0.) + time.monotonic()-started


class JaxNavValue(ValueBase):
    def _score_values(self, obs_list, actions):
        obs = jnp.asarray(np.stack(obs_list), dtype=jnp.float32)
        q_target = self._values(obs)
        # Preserve the reference's ONLINE-network action weights and temperature.
        q_online = self._value_apply(self.params, obs)
        probs = np.asarray(jax.nn.softmax(q_online / self.config['cf_action_prob_temperature'], axis=-1))
        scores = action_gaps(q_target, actions, probs)
        assert np.isfinite(scores).all()
        self.value_score_calls = getattr(self, 'value_score_calls', 0) + 1
        return scores, np.asarray(q_online)

    def _score_index_chunk(self, indices):
        started = time.monotonic()
        valid = [(i, self.buffer.get_transition(idx)) for i, idx in enumerate(indices)
                 if self.buffer.get_jax_state(idx) is not None]
        if not valid:
            return
        scores, _ = self._score_values([t['s'] for _, t in valid], [int(t['a']) for _, t in valid])
        self.buffer.update_consequence_scores(np.asarray(indices)[[i for i, _ in valid]], scores,
                                              q_update_count=self.q_update_count)
        self._last_rollout_n = 0
        self._last_rollout_reasons = {}
        self.value_score_seconds = getattr(self, 'value_score_seconds', 0.) + time.monotonic()-started

    def _score_states(self, states_batched, obs_list, actions_taken):
        # Keep the same probe, timing, key advancement, and host RNG use.
        return self._score_values(obs_list, actions_taken)
