"""Explicit stochastic-lake transfer recipe; legacy scorers remain separate."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from .dqn_vectorized import FrozenLakeDQNVectorized
from .consequence_dqn_vectorized import FrozenLakeConsequenceDQNVectorized
from counterfactual_rl.replay.consequence_buffers import ConsequenceReplayBuffer


def build_return_sampler(env, horizon, gamma):
    """Greedy policy is frozen during scoring; table lookup equals network argmax."""
    def one(policy, state, action, key):
        key, sk = jax.random.split(key)
        _, ns, reward, done, _ = env.step(sk, state, action)
        def step(carry, _):
            state, key, ret, discount, ended = carry
            key, sk = jax.random.split(key)
            _, ns, reward, done, _ = env.step(sk, state, policy[state])
            return (ns, key, ret + discount * jnp.where(ended, 0., reward),
                    discount * gamma, ended | done), None
        final, _ = jax.lax.scan(step, (ns, key, reward, jnp.float32(gamma), done), None, length=horizon-1)
        return final[2]
    over_trials = jax.vmap(one, in_axes=(None, None, None, 0))
    over_actions = jax.vmap(over_trials, in_axes=(None, None, 0, 0))
    return jax.jit(jax.vmap(over_actions, in_axes=(None, 0, None, 0)))


def score_table(returns, action_probs):
    """Weighted alternative-action Wasserstein distance, excluding taken action."""
    n_states, n_actions, _ = returns.shape
    result = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        distances = np.zeros((n_actions, n_actions))
        for a in range(n_actions):
            for b in range(a):
                distances[a, b] = distances[b, a] = wasserstein_distance(returns[s, a], returns[s, b])
        for a in range(n_actions):
            weights = np.asarray(action_probs[s], dtype=np.float64).copy()
            weights[a] = 0.
            result[s, a] = (np.dot(weights, distances[a]) / weights.sum() if weights.sum() > 0
                            else np.mean(np.delete(distances[a], a)))
    return result


class TransferMixin:
    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return
        transitions, indices, weights = self.buffer.sample(self.batch_size)
        s = jnp.asarray([t['s'] for t in transitions], dtype=jnp.int32)
        a = jnp.asarray([t['a'] for t in transitions], dtype=jnp.int32)
        r = jnp.asarray([t['r'] for t in transitions], dtype=jnp.float32)
        ns = jnp.asarray([t["s'"] for t in transitions], dtype=jnp.int32)
        d = jnp.asarray([t['done'] for t in transitions], dtype=jnp.float32)
        self.params, self.opt_state, td = self._update_step(
            self.params, self.target_params, self.opt_state, s, a, r, ns, d,
            jnp.asarray(weights, dtype=jnp.float32))
        td = np.asarray(td)
        self._last_loss = float(np.mean(np.asarray(weights, dtype=np.float32) * td**2))
        self._last_td_abs = float(np.mean(np.abs(td)))
        assert np.isfinite(self._last_loss), 'Non-finite TD loss'
        self.buffer.update_priorities(indices, td)
        self.q_update_count += 1


class TransferDQN(TransferMixin, FrozenLakeDQNVectorized):
    def __init__(self, config):
        super().__init__(config)
        self.q_update_count = 0


class TransferCCE(TransferMixin, FrozenLakeConsequenceDQNVectorized):
    def __init__(self, config):
        super().__init__(config)
        assert config['score_on_insert'] and config['consequence_metric'] == 'wasserstein'
        per = config['PER_parameters']
        self.buffer = ConsequenceReplayBuffer(
            capacity=config['buffer_capacity'], eps=per['eps'], beta=per['beta'],
            max_priority=per['maximum_priority'], priority_mixing=config['priority_mixing'],
            mu=config['mu'], mu_c=config['mu_c'], mu_delta=config['mu_delta'],
            normalise_priorities=config['normalise_priorities'])
        self._score_key = jax.random.PRNGKey(config['seed'] + 730001)
        self._return_sampler = build_return_sampler(self.env, self.cf_horizon, self.cf_gamma)
        self._scored_arrivals = 0

    def _add_chunk_to_buffer(self, outputs):
        s, a, r, ns, done = [np.asarray(x).reshape(-1) for x in outputs[:5]]
        assert len(s) <= self.buffer.capacity, 'Cannot score already overwritten arrivals'
        q = np.asarray(jax.vmap(self.network.apply, in_axes=(None, 0))(
            self.params, jnp.arange(self.n_states, dtype=jnp.int32)))
        policy = jnp.asarray(q.argmax(axis=1), dtype=jnp.int32)
        self._score_key, key = jax.random.split(self._score_key)
        keys = jax.random.split(key, self.n_states * 4 * self.cf_n_rollouts).reshape(self.n_states, 4, self.cf_n_rollouts, 2)
        returns = np.asarray(self._return_sampler(policy, jnp.arange(self.n_states, dtype=jnp.int32),
                                                 jnp.arange(4, dtype=jnp.int32), keys))
        probs = np.asarray(jax.nn.softmax(jnp.asarray(q) / self.config['cf_action_prob_temperature'], axis=-1))
        scores = score_table(returns, probs)
        written = self.buffer.add_batch(dict(s=s, a=a, r=r, **{"s'": ns}, done=done), jax_states=s)
        self.buffer.update_consequence_scores(written, scores[s, a], q_update_count=self.q_update_count)
        self._scored_arrivals += len(s)
        assert np.all(self.buffer.scored_at[:len(self.buffer)] >= 0), 'Lost score-on-insert coverage'
        assert np.isfinite(self.buffer.consequence_scores[:len(self.buffer)]).all()


def exact_return_law(env, policy, horizon, gamma):
    """Reference ONLY: law of first-goal time under a fixed policy, not training scores."""
    n, a = env.n_states, env.n_actions
    transient = np.zeros((n, a, n))
    goal = np.zeros((n, a))
    for s in range(n):
        for action in range(a):
            for p, ns, r, done in env.P[s][action]:
                if r > 0:
                    goal[s, action] += p
                elif not done:
                    transient[s, action, ns] += p
    masses = [goal]
    reach = goal[np.arange(n), policy]
    for _ in range(1, horizon):
        next_mass = np.einsum('sak,k->sa', transient, reach)
        masses.append(next_mass)
        reach = next_mass[np.arange(n), policy]
    mass = np.stack(masses, axis=-1)
    zero = np.maximum(0., 1. - mass.sum(axis=-1, keepdims=True))
    return np.r_[0., gamma ** np.arange(horizon)], np.concatenate([zero, mass], axis=-1)
