"""
Consequence-weighted DQN (Algorithm 2) for FrozenLake.

Variants selected via config:
  algorithm='consequence-dqn', mu=1.0             → DQN + CCE-only
  algorithm='consequence-dqn', priority_mixing='additive'       → DQN + CCE + TD additive (Eq. 4)
  algorithm='consequence-dqn', priority_mixing='multiplicative' → DQN + CCE + TD multiplicative (Eq. 5)

Simplified vs SMAX: no beam search — all 4 actions are always evaluated
since the action space is tiny and there are no action masks.
"""

import numpy as np
from typing import Optional

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .dqn import FrozenLakeDQN
from ..shared.consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.analysis.metrics import compute_consequence_metric


_ALL_ACTIONS = jnp.array([0, 1, 2, 3], dtype=jnp.int32)


class FrozenLakeConsequenceDQN(FrozenLakeDQN):
    """
    Consequence-weighted DQN for FrozenLake (Algorithm 2).

    Stores JAX state at each transition; periodically runs vmapped rollouts
    for all 4 actions from each scored state and uses the resulting return
    distributions to weight buffer sampling priorities.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        per = self.config.get('PER_parameters', {})
        self.buffer = ConsequenceReplayBuffer(
            capacity=self.config['buffer_capacity'],
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.25),
            max_priority=per.get('maximum_priority', 1.0),
            mu=self.config.get('mu', 0.5),
            priority_mixing=self.config.get('priority_mixing', 'additive'),
            mu_c=self.config.get('mu_c', 1.0),
            mu_delta=self.config.get('mu_delta', 1.0),
        )

        self.score_interval = self.config.get('score_interval', 100)
        self.n_score_sample = self.config.get('n_score_sample', 128)
        self.consequence_metric = self.config.get('consequence_metric', 'jensen_shannon')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')
        self.cf_horizon = self.config.get('cf_horizon', 10)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 20)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)

        self.q_update_count = 0
        self._compiled_rollout_fn = None

    def _build_rollout_fn(self):
        """
        Build triple-vmapped JIT rollout function.

        Axes:
          vmap over B transition states   → (B, 4, N)
          vmap over 4 actions             → (4, N)
          vmap over N rollout keys        → (N,)
          lax.scan over H horizon steps   → scalar return
        """
        env = self.env
        network = self.network
        horizon = self.cf_horizon
        gamma = self.cf_gamma

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
            return final[2]  # cumulative discounted return

        # vmap over N rollouts for one (state, action)
        over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, 0))
        # vmap over 4 actions for one state; keys shape (4, N, 2)
        over_actions = jax.vmap(over_rollouts, in_axes=(None, None, 0, 0))
        # vmap over B states; actions fixed (4,), keys shape (B, 4, N, 2)
        over_states = jax.vmap(over_actions, in_axes=(None, 0, None, 0))

        self._compiled_rollout_fn = jax.jit(over_states)

    def _score_buffer_transitions(self):
        """
        Algorithm 2: score a uniform sample of buffer transitions.

        Returns array shape (B, 4, N) of discounted returns, then computes
        consequence metric per transition and updates buffer scores.
        """
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        transitions, indices = self.buffer.sample_uniform(n_score)

        # Collect states that have saved jax_state
        valid_states = []
        valid_actions_taken = []
        valid_indices = []
        for i, (t, idx) in enumerate(zip(transitions, indices)):
            s = self.buffer.get_jax_state(idx)
            if s is None:
                continue
            valid_states.append(jnp.int32(s))
            valid_actions_taken.append(int(t['a']))
            valid_indices.append(i)

        if not valid_states:
            return

        B = len(valid_states)
        N = self.cf_n_rollouts

        # Build compiled function once
        if self._compiled_rollout_fn is None:
            print("Compiling consequence rollout function (one-time cost)...")
            self._build_rollout_fn()

        # states_array: (B,)  actions: (4,)  keys: (B, 4, N, 2)
        states_array = jnp.array(valid_states, dtype=jnp.int32)

        self._key, subkey = jax.random.split(self._key)
        keys_flat = jax.random.split(subkey, B * 4 * N)
        keys_array = keys_flat.reshape(B, 4, N, 2)

        returns = self._compiled_rollout_fn(
            self.params, states_array, _ALL_ACTIONS, keys_array
        )
        returns = jax.block_until_ready(returns)
        returns_np = np.array(returns)  # (B, 4, N)

        # Compute consequence score per transition
        scores = np.zeros(B)
        for i in range(B):
            taken_action = valid_actions_taken[i]
            return_distributions = {(a,): returns_np[i, a] for a in range(4)}
            scores[i] = compute_consequence_metric(
                action=(taken_action,),
                return_distributions=return_distributions,
                metric=self.consequence_metric,
                aggregation=self.consequence_aggregation,
            )

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        scored_indices = indices[np.array(valid_indices)]
        self.buffer.update_consequence_scores(scored_indices, scores)

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1

        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        transitions, indices, weights = self.buffer.sample(self.batch_size)
        states  = jnp.array([t['s']    for t in transitions], dtype=jnp.int32)
        actions = jnp.array([t['a']    for t in transitions], dtype=jnp.int32)
        rewards = jnp.array([t['r']    for t in transitions], dtype=jnp.float32)
        nexts   = jnp.array([t["s'"]   for t in transitions], dtype=jnp.int32)
        dones   = jnp.array([t['done'] for t in transitions], dtype=jnp.float32)
        wts     = jnp.array(weights, dtype=jnp.float32)

        self.params, self.opt_state, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, nexts, dones, wts,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'FrozenLakeConsequenceDQN':
        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 100)
        eval_episodes = self.config.get('eval_episodes', 50)
        save_every = self.config.get('save_every', 500)

        # Reuse parent metrics logger setup
        from .dqn import _MetricsLogger
        self.metrics_logger = _MetricsLogger(
            config=self.config,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        import os
        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_success = -1.0
        np.random.seed(self.config.get('seed', 0))

        if verbose:
            mixing = self.config.get('priority_mixing', 'additive')
            mu = self.config.get('mu', 0.5)
            print(f"Training Consequence-DQN on FrozenLake-{self.config['map_name']}")
            print(f"  Slippery: {self.config.get('is_slippery', True)}  |  n_states: {self.n_states}")
            print(f"  Priority mixing: {mixing}  |  mu: {mu}")
            print(f"  Metric: {self.consequence_metric}  |  Score interval: {self.score_interval}")
            print(f"  CF rollouts: {self.cf_n_rollouts}  |  CF horizon: {self.cf_horizon}")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            self._current_episode = episode
            timer.begin_episode(episode)

            self._key, rk = jax.random.split(self._key)
            _, state = self.env.reset(rk)
            state = int(state)

            done = False
            ep_return = 0.0
            ep_steps = 0

            while not done:
                saved_state = state  # save before step for counterfactual rollouts

                action = self.select_action(state)
                self._key, sk = jax.random.split(self._key)
                _, next_state, reward, done, _ = self.env.step(
                    sk, jnp.int32(state), jnp.int32(action)
                )
                next_state = int(next_state)
                reward = float(reward)
                done = bool(done)

                self.buffer.add(
                    {'s': state, 'a': action, 'r': reward, "s'": next_state, 'done': done},
                    jax_state=saved_state,
                )
                self.total_steps += 1

                if self.total_steps % self.n_steps_per_update == 0:
                    self._update()
                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()

                state = next_state
                ep_return += reward
                ep_steps += 1

            self.episode_returns.append(ep_return)
            self.episode_lengths.append(ep_steps)

            decay = min(1.0, episode / self.epsilon_decay_episodes)
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay

            if len(self.episode_returns) >= 100:
                avg = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"AvgR(100): {avg:.3f}  ε: {self.epsilon:.3f}")

            if (episode + 1) % save_every == 0:
                self.save(last_path)

            if eval_interval and (episode + 1) % eval_interval == 0:
                metrics = self.evaluate(eval_episodes)
                self.metrics_logger.log_eval(episode + 1, self.q_update_count, self.epsilon, metrics)
                if metrics['success_rate'] > best_success:
                    best_success = metrics['success_rate']
                    self.save(best_path)
                    if verbose:
                        print(f"\n  New best: {best_success:.1%} success rate at ep {episode + 1}")

            timer.flush_episode()

        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
