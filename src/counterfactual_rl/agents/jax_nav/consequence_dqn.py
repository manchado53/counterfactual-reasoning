"""
Consequence-weighted DQN (Algorithm 2) for JaxNav.

Mirror of ``agents/frozen_lake/consequence_dqn.py``. Same scoring pipeline —
store the env state at each transition, periodically roll out all actions from a
uniform sample of stored states, and weight replay priorities by the spread
(total variation) among the per-action return distributions.

Two JaxNav-specific changes, both discussed in the plan:
  * State is a pytree (not an int): stored ``jax_state`` is a full JaxNav ``State``;
    rollouts thread the observation to the network while stepping the state.
  * The rollout continuation policy is STOCHASTIC (softmax temperature
    ``cf_rollout_temperature``). JaxNav transitions are deterministic, so a greedy
    continuation would collapse each action's N rollouts to a single value and TV
    would degrade to a coarse 0/1; sampling restores graded spread. Set the
    temperature to 0 for a greedy ablation.

Variants (identical knobs to FrozenLake):
  algorithm='consequence-dqn', mu=1.0                            → CCE-only
  algorithm='consequence-dqn', priority_mixing='additive'        → CCE + TD additive (Eq. 4)
  algorithm='consequence-dqn', priority_mixing='multiplicative'  → CCE + TD multiplicative (Eq. 5)
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .dqn import JaxNavDQN, _MetricsLogger
from ..shared.consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.analysis.metrics import compute_consequence_metric


class JaxNavConsequenceDQN(JaxNavDQN):
    """Consequence-weighted DQN for JaxNav (Algorithm 2)."""

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
        self.consequence_metric = self.config.get('consequence_metric', 'total_variation')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')
        self.cf_horizon = self.config.get('cf_horizon', 20)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 20)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)
        self.cf_rollout_temperature = self.config.get('cf_rollout_temperature', 0.5)

        self._all_actions = jnp.arange(self.n_actions, dtype=jnp.int32)
        self.q_update_count = 0
        self._compiled_rollout_fn = None

    # ------------------------------------------------------------------
    # Counterfactual rollout
    # ------------------------------------------------------------------

    def _build_rollout_fn(self):
        """Triple-vmapped JIT rollout: (B states) x (n_actions) x (N rollouts) -> returns (B, A, N)."""
        env = self.env
        network = self.network
        horizon = self.cf_horizon
        gamma = self.cf_gamma
        temperature = self.cf_rollout_temperature

        def single_rollout(params, state, first_action, rng_key):
            rng_key, step_key = jax.random.split(rng_key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, first_action)
            init_carry = (next_state, next_obs, rng_key, reward, jnp.float32(gamma), done)

            def scan_step(carry, _):
                s, obs, key, cum_ret, disc, done_flag = carry
                q = network.apply(params, obs)
                key, ak, sk = jax.random.split(key, 3)
                if temperature and temperature > 0:
                    # stochastic continuation -> spreads the N rollouts (deterministic env)
                    action = jax.random.categorical(ak, q / temperature)
                else:
                    action = jnp.argmax(q)
                n_obs, ns, r, nd, _ = env.step(sk, s, action)
                masked_r = jnp.where(done_flag, 0.0, r)
                new_cum = cum_ret + disc * masked_r
                new_disc = jnp.where(done_flag, disc, disc * gamma)
                new_done = jnp.logical_or(done_flag, nd)
                return (ns, n_obs, key, new_cum, new_disc, new_done), None

            final, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=horizon - 1)
            return final[3]  # cumulative discounted return

        over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, 0))   # N keys
        over_actions  = jax.vmap(over_rollouts,  in_axes=(None, None, 0, 0))      # A actions
        over_states   = jax.vmap(over_actions,   in_axes=(None, 0, None, 0))      # B states (pytree)
        self._compiled_rollout_fn = jax.jit(over_states)

    def _score_buffer_transitions(self):
        """Score a uniform sample of stored transitions by counterfactual return spread."""
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        transitions, indices = self.buffer.sample_uniform(n_score)

        valid_states, valid_actions_taken, valid_indices = [], [], []
        for i, (t, idx) in enumerate(zip(transitions, indices)):
            s = self.buffer.get_jax_state(idx)
            if s is None:
                continue
            valid_states.append(s)                 # a JaxNav State pytree (numpy leaves)
            valid_actions_taken.append(int(t['a']))
            valid_indices.append(i)

        if not valid_states:
            return

        B = len(valid_states)
        N = self.cf_n_rollouts
        A = self.n_actions

        if self._compiled_rollout_fn is None:
            print("Compiling consequence rollout function (one-time cost)...")
            self._build_rollout_fn()

        # Stack the per-transition State pytrees into one batched pytree (leading axis B).
        states_batched = jax.tree.map(lambda *leaves: jnp.stack(leaves), *valid_states)

        self._key, subkey = jax.random.split(self._key)
        keys_array = jax.random.split(subkey, B * A * N).reshape(B, A, N, 2)

        returns = self._compiled_rollout_fn(self.params, states_batched, self._all_actions, keys_array)
        returns = jax.block_until_ready(returns)
        returns_np = np.array(returns)  # (B, A, N)

        scores = np.zeros(B)
        for i in range(B):
            taken_action = valid_actions_taken[i]
            return_distributions = {(a,): returns_np[i, a] for a in range(A)}
            scores[i] = compute_consequence_metric(
                action=(taken_action,),
                return_distributions=return_distributions,
                metric=self.consequence_metric,
                aggregation=self.consequence_aggregation,
            )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        scored_indices = indices[np.array(valid_indices)]
        self.buffer.update_consequence_scores(scored_indices, scores)

    # ------------------------------------------------------------------
    # Training step (score periodically, then a normal Q-update)
    # ------------------------------------------------------------------

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1
        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        transitions, indices, weights = self.buffer.sample(self.batch_size)
        states  = jnp.asarray(np.stack([t['s']  for t in transitions]), dtype=jnp.float32)
        actions = jnp.asarray([t['a']    for t in transitions], dtype=jnp.int32)
        rewards = jnp.asarray([t['r']    for t in transitions], dtype=jnp.float32)
        nexts   = jnp.asarray(np.stack([t["s'"] for t in transitions]), dtype=jnp.float32)
        dones   = jnp.asarray([t['done'] for t in transitions], dtype=jnp.float32)
        wts     = jnp.asarray(weights, dtype=jnp.float32)

        self.params, self.opt_state, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, nexts, dones, wts,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

    # ------------------------------------------------------------------
    # Sequential learn (the vectorized subclass is the workhorse)
    # ------------------------------------------------------------------

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'JaxNavConsequenceDQN':
        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 300)
        eval_episodes = self.config.get('eval_episodes', 100)
        save_every = self.config.get('save_every', 1000)

        self.metrics_logger = _MetricsLogger(
            config=self.config, n_episodes=n_episodes,
            eval_interval=eval_interval, eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_success = -1.0

        np.random.seed(self.config.get('seed', 0))

        if verbose:
            print(f"Training Consequence-DQN on JaxNav [{self.config.get('scenario') or self.config.get('map_id')}]")
            print(f"  Priority mixing: {self.config.get('priority_mixing')}  |  mu: {self.config.get('mu')}")
            print(f"  Metric: {self.consequence_metric}  |  Score interval: {self.score_interval}")
            print(f"  CF rollouts: {self.cf_n_rollouts}  |  horizon: {self.cf_horizon}  |  temp: {self.cf_rollout_temperature}")
            print(f"  JAX backend: {jax.default_backend()}  |  Run dir: {self.metrics_logger.dir}")

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            self._current_episode = episode
            self._key, rk = jax.random.split(self._key)
            obs, state = self.env.reset(rk)
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < self.env.max_steps:
                saved_state = jax.tree.map(np.array, state)   # pre-step state for rollouts
                action = self.select_action(obs)
                self._key, sk = jax.random.split(self._key)
                next_obs, next_state, reward, done, _ = self.env.step(sk, state, jnp.int32(action))
                reward = float(reward)
                done = bool(done)
                self.buffer.add(
                    {'s': np.asarray(obs, np.float32), 'a': int(action),
                     'r': reward, "s'": np.asarray(next_obs, np.float32), 'done': done},
                    jax_state=saved_state,
                )
                self.total_steps += 1
                if self.total_steps % self.n_steps_per_update == 0:
                    self._update()
                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()
                state, obs = next_state, next_obs
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
                if metrics['win_rate'] > best_success:
                    best_success = metrics['win_rate']
                    self.save(best_path)

        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
