"""
Consequence-weighted DQN (Algorithm 2) for routing (CVRP / TSP).

Variants selected via config:
  algorithm='consequence-dqn', mu=1.0                           -> DQN + CCE-only
  algorithm='consequence-dqn', priority_mixing='additive'       -> CCE + TD additive (Eq. 4)
  algorithm='consequence-dqn', priority_mixing='multiplicative' -> CCE + TD multiplicative (Eq. 5)

Adapted from agents/frozen_lake/consequence_dqn.py. Routing needs ACTION MASKING in two
extra places beyond the base trainer's three:

  4. the rollout policy inside the counterfactual scan — an unmasked argmax would drive
     the rollout into already-served stops, corrupting every return distribution;
  5. the counterfactual ACTION SET — rollouts are computed for all n_actions to keep the
     vmap shape static, but illegal ones self-loop with a penalty and are dropped before
     the divergence. Including them would make every transition look consequential purely
     because of the illegal-action penalty.

NOTE on determinism: with travel_noise = 0 the env is deterministic, so greedy rollouts
give point-mass return distributions and the total-variation score collapses to a coarse
0/1 (the known degeneracy — see envs/cvrp.py). Claim 2 runs deterministically anyway
because that is where CCE's replay advantage lives, but be aware the CCE signal is a
binary "does this action change the outcome at all" there rather than a graded score.
"""

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .dqn import CVRPDQN, MASK_FILL
from ..shared.consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.analysis.metrics import compute_consequence_metric


class CVRPConsequenceDQN(CVRPDQN):
    """
    Consequence-weighted DQN for routing (Algorithm 2).

    Stores the integer state at each transition; periodically runs vmapped rollouts for
    every action from each scored state and uses the resulting return distributions to
    weight buffer sampling priorities.
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
        self.consequence_metric = self.config.get('consequence_metric', 'total_variation')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')
        self.cf_horizon = self.config.get('cf_horizon', 30)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 20)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)

        self.q_update_count = 0
        self._compiled_rollout_fn = None
        self._all_actions = jnp.arange(self.n_actions, dtype=jnp.int32)

        self.log_sampling = self.config.get('log_sampling', False)
        self.sampling_snapshot_interval = self.config.get('sampling_snapshot_interval', 2000)
        if self.log_sampling:
            self.buffer.enable_draw_log = True

    def _build_rollout_fn(self):
        """
        Triple-vmapped JIT rollout function.

        Axes:
          vmap over B transition states -> (B, A, N)
          vmap over A actions           -> (A, N)
          vmap over N rollout keys      -> (N,)
          lax.scan over H horizon steps -> scalar return
        """
        env = self.env
        network = self.network
        horizon = self.cf_horizon
        gamma = self.cf_gamma
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
            return final[2]  # cumulative discounted return

        over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, 0))
        over_actions = jax.vmap(over_rollouts, in_axes=(None, None, 0, 0))
        over_states = jax.vmap(over_actions, in_axes=(None, 0, None, 0))
        self._compiled_rollout_fn = jax.jit(over_states)

    def _score_buffer_transitions(self):
        """Algorithm 2: score a uniform sample of buffer transitions."""
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        timer = self.metrics_logger.timer
        ep = self._current_episode

        with timer('update.scoring.sample', episode=ep):
            transitions, indices = self.buffer.sample_uniform(n_score)
            valid_states, valid_actions_taken, valid_indices = [], [], []
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
        A = self.n_actions

        if self._compiled_rollout_fn is None:
            print("Compiling consequence rollout function (one-time cost)...")
            self._build_rollout_fn()

        states_array = jnp.array(valid_states, dtype=jnp.int32)
        self._key, subkey = jax.random.split(self._key)
        keys_array = jax.random.split(subkey, B * A * N).reshape(B, A, N, 2)

        with timer('update.scoring.rollouts', episode=ep, batch_size=B):
            returns = self._compiled_rollout_fn(
                self.params, states_array, self._all_actions, keys_array
            )
            returns = jax.block_until_ready(returns)
            returns_np = np.array(returns)  # (B, A, N)

        with timer('update.scoring.metrics', episode=ep):
            scores = np.zeros(B)
            for i in range(B):
                s = int(valid_states[i])
                legal = np.flatnonzero(self._masks_np[s])
                if legal.size < 2:
                    continue  # forced move: no alternative, so no consequence
                taken = valid_actions_taken[i]
                if taken not in legal:
                    continue  # defensive: a masked action should never have been stored
                scores[i] = compute_consequence_metric(
                    action=(taken,),
                    return_distributions={(int(a),): returns_np[i, a] for a in legal},
                    metric=self.consequence_metric,
                    aggregation=self.consequence_aggregation,
                )
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        with timer('update.scoring.buffer_update', episode=ep):
            scored_indices = indices[np.array(valid_indices)]
            self.buffer.update_consequence_scores(scored_indices, scores)

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1

        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        ep = self._current_episode
        timer = self.metrics_logger.timer
        with timer('update.q_update', episode=ep):
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

        if (self.log_sampling and self.sampling_snapshot_interval > 0
                and self.q_update_count % self.sampling_snapshot_interval == 0):
            self.buffer.snapshot_draws(self.q_update_count, self._current_episode)

    def learn(self, n_episodes: Optional[int] = None,
              verbose: bool = True) -> 'CVRPConsequenceDQN':
        """
        Same loop as CVRPDQN.learn, except each transition also stores its integer state
        so the counterfactual rollouts can restart from it.
        """
        import os

        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 100)
        eval_episodes = self.config.get('eval_episodes', 20)
        save_every = self.config.get('save_every', 500)

        from .dqn import _MetricsLogger
        self.metrics_logger = _MetricsLogger(
            config=self.config, env=self.env, n_episodes=n_episodes,
            eval_interval=eval_interval, eval_episodes=eval_episodes,
            optimal_length=self.optimal_length,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_ratio = -1.0
        early_stop = self.config.get('early_stop_opt_ratio', None)

        n_ckpts = self.config.get('n_checkpoints', 100)
        ckpt_interval = max(1, n_episodes // n_ckpts) if n_ckpts > 0 else 0
        ckpt_dir = os.path.join(self.metrics_logger.dir, 'checkpoints')
        if ckpt_interval > 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        np.random.seed(self.config.get('seed', 0))

        if verbose:
            mixing = self.config.get('priority_mixing', 'additive')
            mu = self.config.get('mu', 0.5)
            mode = f"CVRP cap={self.env.capacity}" if self.env.is_capacitated else "TSP"
            print(f"Training Consequence-DQN on {mode} ({self.env.n_customers} customers)")
            print(f"  States: {self.n_states}  |  optimal length: {self.optimal_length:.4f}")
            print(f"  Priority mixing: {mixing}  |  mu: {mu}")
            print(f"  Metric: {self.consequence_metric}  |  Score interval: {self.score_interval}")
            print(f"  CF rollouts: {self.cf_n_rollouts}  |  CF horizon: {self.cf_horizon}")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            self._current_episode = episode
            timer.begin_episode(episode)

            with timer('env', episode=episode):
                self._key, rk = jax.random.split(self._key)
                _, state = self.env.reset(rk)
                state = int(state)

            done = False
            ep_return = 0.0
            ep_steps = 0

            while not done and ep_steps < 200:
                saved_state = state  # restart point for counterfactual rollouts

                with timer('action', episode=episode):
                    action = self.select_action(state)

                with timer('env', episode=episode):
                    self._key, sk = jax.random.split(self._key)
                    _, next_state, reward, done, _ = self.env.step(
                        sk, jnp.int32(state), jnp.int32(action)
                    )
                    next_state = int(next_state)
                    reward = float(reward)
                    done = bool(done)

                with timer('buffer.add', episode=episode):
                    self.buffer.add(
                        {'s': state, 'a': action, 'r': reward,
                         "s'": next_state, 'done': done},
                        jax_state=saved_state,
                    )
                self.total_steps += 1

                with timer('update', episode=episode):
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

            if ckpt_interval > 0 and (episode + 1) % ckpt_interval == 0:
                self.save(os.path.join(ckpt_dir, f'ckpt_{episode+1:07d}.pkl'))

            if eval_interval and (episode + 1) % eval_interval == 0:
                with timer('eval', episode=episode):
                    metrics = self.evaluate(eval_episodes)
                q_updates = self.total_steps // self.n_steps_per_update
                self.metrics_logger.log_eval(episode + 1, q_updates, self.epsilon, metrics)
                if metrics['opt_ratio'] > best_ratio:
                    best_ratio = metrics['opt_ratio']
                    self.save(best_path)
                    if verbose:
                        print(f"\n  New best: {best_ratio:.4f} of optimal at ep {episode + 1}")
                if early_stop is not None and metrics['opt_ratio'] >= early_stop:
                    if ckpt_interval > 0:
                        self.save(os.path.join(ckpt_dir, f'ckpt_{episode+1:07d}.pkl'))
                    if verbose:
                        print(f"\n  Early stop: {metrics['opt_ratio']:.4f} >= {early_stop}")
                    break

            timer.flush_episode()

        self.save(last_path)
        if self.log_sampling:
            self.buffer.dump_sampling(
                os.path.join(self.metrics_logger.dir, 'sampling.npz'),
                self.n_states, self.env.n_actions,
            )
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
