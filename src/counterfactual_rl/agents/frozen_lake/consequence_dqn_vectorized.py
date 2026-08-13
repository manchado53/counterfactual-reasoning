"""
Consequence-weighted DQN (Algorithm 2) with vectorized episode collection.

Inherits CCE scoring, rollout function, and buffer from FrozenLakeConsequenceDQN.
Only learn() is replaced with the chunk-based vectorized loop from dqn_vectorized.py,
extended to store jax_state per transition for counterfactual rollouts.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .consequence_dqn import FrozenLakeConsequenceDQN
from .dqn import _MetricsLogger
from .dqn_vectorized import FrozenLakeDQNVectorized


class FrozenLakeConsequenceDQNVectorized(FrozenLakeConsequenceDQN):
    """
    CCE-DQN with vectorized collection.

    Builds the same lax.scan + vmap collect function as FrozenLakeDQNVectorized
    but overrides _add_chunk_to_buffer to also store jax_states for CCE rollouts.
    """

    def _build_collect_fn(self):
        # Delegate to the vectorized DQN implementation
        FrozenLakeDQNVectorized._build_collect_fn(self)

    def _build_eval_fn(self):
        FrozenLakeDQNVectorized._build_eval_fn(self)

    def evaluate(self, n_episodes: int = 100) -> dict:
        return FrozenLakeDQNVectorized.evaluate(self, n_episodes)

    def _add_chunk_to_buffer(self, outputs):
        """Add chunk transitions to ConsequenceReplayBuffer with jax_states."""
        states_np      = np.array(outputs[0])  # (n_envs, collect_steps) — state BEFORE step
        actions_np     = np.array(outputs[1])
        rewards_np     = np.array(outputs[2])
        next_states_np = np.array(outputs[3])
        dones_np       = np.array(outputs[4])
        n = states_np.size
        self.buffer.add_batch(
            {
                's':    states_np.reshape(n),
                'a':    actions_np.reshape(n),
                'r':    rewards_np.reshape(n).astype(np.float32),
                "s'":   next_states_np.reshape(n),
                'done': dones_np.reshape(n),
            },
            jax_states=states_np.reshape(n),  # state before step = starting state for rollouts
        )

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'FrozenLakeConsequenceDQNVectorized':
        n_episodes    = n_episodes or self.config['n_episodes']
        n_envs        = self.config.get('n_envs', 256)
        collect_steps = self.config.get('collect_steps', 128)
        eval_interval = self.config.get('eval_interval', 300)
        eval_episodes = self.config.get('eval_episodes', 100)
        save_every    = self.config.get('save_every', 500)

        self.metrics_logger = _MetricsLogger(
            config=self.config,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_success = -1.0
        early_stop_win_rate = self.config.get('early_stop_win_rate', None)

        n_ckpts = self.config.get('n_checkpoints', 100)
        ckpt_interval = max(1, n_episodes // n_ckpts) if n_ckpts > 0 else 0
        ckpt_dir = os.path.join(self.metrics_logger.dir, 'checkpoints')
        if ckpt_interval > 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        np.random.seed(self.config.get('seed', 0))
        self._build_collect_fn()

        # Initialize n_envs environments
        self._key, init_key = jax.random.split(self._key)
        init_keys = jax.random.split(init_key, n_envs)
        current_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)

        if verbose:
            mixing = self.config.get('priority_mixing', 'additive')
            mu = self.config.get('mu', 0.25)
            print(f"Training Consequence-DQN on FrozenLake-{self.config['map_name']} [VECTORIZED]")
            print(f"  n_envs={n_envs}  collect_steps={collect_steps}  "
                  f"({n_envs * collect_steps} transitions/chunk)")
            print(f"  Priority mixing: {mixing}  |  mu: {mu}")
            print(f"  Metric: {self.consequence_metric}  |  Score interval: {self.score_interval}")
            print(f"  CF rollouts: {self.cf_n_rollouts}  |  CF horizon: {self.cf_horizon}")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        total_episodes = len(self.episode_returns)
        prev_eval_ep   = 0
        prev_save_ep   = 0
        prev_ckpt_ep   = 0
        n_chunk_steps  = n_envs * collect_steps

        pbar = tqdm(total=n_episodes, disable=not verbose)
        pbar.update(total_episodes)

        while total_episodes < n_episodes:
            self._key, chunk_key = jax.random.split(self._key)
            step_keys = jax.random.split(chunk_key, n_envs * collect_steps).reshape(
                n_envs, collect_steps, 2
            )

            current_states, outputs = self._collect_fn(
                self.params, jnp.float32(self.epsilon), current_states, step_keys
            )
            jax.block_until_ready(current_states)

            self._add_chunk_to_buffer(outputs)

            ep_returns_np = np.array(outputs[5])
            ep_lengths_np = np.array(outputs[6])
            ep_ended_np   = np.array(outputs[7]).astype(bool)
            ended_flat    = ep_ended_np.reshape(-1)
            completed_returns = ep_returns_np.reshape(-1)[ended_flat]
            completed_lengths = ep_lengths_np.reshape(-1)[ended_flat]
            self.episode_returns.extend(completed_returns.tolist())
            self.episode_lengths.extend(completed_lengths.tolist())
            n_new = len(completed_returns)
            total_episodes = len(self.episode_returns)
            self._current_episode = total_episodes

            prev_steps = self.total_steps
            self.total_steps += n_chunk_steps
            n_q_updates = (self.total_steps // self.n_steps_per_update) - \
                          (prev_steps // self.n_steps_per_update)
            q_start = prev_steps // self.n_steps_per_update
            target_freq_q = max(1, self.target_update_freq // self.n_steps_per_update)
            for i in range(n_q_updates):
                self._update()
                if (q_start + i) % target_freq_q == 0:
                    self._update_target_network()

            decay = min(1.0, total_episodes / max(1, self.epsilon_decay_episodes))
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay

            pbar.update(n_new)
            if len(self.episode_returns) >= 100:
                avg = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"AvgR(100): {avg:.3f}  ε: {self.epsilon:.3f}")

            if total_episodes // save_every > prev_save_ep // save_every:
                self.save(last_path)
                prev_save_ep = total_episodes

            if ckpt_interval > 0 and \
               total_episodes // ckpt_interval > prev_ckpt_ep // ckpt_interval:
                self.save(os.path.join(ckpt_dir, f'ckpt_{total_episodes:07d}.pkl'))
                prev_ckpt_ep = total_episodes

            if eval_interval and \
               total_episodes // eval_interval > prev_eval_ep // eval_interval:
                with timer('eval', episode=total_episodes):
                    metrics = self.evaluate(eval_episodes)
                self.metrics_logger.log_eval(
                    total_episodes, self.q_update_count, self.epsilon, metrics
                )
                if metrics['win_rate'] > best_success:
                    best_success = metrics['win_rate']
                    self.save(best_path)
                    if verbose:
                        print(f"\n  New best: {best_success:.1%} at ep {total_episodes}")
                if early_stop_win_rate and metrics['win_rate'] >= early_stop_win_rate:
                    if verbose:
                        print(f"\n  Early stop: {metrics['win_rate']:.1%} >= {early_stop_win_rate:.1%}")
                    break
                prev_eval_ep = total_episodes

        pbar.close()
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
