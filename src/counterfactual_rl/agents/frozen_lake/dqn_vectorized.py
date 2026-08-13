"""
Vectorized DQN for FrozenLake using lax.scan + vmap over n_envs parallel environments.

~10x faster than the sequential dqn.py. Inherits Q-network, _update, evaluate, and
save/load unchanged. Only learn() is replaced with a chunk-based collection loop.

Logging format is identical to dqn.py so parse_logs.py works without changes.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .dqn import FrozenLakeDQN, _MetricsLogger


class FrozenLakeDQNVectorized(FrozenLakeDQN):
    """
    DQN (uniform or PER) with vectorized episode collection.

    collect_steps steps are scanned over n_envs environments in parallel each chunk.
    Auto-reset: when an env reaches a terminal state, it resets immediately and continues
    collecting from the next episode without Python overhead.
    """

    def _build_collect_fn(self):
        env = self.env
        network = self.network
        n_actions = self.n_actions

        def _collect_single_env(params, epsilon, init_state, step_keys):
            """Run one environment for collect_steps steps via lax.scan."""

            def _step(carry, step_key):
                state, already_done, cum_return, ep_length = carry
                k_reset, k_rand_act, k_explore, k_env = jax.random.split(step_key, 4)

                # Auto-reset when previous step ended the episode
                _, new_state = env.reset(k_reset)
                state = jnp.where(already_done, new_state, state)
                cum_return = jnp.where(already_done, jnp.float32(0.0), cum_return)
                ep_length = jnp.where(already_done, jnp.int32(0), ep_length)

                # ε-greedy action selection
                q = network.apply(params, state)
                greedy_act = jnp.argmax(q)
                random_act = jax.random.randint(k_rand_act, (), 0, n_actions)
                action = jnp.where(jax.random.uniform(k_explore) < epsilon, random_act, greedy_act)

                # Environment step
                _, next_state, reward, done, _ = env.step(k_env, state, action)
                new_cum = cum_return + reward
                new_len = ep_length + jnp.int32(1)

                # Emit episode stats only at episode boundary
                ep_return_out = jnp.where(done, new_cum, jnp.float32(0.0))
                ep_length_out = jnp.where(done, new_len, jnp.int32(0))
                ep_ended_out = done

                return (next_state, done, new_cum, new_len), (
                    state, action, reward, next_state, done,
                    ep_return_out, ep_length_out, ep_ended_out,
                )

            init_carry = (
                jnp.int32(init_state),
                jnp.bool_(False),
                jnp.float32(0.0),
                jnp.int32(0),
            )
            (final_state, _, _, _), outputs = jax.lax.scan(_step, init_carry, step_keys)
            return final_state, outputs

        self._collect_fn = jax.jit(
            jax.vmap(_collect_single_env, in_axes=(None, None, 0, 0))
        )

    def _add_chunk_to_buffer(self, outputs):
        """Add all transitions from a chunk to the replay buffer."""
        states_np      = np.array(outputs[0])  # (n_envs, collect_steps)
        actions_np     = np.array(outputs[1])
        rewards_np     = np.array(outputs[2])
        next_states_np = np.array(outputs[3])
        dones_np       = np.array(outputs[4])
        n = states_np.size
        self.buffer.add_batch({
            's':    states_np.reshape(n),
            'a':    actions_np.reshape(n),
            'r':    rewards_np.reshape(n).astype(np.float32),
            "s'":   next_states_np.reshape(n),
            'done': dones_np.reshape(n),
        })

    def _build_eval_fn(self):
        env = self.env
        network = self.network

        def _eval_single_env(params, init_state, step_keys):
            def _step(carry, step_key):
                state, already_done, cum_return, ep_steps = carry
                action = jnp.argmax(network.apply(params, state))
                _, next_state, reward, done, _ = env.step(step_key, state, action)
                new_done = jnp.logical_or(already_done, done)
                new_steps = ep_steps + jnp.where(already_done, jnp.int32(0), jnp.int32(1))
                new_cum   = cum_return + jnp.where(already_done, jnp.float32(0.0), reward)
                return (next_state, new_done, new_cum, new_steps), None

            init = (jnp.int32(init_state), jnp.bool_(False), jnp.float32(0.0), jnp.int32(0))
            (_, _, final_return, final_steps), _ = jax.lax.scan(_step, init, step_keys)
            return final_return, final_steps

        self._eval_fn = jax.jit(jax.vmap(_eval_single_env, in_axes=(None, 0, 0)))

    def evaluate(self, n_episodes: int = 100) -> dict:
        if not hasattr(self, '_eval_fn'):
            self._build_eval_fn()

        self._key, eval_key = jax.random.split(self._key)
        all_keys  = jax.random.split(eval_key, n_episodes * 201)
        init_keys = all_keys[:n_episodes]
        step_keys = all_keys[n_episodes:].reshape(n_episodes, 200, 2)

        init_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)
        returns, lengths = self._eval_fn(self.params, init_states, step_keys)
        jax.block_until_ready(returns)

        returns_np = np.array(returns)
        lengths_np = np.array(lengths)
        return {
            'win_rate':   float(np.mean(returns_np > 0)),
            'avg_length': float(np.mean(lengths_np.astype(float))),
            'avg_return': float(np.mean(returns_np)),
        }

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'FrozenLakeDQNVectorized':
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
            alg = self.config['algorithm'].upper()
            print(f"Training {alg} on FrozenLake-{self.config['map_name']} [VECTORIZED]")
            print(f"  n_envs={n_envs}  collect_steps={collect_steps}  "
                  f"({n_envs * collect_steps} transitions/chunk)")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        total_episodes   = len(self.episode_returns)
        prev_eval_ep     = 0
        prev_save_ep     = 0
        prev_ckpt_ep     = 0
        n_chunk_steps    = n_envs * collect_steps

        pbar = tqdm(total=n_episodes, disable=not verbose)
        pbar.update(total_episodes)

        while total_episodes < n_episodes:
            # Generate per-step keys for all envs in this chunk
            self._key, chunk_key = jax.random.split(self._key)
            step_keys = jax.random.split(chunk_key, n_envs * collect_steps).reshape(
                n_envs, collect_steps, 2
            )

            # Collect
            current_states, outputs = self._collect_fn(
                self.params, jnp.float32(self.epsilon), current_states, step_keys
            )
            jax.block_until_ready(current_states)

            # Add transitions to buffer
            self._add_chunk_to_buffer(outputs)

            # Count completed episodes
            ep_returns_np = np.array(outputs[5])  # (n_envs, collect_steps)
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

            # Steps and Q-updates
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

            # Epsilon decay (per completed episode, matches sequential behaviour)
            decay = min(1.0, total_episodes / max(1, self.epsilon_decay_episodes))
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay

            pbar.update(n_new)
            if len(self.episode_returns) >= 100:
                avg = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"AvgR(100): {avg:.3f}  ε: {self.epsilon:.3f}")

            # Periodic save
            if total_episodes // save_every > prev_save_ep // save_every:
                self.save(last_path)
                prev_save_ep = total_episodes

            # Checkpoint
            if ckpt_interval > 0 and \
               total_episodes // ckpt_interval > prev_ckpt_ep // ckpt_interval:
                self.save(os.path.join(ckpt_dir, f'ckpt_{total_episodes:07d}.pkl'))
                prev_ckpt_ep = total_episodes

            # Eval
            if eval_interval and \
               total_episodes // eval_interval > prev_eval_ep // eval_interval:
                with timer('eval', episode=total_episodes):
                    metrics = self.evaluate(eval_episodes)
                q_updates = self.total_steps // self.n_steps_per_update
                self.metrics_logger.log_eval(total_episodes, q_updates, self.epsilon, metrics)
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
