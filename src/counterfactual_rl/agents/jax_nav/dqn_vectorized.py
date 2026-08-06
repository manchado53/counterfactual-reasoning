"""
Vectorized DQN for JaxNav — lax.scan + vmap over n_envs parallel environments.

Mirror of ``agents/frozen_lake/dqn_vectorized.py``. Inherits the Q-network,
``_update``, save/load from ``JaxNavDQN``; replaces ``learn`` with a chunk-based
collection loop and overrides ``evaluate`` with a vectorized rollout.

Differences from the FrozenLake version, both because JaxNav state is a pytree
(not an int) and obs != state:
  * the scan carry holds BOTH the ``State`` pytree (to step) and the obs vector
    (to feed the network); the auto-reset ``where`` becomes a ``jax.tree.map``
    over the state pytree;
  * transitions stored in the buffer are the obs vectors (the pytree state is
    only needed by CCE).
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .dqn import JaxNavDQN, _MetricsLogger


def _tree_where(cond, a, b):
    """Select pytree a where cond else b (cond broadcasts over each leaf)."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


class JaxNavDQNVectorized(JaxNavDQN):
    """DQN (uniform or PER) with vectorized episode collection over n_envs."""

    def _build_collect_fn(self):
        env = self.env
        network = self.network
        n_actions = self.n_actions

        def _collect_single_env(params, epsilon, init_state, step_keys):
            def _step(carry, step_key):
                state, obs, already_done, cum_return, ep_length = carry
                k_reset, k_rand_act, k_explore, k_env = jax.random.split(step_key, 4)

                # Auto-reset when the previous step ended the episode
                new_obs, new_state = env.reset(k_reset)
                state = _tree_where(already_done, new_state, state)
                obs = jnp.where(already_done, new_obs, obs)
                cum_return = jnp.where(already_done, jnp.float32(0.0), cum_return)
                ep_length = jnp.where(already_done, jnp.int32(0), ep_length)

                # ε-greedy action selection (on the observation)
                q = network.apply(params, obs)
                greedy_act = jnp.argmax(q)
                random_act = jax.random.randint(k_rand_act, (), 0, n_actions)
                action = jnp.where(jax.random.uniform(k_explore) < epsilon, random_act, greedy_act)

                # Environment step
                next_obs, next_state, reward, done, _ = env.step(k_env, state, action)
                new_cum = cum_return + reward
                new_len = ep_length + jnp.int32(1)

                ep_return_out = jnp.where(done, new_cum, jnp.float32(0.0))
                ep_length_out = jnp.where(done, new_len, jnp.int32(0))

                return (next_state, next_obs, done, new_cum, new_len), (
                    obs, action, reward, next_obs, done,
                    ep_return_out, ep_length_out, done,
                )

            init_obs = env.get_obs(init_state)
            init_carry = (
                init_state, init_obs, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0),
            )
            (final_state, _, _, _, _), outputs = jax.lax.scan(_step, init_carry, step_keys)
            return final_state, outputs

        self._collect_fn = jax.jit(
            jax.vmap(_collect_single_env, in_axes=(None, None, 0, 0))
        )

    def _add_chunk_to_buffer(self, outputs):
        obs_np      = np.array(outputs[0])   # (n_envs, collect_steps, obs_dim)
        actions_np  = np.array(outputs[1])
        rewards_np  = np.array(outputs[2])
        next_obs_np = np.array(outputs[3])
        dones_np    = np.array(outputs[4])
        n = obs_np.shape[0] * obs_np.shape[1]
        self.buffer.add_batch({
            's':    obs_np.reshape(n, self.obs_dim),
            'a':    actions_np.reshape(n),
            'r':    rewards_np.reshape(n).astype(np.float32),
            "s'":   next_obs_np.reshape(n, self.obs_dim),
            'done': dones_np.reshape(n),
        })

    def _build_eval_fn(self):
        env = self.env
        network = self.network

        def _eval_single_env(params, init_state, step_keys):
            def _step(carry, step_key):
                state, obs, already_done, cum_return, ep_steps = carry
                action = jnp.argmax(network.apply(params, obs))
                next_obs, next_state, reward, done, _ = env.step(step_key, state, action)
                new_done = jnp.logical_or(already_done, done)
                new_steps = ep_steps + jnp.where(already_done, jnp.int32(0), jnp.int32(1))
                new_cum   = cum_return + jnp.where(already_done, jnp.float32(0.0), reward)
                return (next_state, next_obs, new_done, new_cum, new_steps), None

            init_obs = env.get_obs(init_state)
            init = (init_state, init_obs, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0))
            (final_state, _, _, final_return, final_steps), _ = jax.lax.scan(_step, init, step_keys)
            return final_return, final_steps, final_state.goal_reached[0]

        self._eval_fn = jax.jit(jax.vmap(_eval_single_env, in_axes=(None, 0, 0)))

    def evaluate(self, n_episodes: int = 100) -> dict:
        if not hasattr(self, '_eval_fn'):
            self._build_eval_fn()
        H = self.env.max_steps
        self._key, eval_key = jax.random.split(self._key)
        all_keys  = jax.random.split(eval_key, n_episodes * (H + 1))
        init_keys = all_keys[:n_episodes]
        step_keys = all_keys[n_episodes:].reshape(n_episodes, H, 2)

        init_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)
        returns, lengths, goals = self._eval_fn(self.params, init_states, step_keys)
        jax.block_until_ready(returns)

        returns_np = np.array(returns)
        lengths_np = np.array(lengths)
        goals_np = np.array(goals)
        return {
            # win_rate == goal-reach rate (from the robot's goal_reached flag), so it is
            # independent of reward shaping; identical to mean(return>0) under sparse reward.
            'win_rate':   float(np.mean(goals_np)),
            'avg_length': float(np.mean(lengths_np.astype(float))),
            'avg_return': float(np.mean(returns_np)),
        }

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'JaxNavDQNVectorized':
        n_episodes    = n_episodes or self.config['n_episodes']
        n_envs        = self.config.get('n_envs', 128)
        collect_steps = self.config.get('collect_steps', 32)
        eval_interval = self.config.get('eval_interval', 300)
        eval_episodes = self.config.get('eval_episodes', 100)
        save_every    = self.config.get('save_every', 1000)

        self.metrics_logger = _MetricsLogger(
            config=self.config, n_episodes=n_episodes,
            eval_interval=eval_interval, eval_episodes=eval_episodes,
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

        self._key, init_key = jax.random.split(self._key)
        init_keys = jax.random.split(init_key, n_envs)
        current_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)

        if verbose:
            alg = self.config['algorithm'].upper()
            print(f"Training {alg} on JaxNav [{self.config.get('scenario') or self.config.get('map_id')}] [VECTORIZED]")
            print(f"  n_envs={n_envs}  collect_steps={collect_steps}  "
                  f"({n_envs * collect_steps} transitions/chunk)")
            print(f"  obs_dim={self.obs_dim}  n_actions={self.n_actions}")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        total_episodes = len(self.episode_returns)
        prev_eval_ep = prev_save_ep = prev_ckpt_ep = 0
        n_chunk_steps = n_envs * collect_steps

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
            jax.block_until_ready(outputs[4])

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

            if ckpt_interval > 0 and total_episodes // ckpt_interval > prev_ckpt_ep // ckpt_interval:
                self.save(os.path.join(ckpt_dir, f'ckpt_{total_episodes:07d}.pkl'))
                prev_ckpt_ep = total_episodes

            if eval_interval and total_episodes // eval_interval > prev_eval_ep // eval_interval:
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
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
