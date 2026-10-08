"""
Vectorized DQN for the neural CVRP -- lax.scan + vmap over n_envs parallel instances.

Mirror of ``agents/jax_nav/dqn_vectorized.py``. Inherits the network, ``_update``
and save/load from ``CVRPNeuralDQN``; replaces ``learn`` with the chunk-based
collection loop and overrides ``evaluate`` with a vectorized greedy decode.

Two CVRP-specific changes:
  * exploration is Boltzmann over the MASKED Q-values (RP-DQN's shipped default),
    not epsilon-greedy over all actions. Sampling an illegal action would be a
    wasted buffer slot and the env would have to special-case it.
  * the primary eval metric is mean tour length, reported as a quality ratio so
    the shared claim-2 analysis has something that increases.
"""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .dqn import CVRPNeuralDQN, _MetricsLogger, MASK_VALUE
from counterfactual_rl.envs.cvrp_neural import CVRPNeuralEnv


def _tree_where(cond, a, b):
    """Select pytree a where cond else b (cond broadcasts over each leaf)."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def _explore_action(key, q, temp, mode):
    """One exploratory action from masked Q-values.

    Both branches sample over LEGAL actions only: illegal entries sit at
    MASK_VALUE, which is -1e9, so their probability underflows to exactly zero.
    """
    if mode == 'boltzmann':
        return jax.random.categorical(key, q / jnp.maximum(temp, 1e-6))
    # epsilon-greedy, with the random draw restricted to the legal set
    k_pick, k_coin = jax.random.split(key)
    legal_logits = jnp.where(q > MASK_VALUE / 2, 0.0, MASK_VALUE)
    random_act = jax.random.categorical(k_pick, legal_logits)
    return jnp.where(jax.random.uniform(k_coin) < temp, random_act, jnp.argmax(q))


class CVRPNeuralDQNVectorized(CVRPNeuralDQN):
    """DQN (uniform or PER) with vectorized collection over n_envs instances."""

    def _build_collect_fn(self):
        env = self.env
        network = self.network
        mode = self.exploration

        def _collect_single_env(params, temp, init_state, step_keys):
            def _step(carry, step_key):
                state, obs, already_done, cum_return, ep_length = carry
                k_reset, k_act, k_env = jax.random.split(step_key, 3)

                # Auto-reset when the previous step ended the episode. A fresh
                # instance is drawn -- that is the benchmark's own protocol.
                new_obs, new_state = env.reset(k_reset)
                state = _tree_where(already_done, new_state, state)
                obs = jnp.where(already_done, new_obs, obs)
                cum_return = jnp.where(already_done, jnp.float32(0.0), cum_return)
                ep_length = jnp.where(already_done, jnp.int32(0), ep_length)

                q = network.apply(params, obs)
                action = _explore_action(k_act, q, temp, mode)

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
        obs_np      = np.array(outputs[0])   # (n_envs, collect_steps, n_nodes, F)
        actions_np  = np.array(outputs[1])
        rewards_np  = np.array(outputs[2])
        next_obs_np = np.array(outputs[3])
        dones_np    = np.array(outputs[4])
        n = obs_np.shape[0] * obs_np.shape[1]
        self.buffer.add_batch({
            's':    obs_np.reshape(n, *self.obs_shape),
            'a':    actions_np.reshape(n),
            'r':    rewards_np.reshape(n).astype(np.float32),
            "s'":   next_obs_np.reshape(n, *self.obs_shape),
            'done': dones_np.reshape(n),
        })

    # ------------------------------------------------------------------

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
            (final_state, _, final_done, final_return, final_steps), _ = jax.lax.scan(
                _step, init, step_keys
            )
            # solved == every customer served AND parked back at the depot
            solved = jnp.all(final_state.visited[1:]) & (final_state.current == 0)
            return final_return, final_steps, solved

        self._eval_fn = jax.jit(jax.vmap(_eval_single_env, in_axes=(None, 0, 0)))

    def evaluate(self, n_episodes: int = 512, init_states=None) -> dict:
        """Greedy decode. ``init_states`` lets Gate 1 evaluate on a FIXED test set."""
        if not hasattr(self, '_eval_fn'):
            self._build_eval_fn()
        H = self.env.max_steps
        self._key, eval_key = jax.random.split(self._key)

        if init_states is None:
            all_keys  = jax.random.split(eval_key, n_episodes * (H + 1))
            init_keys = all_keys[:n_episodes]
            step_keys = all_keys[n_episodes:].reshape(n_episodes, H, 2)
            init_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)
        else:
            n_episodes = jax.tree.leaves(init_states)[0].shape[0]
            step_keys = jax.random.split(eval_key, n_episodes * H).reshape(n_episodes, H, 2)

        returns, lengths, solved = self._eval_fn(self.params, init_states, step_keys)
        jax.block_until_ready(returns)

        returns_np = np.array(returns)
        lengths_np = np.array(lengths)
        solved_np = np.array(solved)
        avg_return = float(np.mean(returns_np))
        return {
            # Quality ratio, not a win rate -- see _MetricsLogger. Kept under this key
            # so analysis/claim2 parses it without a special case.
            'win_rate':    self._quality(-avg_return),
            'avg_length':  float(np.mean(lengths_np.astype(float))),
            'avg_return':  avg_return,
            'mean_tour':   -avg_return,
            'solved_frac': float(np.mean(solved_np)),
        }

    # ------------------------------------------------------------------

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True):
        n_episodes    = n_episodes or self.config['n_episodes']
        n_envs        = self.config.get('n_envs', 64)
        collect_steps = self.config.get('collect_steps', 32)
        eval_interval = self.config.get('eval_interval', 2500)
        eval_episodes = self.config.get('eval_episodes', 512)
        save_every    = self.config.get('save_every', 10000)

        self.metrics_logger = _MetricsLogger(
            config=self.config, n_episodes=n_episodes,
            eval_interval=eval_interval, eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_quality = -1.0

        n_ckpts = self.config.get('n_checkpoints', 50)
        ckpt_interval = max(1, n_episodes // n_ckpts) if n_ckpts > 0 else 0
        ckpt_dir = os.path.join(self.metrics_logger.dir, 'checkpoints')
        if ckpt_interval > 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        np.random.seed(self.config.get('seed', 0))
        self._build_collect_fn()

        self._key, init_key = jax.random.split(self._key)
        init_keys = jax.random.split(init_key, n_envs)
        current_states = jax.vmap(lambda k: self.env.reset(k)[1])(init_keys)

        decay_episodes = max(1, int(self.exploration_fraction * n_episodes))

        if verbose:
            alg = self.config['algorithm'].upper()
            print(f"Training {alg} on CVRP n={self.env.n_customers} "
                  f"cap={self.env.capacity} [VECTORIZED]")
            print(f"  n_envs={n_envs}  collect_steps={collect_steps}  "
                  f"({n_envs * collect_steps} transitions/chunk)")
            print(f"  obs_shape={self.obs_shape}  n_actions={self.n_actions}  "
                  f"exploration={self.exploration} ({self._expl_start}->{self._expl_end} "
                  f"over {decay_episodes} eps)")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}", flush=True)

        total_episodes = len(self.episode_returns)
        prev_eval_ep = prev_save_ep = prev_ckpt_ep = 0
        n_chunk_steps = n_envs * collect_steps

        pbar = tqdm(total=n_episodes, disable=not verbose)
        pbar.update(total_episodes)

        while total_episodes < n_episodes:
            # 'buffer.add' and 'update' are accumulated components in TrainingTimer;
            # without a begin/flush window they would pile into one record at close().
            timer.begin_episode(total_episodes)
            self._key, chunk_key = jax.random.split(self._key)
            step_keys = jax.random.split(chunk_key, n_envs * collect_steps).reshape(
                n_envs, collect_steps, 2
            )

            with timer('collect', episode=total_episodes):
                current_states, outputs = self._collect_fn(
                    self.params, jnp.float32(self.epsilon), current_states, step_keys
                )
                jax.block_until_ready(outputs[4])

            with timer('buffer.add', episode=total_episodes):
                self._add_chunk_to_buffer(outputs)

            ep_returns_np = np.array(outputs[5])
            ep_lengths_np = np.array(outputs[6])
            ended_flat    = np.array(outputs[7]).astype(bool).reshape(-1)
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
            # Polyak runs inside _update; the hard sync is only used if target_tau is None.
            target_freq_q = max(1, self.target_update_freq // self.n_steps_per_update)
            with timer('update', episode=total_episodes):
                for i in range(n_q_updates):
                    self._update()
                    if self.target_tau is None and (q_start + i) % target_freq_q == 0:
                        self._update_target_network()

            decay = min(1.0, total_episodes / decay_episodes)
            self.epsilon = self._expl_start + (self._expl_end - self._expl_start) * decay

            pbar.update(n_new)
            if len(self.episode_returns) >= 100:
                avg = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"tour(100): {-avg:.3f}  T: {self.epsilon:.3f}")

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
                if verbose:
                    print(f"\n  ep {total_episodes}: tour {metrics['mean_tour']:.4f} "
                          f"quality {metrics['win_rate']:.4f} "
                          f"solved {metrics['solved_frac']:.3f}", flush=True)
                if metrics['win_rate'] > best_quality:
                    best_quality = metrics['win_rate']
                    self.save(best_path)
                prev_eval_ep = total_episodes

            timer.flush_episode()

        pbar.close()
        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
