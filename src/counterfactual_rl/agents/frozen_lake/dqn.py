"""
DQN and DQN+PER agents for FrozenLake.

Variants selected via config['algorithm']:
  'dqn-uniform' — Vanilla DQN, uniform buffer sampling
  'dqn'         — DQN with Prioritized Experience Replay
"""

import os
import pickle
from datetime import datetime
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.timing import TrainingTimer
from .config import DEFAULT_CONFIG
from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv


# ── Q-Network ─────────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    n_states: int
    hidden_dim: int = 64
    n_layers: int = 2
    n_actions: int = 4

    @nn.compact
    def __call__(self, state_idx):
        x = jax.nn.one_hot(state_idx, self.n_states)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


# ── Metrics Logger ─────────────────────────────────────────────────────────────

class _MetricsLogger:
    _HEADER = (
        f"{'episode':>8} {'updates':>10} {'epsilon':>8} "
        f"{'success_rate':>14} {'avg_steps':>10} {'avg_return':>12}\n"
    )

    def __init__(self, config: dict, n_episodes: int, eval_interval, eval_episodes: int):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self.timer = TrainingTimer(run_dir)

        self._path = os.path.join(run_dir, 'metrics.log')
        self._file = open(self._path, 'w')
        self._file.write(f"# FrozenLake DQN — {datetime.now()}\n")
        self._file.write(
            f"# Map: {config['map_name']}  Slippery: {config.get('is_slippery', True)}"
            f"  Algorithm: {config.get('algorithm')}\n"
        )
        self._file.write(
            f"# Episodes: {n_episodes}  Eval interval: {eval_interval}"
            f"  Eval episodes: {eval_episodes}\n#\n"
        )
        for k, v in config.items():
            self._file.write(f"# {k}: {v}\n")
        self._file.write("#\n")
        self._file.write(self._HEADER)
        self._file.flush()

        self._episodes: list = []
        self._success_rates: list = []
        self._avg_steps: list = []
        self._avg_returns: list = []

    def log_eval(self, episode: int, updates: int, epsilon: float, metrics: dict):
        self._file.write(
            f"{episode:>8d} {updates:>10d} {epsilon:>8.3f} "
            f"{metrics['success_rate']:>14.1%} {metrics['avg_steps']:>10.1f} "
            f"{metrics['avg_return']:>12.3f}\n"
        )
        self._file.flush()
        self._episodes.append(episode)
        self._success_rates.append(metrics['success_rate'])
        self._avg_steps.append(metrics['avg_steps'])
        self._avg_returns.append(metrics['avg_return'])

    def plot_eval_curves(self):
        if not self._episodes:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self._episodes, self._success_rates)
        axes[0].set(title='Success Rate', xlabel='Episode', ylabel='Rate')
        axes[1].plot(self._episodes, self._avg_steps)
        axes[1].set(title='Avg Steps to Goal', xlabel='Episode')
        axes[2].plot(self._episodes, self._avg_returns)
        axes[2].set(title='Avg Return', xlabel='Episode')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, 'eval_curves.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    def plot_training_curves(self, episode_returns: list, episode_lengths: list):
        if not episode_returns:
            return
        window = min(100, len(episode_returns))
        kern = np.ones(window) / window
        avg_r = np.convolve(episode_returns, kern, mode='valid')
        avg_l = np.convolve(episode_lengths, kern, mode='valid')
        x = np.arange(window - 1, len(episode_returns))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(episode_returns, alpha=0.3, color='steelblue')
        ax1.plot(x, avg_r, color='steelblue', linewidth=2, label=f'Rolling {window}')
        ax1.set(title='Episode Return', xlabel='Episode')
        ax1.legend()
        ax2.plot(episode_lengths, alpha=0.3, color='darkorange')
        ax2.plot(x, avg_l, color='darkorange', linewidth=2, label=f'Rolling {window}')
        ax2.set(title='Episode Length', xlabel='Episode')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.dir, 'training_curves.png'), dpi=120, bbox_inches='tight'
        )
        plt.close(fig)

    def close(self):
        self.plot_eval_curves()
        self._file.close()
        self.timer.close()


# ── Agent ─────────────────────────────────────────────────────────────────────

class FrozenLakeDQN:
    """
    DQN (uniform or PER) for single-agent FrozenLake.

    Observation encoding: one-hot(state_index) → (n_states,) MLP input.
    No action masking — all 4 directions always valid; walls clamp position.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env = FrozenLakeEnv(
            map_name=self.config['map_name'],
            is_slippery=self.config.get('is_slippery', True),
        )
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions

        self._key = jax.random.PRNGKey(self.config.get('seed', 0))

        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_start = self.config['epsilon_start']
        self.epsilon_end = self.config['epsilon_end']
        self.epsilon_decay_episodes = self.config['epsilon_decay_episodes']
        self.batch_size = self.config['batch_size']
        self.target_update_freq = self.config['target_update_freq']
        self.n_steps_per_update = self.config['n_steps_per_update']

        self.network = _QNetwork(
            n_states=self.n_states,
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
        )

        self._key, init_key = jax.random.split(self._key)
        self.params = self.network.init(init_key, jnp.int32(0))
        self.target_params = jax.tree.map(jnp.copy, self.params)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(self.config['alpha']),
        )
        self.opt_state = self.optimizer.init(self.params)

        per = self.config.get('PER_parameters', {})
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config['buffer_capacity'],
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.25),
            max_priority=per.get('maximum_priority', 1.0),
            uniform=(self.config['algorithm'] == 'dqn-uniform'),
        )

        self.total_steps = 0
        self.episode_returns: list = []
        self.episode_lengths: list = []
        self._current_episode = 0

        self._build_jit_fns()

    def _build_jit_fns(self):
        network = self.network
        gamma = self.gamma

        @jax.jit
        def greedy_action(params, state_idx):
            return jnp.argmax(network.apply(params, state_idx))

        @jax.jit
        def update_step(params, target_params, opt_state,
                        states, actions, rewards, next_states, dones, weights):
            def loss_fn(p):
                q = jax.vmap(network.apply, in_axes=(None, 0))(p, states)          # (B, 4)
                q_taken = q[jnp.arange(q.shape[0]), actions]                        # (B,)
                next_q = jax.vmap(
                    network.apply, in_axes=(None, 0)
                )(target_params, next_states)                                        # (B, 4)
                max_next_q = next_q.max(axis=-1)                                     # (B,)
                targets = rewards + gamma * max_next_q * (1.0 - dones)
                td_errors = targets - q_taken
                loss = jnp.mean(weights * td_errors ** 2)
                return loss, td_errors

            (_, td_errors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, td_errors

        self._greedy_action = greedy_action
        self._update_step = update_step

    def select_action(self, state: int) -> int:
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(self._greedy_action(self.params, jnp.int32(state)))

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return
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

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def evaluate(self, n_episodes: int = 50) -> dict:
        wins = steps = 0
        total_return = 0.0
        for _ in range(n_episodes):
            self._key, rk = jax.random.split(self._key)
            _, state = self.env.reset(rk)
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < 200:
                action = int(self._greedy_action(self.params, jnp.int32(int(state))))
                self._key, sk = jax.random.split(self._key)
                _, state, reward, done, _ = self.env.step(
                    sk, jnp.int32(int(state)), jnp.int32(action)
                )
                ep_return += float(reward)
                ep_steps += 1
                done = bool(done)
            wins += int(ep_return > 0)
            steps += ep_steps
            total_return += ep_return
        return {
            'success_rate': wins / n_episodes,
            'avg_steps': steps / n_episodes,
            'avg_return': total_return / n_episodes,
        }

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'FrozenLakeDQN':
        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 100)
        eval_episodes = self.config.get('eval_episodes', 50)
        save_every = self.config.get('save_every', 500)

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
        np.random.seed(self.config.get('seed', 0))

        if verbose:
            alg = self.config['algorithm'].upper()
            print(f"Training {alg} on FrozenLake-{self.config['map_name']}")
            print(f"  Slippery: {self.config.get('is_slippery', True)}  |  n_states: {self.n_states}")
            print(f"  Episodes: {n_episodes}  |  ε: {self.epsilon_start}→{self.epsilon_end} over {self.epsilon_decay_episodes} eps")
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
                action = self.select_action(state)
                self._key, sk = jax.random.split(self._key)
                _, next_state, reward, done, _ = self.env.step(
                    sk, jnp.int32(state), jnp.int32(action)
                )
                next_state = int(next_state)
                reward = float(reward)
                done = bool(done)

                self.buffer.add({'s': state, 'a': action, 'r': reward, "s'": next_state, 'done': done})
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
                self.metrics_logger.log_eval(episode + 1, self.total_steps, self.epsilon, metrics)
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

    def save(self, path: str):
        ckpt = {
            'params': jax.tree.map(np.array, self.params),
            'target_params': jax.tree.map(np.array, self.target_params),
            'opt_state': jax.tree.map(
                lambda x: np.array(x) if hasattr(x, 'shape') else x, self.opt_state
            ),
            'config': self.config,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'total_steps': self.total_steps,
        }
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        self.params = jax.tree.map(jnp.array, ckpt['params'])
        self.target_params = jax.tree.map(jnp.array, ckpt['target_params'])
        self.opt_state = jax.tree.map(
            lambda x: jnp.array(x) if hasattr(x, 'shape') else x, ckpt['opt_state']
        )
        self.episode_returns = ckpt.get('episode_returns', [])
        self.episode_lengths = ckpt.get('episode_lengths', [])
        self.total_steps = ckpt.get('total_steps', 0)
        self._build_jit_fns()
