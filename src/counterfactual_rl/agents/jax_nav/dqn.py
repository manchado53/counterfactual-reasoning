"""
DQN and DQN+PER agents for JaxNav (single-agent, discrete 15-action navigation).

Mirror of ``agents/frozen_lake/dqn.py``. The only structural differences, both
forced by JaxNav being a continuous-observation robot rather than a tabular grid:

  * The Q-network is an MLP over the 205-vector observation (no one-hot of an
    integer state).
  * State and observation are DISTINCT: the JaxNav ``State`` pytree is what you
    step; the 205-vector obs is the network input. Transitions store the obs
    vectors (for TD updates); the pytree state is only needed by CCE (see
    ``consequence_dqn.py``), so the plain DQN/PER path here never touches it.

Variants via config['algorithm']:
  'dqn-uniform' — vanilla DQN, uniform buffer
  'dqn'         — DQN + PER
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
from counterfactual_rl.envs.jax_nav import JaxNavEnv


# ── Q-Network ─────────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    """MLP over the JaxNav observation vector. (FrozenLake one-hots an int here.)"""
    obs_dim: int
    hidden_dim: int = 128
    n_layers: int = 3
    n_actions: int = 15

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


# ── Metrics Logger ─────────────────────────────────────────────────────────────

class _MetricsLogger:
    """Same column format as FrozenLake (so analysis/claim2 reuses its parser)."""
    _HEADER = (
        f"{'episode':>8} {'updates':>10} {'epsilon':>8} "
        f"{'win_rate':>14} {'avg_length':>10} {'avg_return':>12}\n"
    )

    def __init__(self, config: dict, n_episodes: int, eval_interval, eval_episodes: int):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self.timer = TrainingTimer(run_dir)

        self._path = os.path.join(run_dir, 'metrics.log')
        self._file = open(self._path, 'w')
        self._file.write(f"# JaxNav DQN — {datetime.now()}\n")
        scenario = config.get('scenario') or f"{config.get('map_id')} {config.get('map_size')}"
        self._file.write(
            f"# Scenario: {scenario}  Algorithm: {config.get('algorithm')}\n"
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
        self._win_rates: list = []
        self._avg_lengths: list = []
        self._avg_returns: list = []

    def log_eval(self, episode: int, updates: int, epsilon: float, metrics: dict):
        self._file.write(
            f"{episode:>8d} {updates:>10d} {epsilon:>8.3f} "
            f"{metrics['win_rate']:>14.1%} {metrics['avg_length']:>10.1f} "
            f"{metrics['avg_return']:>12.3f}\n"
        )
        self._file.flush()
        self._episodes.append(episode)
        self._win_rates.append(metrics['win_rate'])
        self._avg_lengths.append(metrics['avg_length'])
        self._avg_returns.append(metrics['avg_return'])

    def plot_eval_curves(self):
        if not self._episodes:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self._episodes, self._win_rates)
        axes[0].set(title='Goal-reach Rate', xlabel='Episode', ylabel='Rate')
        axes[1].plot(self._episodes, self._avg_lengths)
        axes[1].set(title='Avg Episode Length', xlabel='Episode')
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
        plt.savefig(os.path.join(self.dir, 'training_curves.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    def close(self):
        self.plot_eval_curves()
        self._file.close()
        self.timer.close()


# ── Agent ─────────────────────────────────────────────────────────────────────

class JaxNavDQN:
    """DQN (uniform or PER) for single-agent JaxNav. Vectorized subclass does the heavy runs."""

    def __init__(self, config: Optional[dict] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env = JaxNavEnv(
            scenario=self.config.get('scenario'),
            map_id=self.config.get('map_id', 'Grid-Rand-Poly'),
            map_size=tuple(self.config.get('map_size', (11, 11))),
            fill=self.config.get('fill', 0.3),
            goal_radius=self.config.get('goal_radius', 0.5),
            goal_rew=self.config.get('goal_rew', 1.0),
            coll_rew=self.config.get('coll_rew', 0.0),
            max_steps=self.config.get('max_steps', 200),
            sparse_reward=self.config.get('sparse_reward', True),
        )
        self.obs_dim = self.env.obs_dim
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
            obs_dim=self.obs_dim,
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            n_actions=self.n_actions,
        )

        self._key, init_key = jax.random.split(self._key)
        self.params = self.network.init(init_key, jnp.zeros(self.obs_dim, dtype=jnp.float32))
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
        double_dqn = self.config.get('double_dqn', True)

        @jax.jit
        def greedy_action(params, obs):
            return jnp.argmax(network.apply(params, obs))

        @jax.jit
        def update_step(params, target_params, opt_state,
                        states, actions, rewards, next_states, dones, weights):
            def loss_fn(p):
                q = jax.vmap(network.apply, in_axes=(None, 0))(p, states)           # (B, 15)
                q_taken = q[jnp.arange(q.shape[0]), actions]                         # (B,)
                next_q_target = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
                if double_dqn:
                    # action chosen by the ONLINE net, evaluated by the TARGET net
                    next_q_online = jax.vmap(network.apply, in_axes=(None, 0))(p, next_states)
                    next_actions = jnp.argmax(jax.lax.stop_gradient(next_q_online), axis=-1)
                    max_next_q = next_q_target[jnp.arange(next_q_target.shape[0]), next_actions]
                else:
                    max_next_q = next_q_target.max(axis=-1)
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

    def select_action(self, obs) -> int:
        if np.random.uniform() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(self._greedy_action(self.params, jnp.asarray(obs, dtype=jnp.float32)))

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return
        transitions, indices, weights = self.buffer.sample(self.batch_size)
        states  = jnp.asarray(np.stack([t['s']  for t in transitions]), dtype=jnp.float32)   # (B,205)
        actions = jnp.asarray([t['a']    for t in transitions], dtype=jnp.int32)
        rewards = jnp.asarray([t['r']    for t in transitions], dtype=jnp.float32)
        nexts   = jnp.asarray(np.stack([t["s'"] for t in transitions]), dtype=jnp.float32)   # (B,205)
        dones   = jnp.asarray([t['done'] for t in transitions], dtype=jnp.float32)
        wts     = jnp.asarray(weights, dtype=jnp.float32)

        self.params, self.opt_state, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, nexts, dones, wts,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def evaluate(self, n_episodes: int = 100) -> dict:
        wins = steps = 0
        total_return = 0.0
        max_steps = self.env.max_steps
        for _ in range(n_episodes):
            self._key, rk = jax.random.split(self._key)
            obs, state = self.env.reset(rk)
            done = False
            reached = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < max_steps:
                action = int(self._greedy_action(self.params, jnp.asarray(obs, jnp.float32)))
                self._key, sk = jax.random.split(self._key)
                obs, state, reward, done, _ = self.env.step(sk, state, jnp.int32(action))
                reached = reached or bool(state.goal_reached[0])
                ep_return += float(reward)
                ep_steps += 1
                done = bool(done)
            wins += int(reached)
            steps += ep_steps
            total_return += ep_return
        return {
            'win_rate':   wins / n_episodes,
            'avg_length': steps / n_episodes,
            'avg_return': total_return / n_episodes,
        }

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'JaxNavDQN':
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
        early_stop_win_rate = self.config.get('early_stop_win_rate', None)

        np.random.seed(self.config.get('seed', 0))

        if verbose:
            alg = self.config['algorithm'].upper()
            print(f"Training {alg} on JaxNav [{self.config.get('scenario') or self.config.get('map_id')}]")
            print(f"  obs_dim: {self.obs_dim}  n_actions: {self.n_actions}")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            print(f"  Run dir: {self.metrics_logger.dir}")

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            self._current_episode = episode
            self._key, rk = jax.random.split(self._key)
            obs, state = self.env.reset(rk)
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < self.env.max_steps:
                action = self.select_action(obs)
                self._key, sk = jax.random.split(self._key)
                next_obs, next_state, reward, done, _ = self.env.step(sk, state, jnp.int32(action))
                reward = float(reward)
                done = bool(done)
                self.buffer.add({
                    's': np.asarray(obs, dtype=np.float32), 'a': int(action),
                    'r': reward, "s'": np.asarray(next_obs, dtype=np.float32), 'done': done,
                })
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
                q_updates = self.total_steps // self.n_steps_per_update
                self.metrics_logger.log_eval(episode + 1, q_updates, self.epsilon, metrics)
                if metrics['win_rate'] > best_success:
                    best_success = metrics['win_rate']
                    self.save(best_path)
                if early_stop_win_rate is not None and metrics['win_rate'] >= early_stop_win_rate:
                    break

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
            'epsilon': self.epsilon,
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
        self.epsilon = ckpt.get('epsilon', self.epsilon_start)
        self._build_jit_fns()
