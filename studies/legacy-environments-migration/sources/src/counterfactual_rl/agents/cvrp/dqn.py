"""
DQN and DQN+PER agents for the routing env (CVRP / TSP).

Variants selected via config['algorithm']:
  'dqn-uniform' — Vanilla DQN, uniform buffer sampling
  'dqn'         — DQN with Prioritized Experience Replay

Adapted from agents/frozen_lake/dqn.py. TWO routing-specific changes, both essential:

1. ACTION MASKING. Unlike FrozenLake (all 4 moves always legal), a routing state has a
   varying legal set — a served stop cannot be revisited, and in CVRP a stop whose demand
   exceeds the remaining load is temporarily unreachable. The mask is applied in THREE
   places, and missing any one of them breaks learning:
       a. greedy action selection   (else the policy "teleports" to a served stop)
       b. epsilon-greedy exploration (else most random actions are no-ops with a penalty)
       c. the TARGET network's max over next actions  <- the silent killer: an unmasked
          max would bootstrap from an illegal action's Q-value and poison every target.

2. FEATURE OBSERVATIONS. Routing has thousands of states, so one-hot(state) would force
   the network to memorize each one. The env exposes `state_features` (one-hot current
   node + visited bits + load fraction); the network looks features up by integer index,
   so the replay buffer and CCE scoring still store plain ints.

Metric. "win rate" is meaningless here, so evaluation reports OPT_RATIO — the optimal
plan length divided by the achieved length, in (0, 1], where 1.0 means provably optimal.
It is bounded and higher-is-better, so the rliable/IQM machinery carries over unchanged.
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
from counterfactual_rl.envs.cvrp import CVRPEnv

# Large negative sentinel for masked-out actions. Finite (not -inf) so gradients and
# argmax stay well-defined even if an entire row were masked.
MASK_FILL = -1e9


def build_env(config: dict) -> CVRPEnv:
    """
    Construct the routing env described by a config dict.

    `capacity` is taken from the config so a sweep can turn the load limit into an
    experimental dial (tighter capacity -> more pivotal reload decisions); pass
    capacity=None for TSP mode.
    """
    from counterfactual_rl.envs.cvrp import INSTANCES

    name = config.get('instance', 'default')

    if name not in INSTANCES:
        raise ValueError(f"Unknown instance '{name}'. Options: {sorted(INSTANCES)}")
    spec = INSTANCES[name]

    demand = np.asarray(spec['demand'], dtype=np.int32).copy()
    scale = config.get('demand_scale', 1.0)
    if scale != 1.0:
        demand = np.maximum(1, np.round(demand * scale)).astype(np.int32)
        demand[0] = 0

    capacity = config['capacity'] if 'capacity' in config else spec['capacity']

    return CVRPEnv(node_xy=spec['xy'], demand=demand, capacity=capacity,
                   travel_noise=config.get('travel_noise', 0.0))


# ── Q-Network ─────────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    """MLP over routing features. Emits one Q-value per node (action = 'go to node')."""
    hidden_dim: int = 64
    n_layers: int = 2
    n_actions: int = 11

    @nn.compact
    def __call__(self, features):
        x = features
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


# ── Metrics Logger ─────────────────────────────────────────────────────────────

class _MetricsLogger:
    _HEADER = (
        f"{'episode':>8} {'updates':>10} {'epsilon':>8} "
        f"{'opt_ratio':>14} {'avg_length':>10} {'avg_return':>12}\n"
    )

    def __init__(self, config: dict, env: CVRPEnv, n_episodes: int,
                 eval_interval, eval_episodes: int, optimal_length: float):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self.timer = TrainingTimer(run_dir)

        self._path = os.path.join(run_dir, 'metrics.log')
        self._file = open(self._path, 'w')
        self._file.write(f"# CVRP DQN — {datetime.now()}\n")
        self._file.write(
            f"# Customers: {env.n_customers}  Capacity: {env.capacity}"
            f"  Algorithm: {config.get('algorithm')}\n"
        )
        self._file.write(
            f"# States: {env.n_states}  Optimal length: {optimal_length:.6f}"
            f"  Min loads: {env.min_loads()}\n"
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
        self._opt_ratios: list = []
        self._avg_lengths: list = []
        self._avg_returns: list = []

    def log_eval(self, episode: int, updates: int, epsilon: float, metrics: dict):
        self._file.write(
            f"{episode:>8d} {updates:>10d} {epsilon:>8.3f} "
            f"{metrics['opt_ratio']:>14.4f} {metrics['avg_length']:>10.1f} "
            f"{metrics['avg_return']:>12.3f}\n"
        )
        self._file.flush()
        self._episodes.append(episode)
        self._opt_ratios.append(metrics['opt_ratio'])
        self._avg_lengths.append(metrics['avg_length'])
        self._avg_returns.append(metrics['avg_return'])

    def plot_eval_curves(self):
        if not self._episodes:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self._episodes, self._opt_ratios)
        axes[0].axhline(1.0, color='green', ls='--', lw=1, label='optimal')
        axes[0].set(title='Fraction of optimal', xlabel='Episode', ylabel='opt/achieved')
        axes[0].legend()
        axes[1].plot(self._episodes, self._avg_lengths)
        axes[1].set(title='Avg plan steps', xlabel='Episode')
        axes[2].plot(self._episodes, self._avg_returns)
        axes[2].set(title='Avg Return (-distance)', xlabel='Episode')
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

class CVRPDQN:
    """
    DQN (uniform or PER) for routing, with action masking.

    Observation encoding: env.state_features[state_index] → (feature_dim,) MLP input.
    Action masking: env.action_masks[state_index] → illegal actions get MASK_FILL.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env = build_env(self.config)
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self.features = self.env.state_features       # (n_states, feature_dim)
        self.masks = self.env.action_masks            # (n_states, n_actions) bool
        self._masks_np = np.asarray(self.masks)

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
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            n_actions=self.n_actions,
        )

        self._key, init_key = jax.random.split(self._key)
        self.params = self.network.init(init_key, self.features[0])
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

        self.log_sampling = self.config.get('log_sampling', False)
        self.sampling_snapshot_interval = self.config.get('sampling_snapshot_interval', 2000)
        self._draw_update_count = 0
        if self.log_sampling:
            self.buffer.enable_draw_log = True

        # Exact optimum for the instance — the denominator of the opt_ratio metric.
        from counterfactual_rl.analysis.claim1.cvrp.oracle import optimal_tour
        self._optimal_tour, self.optimal_length = optimal_tour(self.env, gamma=1.0)

        self._build_jit_fns()

    def _build_jit_fns(self):
        network = self.network
        gamma = self.gamma
        features = self.features
        masks = self.masks

        @jax.jit
        def masked_q(params, state_idx):
            """Q-values with illegal actions driven to MASK_FILL."""
            q = network.apply(params, features[state_idx])
            return jnp.where(masks[state_idx], q, MASK_FILL)

        @jax.jit
        def greedy_action(params, state_idx):
            return jnp.argmax(masked_q(params, state_idx))

        @jax.jit
        def update_step(params, target_params, opt_state,
                        states, actions, rewards, next_states, dones, weights):
            def loss_fn(p):
                feats = features[states]                                        # (B, F)
                q = jax.vmap(network.apply, in_axes=(None, 0))(p, feats)        # (B, A)
                q_taken = q[jnp.arange(q.shape[0]), actions]                    # (B,)

                next_feats = features[next_states]
                next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_feats)
                # CRITICAL: mask before the max, or the target bootstraps from an
                # illegal action's Q-value and every update is poisoned.
                next_q = jnp.where(masks[next_states], next_q, MASK_FILL)
                max_next_q = next_q.max(axis=-1)                                # (B,)
                # A terminal state has no legal action (all MASK_FILL); the (1 - done)
                # factor zeroes it out, but clamp so MASK_FILL can never leak in.
                max_next_q = jnp.where(dones > 0, 0.0, max_next_q)

                targets = rewards + gamma * max_next_q * (1.0 - dones)
                td_errors = targets - q_taken
                loss = jnp.mean(weights * td_errors ** 2)
                return loss, td_errors

            (_, td_errors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, td_errors

        self._masked_q = masked_q
        self._greedy_action = greedy_action
        self._update_step = update_step

    def select_action(self, state: int) -> int:
        """Epsilon-greedy over LEGAL actions only."""
        legal = np.flatnonzero(self._masks_np[state])
        if legal.size == 0:
            return 0
        if np.random.uniform() < self.epsilon:
            return int(np.random.choice(legal))
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

        if self.log_sampling and self.sampling_snapshot_interval > 0:
            self._draw_update_count += 1
            if self._draw_update_count % self.sampling_snapshot_interval == 0:
                self.buffer.snapshot_draws(self._draw_update_count, self._current_episode)

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def rollout_greedy(self, max_steps: int = 200):
        """Run the greedy policy once. Returns (node path, total return)."""
        self._key, rk = jax.random.split(self._key)
        _, state = self.env.reset(rk)
        state = int(state)
        path = [int(self.env.state_current_np[state])]
        total, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            action = int(self._greedy_action(self.params, jnp.int32(state)))
            self._key, sk = jax.random.split(self._key)
            _, state, reward, done, _ = self.env.step(sk, jnp.int32(state), jnp.int32(action))
            state = int(state)
            total += float(reward)
            path.append(int(self.env.state_current_np[state]))
            steps += 1
            done = bool(done)
        return path, total

    def evaluate(self, n_episodes: int = 20) -> dict:
        """
        Greedy evaluation. The policy and env are both deterministic, so all episodes
        agree; n_episodes is kept for API symmetry with the other envs.

        opt_ratio = optimal_length / achieved_length in (0, 1]; 1.0 == provably optimal.
        An unfinished plan (hit the step cap) scores 0.0.
        """
        ratios, lengths, returns = [], [], []
        for _ in range(max(1, n_episodes)):
            path, total = self.rollout_greedy()
            finished = (
                len(path) > 2
                and path[-1] == 0
                and sorted(p for p in path if p != 0) == list(range(1, self.env.n_nodes))
            )
            achieved = self.env.tour_length(path)
            ratios.append(self.optimal_length / achieved if (finished and achieved > 0) else 0.0)
            lengths.append(len(path) - 1)
            returns.append(total)
        return {
            'opt_ratio':  float(np.mean(ratios)),
            'avg_length': float(np.mean(lengths)),
            'avg_return': float(np.mean(returns)),
        }

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'CVRPDQN':
        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 100)
        eval_episodes = self.config.get('eval_episodes', 20)
        save_every = self.config.get('save_every', 500)

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
            alg = self.config['algorithm'].upper()
            mode = f"CVRP cap={self.env.capacity}" if self.env.is_capacitated else "TSP"
            print(f"Training {alg} on {mode} ({self.env.n_customers} customers)")
            print(f"  States: {self.n_states}  |  features: {self.env.feature_dim}"
                  f"  |  optimal length: {self.optimal_length:.4f}")
            print(f"  Episodes: {n_episodes}  |  ε: {self.epsilon_start}→{self.epsilon_end}"
                  f" over {self.epsilon_decay_episodes} eps")
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
                    self.buffer.add({'s': state, 'a': action, 'r': reward,
                                     "s'": next_state, 'done': done})
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
