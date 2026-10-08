"""
DQN and DQN+PER for the neural CVRP (the RP-DQN benchmark).

Mirror of ``agents/jax_nav/dqn.py``. The one structural difference is the
Q-network: a Kool-style attention encoder/decoder instead of an MLP, re-run at
every decoding step. That re-run is the whole adaptation that makes a Q-learner
work on routing -- it is what lets the encoder see DYNAMIC node features (served,
infeasible, current) instead of only static coordinates -- and it is also why the
run is expensive.

Everything downstream (the shared rollout, ``_update_step``, the vectorized
collect loop) depends only on the ``network.apply(params, obs) -> (n_actions,)``
contract, so the observation being ``(n_nodes, 8)`` rather than a flat vector
changes nothing else.

Variants via config['algorithm']:
  'dqn-uniform' -- vanilla DQN, uniform buffer
  'dqn'         -- DQN + PER  (this is RP-DQN's own arm, and CCE's baseline)
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
from counterfactual_rl.envs.cvrp_neural import (
    CVRPNeuralEnv, _F_CURRENT, _F_DEPOT, _F_LOAD,
)

# Finite, NOT -inf. The TD target is ``r + gamma * max_next_q * (1 - done)``, and
# at a terminal ``-inf * 0`` is NaN, which would poison the whole batch.
MASK_VALUE = -1e9


# ── Q-Network ─────────────────────────────────────────────────────────────────

class _AttentionBlock(nn.Module):
    """One encoder block: MHA + skip, then feed-forward + skip (paper eq. 4-5).

    Normalization is off by default because RP-DQN ships ``normalization: 'no'``.
    It is also what this repo already learned the hard way on sparse boolean
    observations, where LayerNorm zeroed the gradient and Q diverged.
    """
    embed_dim: int
    n_heads: int
    ff_dim: int
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, h):
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )(h, h, deterministic=True)
        h = h + attn
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)

        ff = nn.Dense(self.ff_dim)(h)
        ff = nn.relu(ff)
        ff = nn.Dense(self.embed_dim)(ff)
        h = h + ff
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        return h


class _QNetwork(nn.Module):
    """Attention encoder + single-head compatibility decoder emitting Q-values.

    Input  ``obs``  (n_nodes, N_FEATURES)
    Output ``q``    (n_nodes,)  -- action i means "drive to node i"; 0 is the depot.

    No softmax: RP-DQN drops it because the model outputs Q-values rather than a
    policy, and infeasible actions are masked to a large negative constant.
    """
    n_actions: int
    embed_dim: int = 128
    n_layers: int = 3
    n_heads: int = 8
    ff_dim: int = 512
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        is_depot = obs[:, _F_DEPOT:_F_DEPOT + 1]                       # (N,1)

        # Paper eq. 1-2: the depot gets its OWN projection, weights not shared.
        h = jnp.where(
            is_depot > 0.5,
            nn.Dense(self.embed_dim, name='proj_depot')(obs),
            nn.Dense(self.embed_dim, name='proj_node')(obs),
        )                                                              # (N,E)

        for i in range(self.n_layers):
            h = _AttentionBlock(
                embed_dim=self.embed_dim, n_heads=self.n_heads,
                ff_dim=self.ff_dim, use_layer_norm=self.use_layer_norm,
                name=f'block_{i}',
            )(h)                                                       # (N,E)

        # Decoder context C = [h_graph ; c ; h_last]  (paper eq. 6).
        h_graph = jnp.mean(h, axis=0)                                  # (E,)
        cur_w = obs[:, _F_CURRENT]                                     # (N,) one-hot
        h_last = cur_w @ h                                             # (E,)
        load = obs[0, _F_LOAD]                                         # scalar c
        ctx = jnp.concatenate([h_graph, h_last, load[None]])           # (2E+1,)
        ctx = nn.Dense(self.embed_dim, name='ctx_proj')(ctx)           # (E,)

        # Attend from the context to the feasible nodes only.
        mask = CVRPNeuralEnv.action_mask_from_obs(obs)                 # (N,) bool
        ctx_hat = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.embed_dim,
            out_features=self.embed_dim, name='dec_mha',
        )(ctx[None, :], h, mask=mask[None, None, :], deterministic=True)[0]

        # Single-head compatibility (paper eq. 7).
        wq = nn.Dense(self.embed_dim, use_bias=False, name='dec_wq')(ctx_hat)   # (E,)
        wk = nn.Dense(self.embed_dim, use_bias=False, name='dec_wk')(h)         # (N,E)
        q = (wk @ wq) / jnp.sqrt(jnp.float32(self.embed_dim))                   # (N,)

        return jnp.where(mask, q, MASK_VALUE)


# ── Metrics Logger ─────────────────────────────────────────────────────────────

class _MetricsLogger:
    """Same 6-column format as FrozenLake/JaxNav, so analysis/claim2 reuses its parser.

    'win_rate' here is ``optimal_ref / mean_tour_length`` -- 1.000 is optimal and
    RP-DQN's published greedy arm is 6.10/6.36 = 0.959. The claim-2 pipeline needs a
    metric that INCREASES with quality, and raw tour length decreases; this also makes
    the registered threshold (0.95) read as "within 5% of optimal".
    """
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
        # The literal 'CVRP' is what parse_logs._detect_env matches on. Do not drop it.
        self._file.write(f"# CVRP (neural, RP-DQN) DQN — {datetime.now()}\n")
        self._file.write(
            f"# Scenario: cvrp_neural n={config.get('n_customers')} "
            f"cap={config.get('capacity')}  Algorithm: {config.get('algorithm')}\n"
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
        axes[0].axhline(0.959, ls='--', c='grey', lw=1, label='RP-DQN greedy (4.26%)')
        axes[0].set(title='Quality (optimal / tour length)', xlabel='Episode', ylabel='Ratio')
        axes[0].legend(fontsize=7)
        axes[1].plot(self._episodes, self._avg_lengths)
        axes[1].set(title='Avg Decisions / Episode', xlabel='Episode')
        axes[2].plot(self._episodes, [-r for r in self._avg_returns])
        axes[2].set(title='Mean Tour Length', xlabel='Episode')
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
        ax1.set(title='Episode Return (= -tour length)', xlabel='Episode')
        ax1.legend()
        ax2.plot(episode_lengths, alpha=0.3, color='darkorange')
        ax2.plot(x, avg_l, color='darkorange', linewidth=2, label=f'Rolling {window}')
        ax2.set(title='Decisions per Episode', xlabel='Episode')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, 'training_curves.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    def close(self):
        self.plot_eval_curves()
        self._file.close()
        self.timer.close()


# ── Agent ─────────────────────────────────────────────────────────────────────

class CVRPNeuralDQN:
    """DQN (uniform or PER) for the neural CVRP. The vectorized subclass does real runs."""

    def __init__(self, config: Optional[dict] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env = CVRPNeuralEnv(
            n_customers=self.config['n_customers'],
            capacity=self.config.get('capacity'),
            max_steps=self.config.get('max_steps'),
        )
        # Write the resolved values back so the metrics.log header records what ran.
        self.config['capacity'] = self.env.capacity
        self.config['max_steps'] = self.env.max_steps

        self.obs_shape = self.env.obs_shape
        self.obs_dim = self.env.obs_dim
        self.n_actions = self.env.n_actions
        self.optimal_ref = self.config.get('optimal_ref', 6.10)

        self._key = jax.random.PRNGKey(self.config.get('seed', 0))

        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.target_update_freq = self.config['target_update_freq']
        self.target_tau = self.config.get('target_tau', None)
        self.n_steps_per_update = self.config['n_steps_per_update']

        # Exploration. Boltzmann is RP-DQN's shipped default; 'epsilon' is kept as an
        # ablation. Both decay linearly over exploration_fraction of the run, and both
        # are reported through the same 'epsilon' column in metrics.log.
        self.exploration = self.config.get('exploration', 'boltzmann')
        if self.exploration == 'boltzmann':
            self._expl_start = self.config['temp_start']
            self._expl_end = self.config['temp_end']
        else:
            self._expl_start = self.config['epsilon_start']
            self._expl_end = self.config['epsilon_end']
        self.epsilon = self._expl_start
        self.exploration_fraction = self.config.get('exploration_fraction', 0.5)

        self.network = _QNetwork(
            n_actions=self.n_actions,
            embed_dim=self.config['embed_dim'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            ff_dim=self.config['ff_dim'],
            use_layer_norm=self.config['use_layer_norm'],
        )

        self._key, init_key = jax.random.split(self._key)
        _, probe_state = self.env.reset(init_key)
        self.params = self.network.init(init_key, self.env.get_obs(probe_state))
        self.target_params = jax.tree.map(jnp.copy, self.params)

        # RP-DQN clips at global norm 1.0 (AttentionModel.py: self.max_grad_norm = 1).
        # We were at 10.0 -- ten times looser, on a transformer, which is exactly where
        # clipping matters most. Kept configurable so the difference can be ablated.
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.get('max_grad_norm', 1.0)),
            optax.adam(self.config['alpha']),
        )
        self.opt_state = self.optimizer.init(self.params)

        per = self.config.get('PER_parameters', {})
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config['buffer_capacity'],
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.4),
            max_priority=per.get('maximum_priority', 1.0),
            uniform=(self.config['algorithm'] == 'dqn-uniform'),
        )

        self.total_steps = 0
        self.episode_returns: list = []
        self.episode_lengths: list = []
        self._current_episode = 0

        self._build_jit_fns()

    # ------------------------------------------------------------------

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
                q = jax.vmap(network.apply, in_axes=(None, 0))(p, states)
                q_taken = q[jnp.arange(q.shape[0]), actions]
                next_q_target = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
                if double_dqn:
                    next_q_online = jax.vmap(network.apply, in_axes=(None, 0))(p, next_states)
                    next_actions = jnp.argmax(jax.lax.stop_gradient(next_q_online), axis=-1)
                    max_next_q = next_q_target[jnp.arange(next_q_target.shape[0]), next_actions]
                else:
                    max_next_q = next_q_target.max(axis=-1)
                # max_next_q is a large finite negative at a fully-masked row, never
                # -inf, so the (1 - dones) gate cannot produce NaN.
                targets = rewards + gamma * max_next_q * (1.0 - dones)
                td_errors = targets - q_taken
                loss = jnp.mean(weights * td_errors ** 2)
                return loss, td_errors

            (_, td_errors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, td_errors

        @jax.jit
        def polyak(target_params, params, tau):
            return jax.tree.map(lambda t, p: tau * t + (1.0 - tau) * p, target_params, params)

        self._greedy_action = greedy_action
        self._update_step = update_step
        self._polyak = polyak

    def select_action(self, obs) -> int:
        obs = jnp.asarray(obs, dtype=jnp.float32)
        if self.exploration == 'boltzmann':
            self._key, k = jax.random.split(self._key)
            q = self.network.apply(self.params, obs)
            return int(jax.random.categorical(k, q / max(self.epsilon, 1e-6)))
        if np.random.uniform() < self.epsilon:
            # Uniform over LEGAL actions only -- a masked action is a no-op the env
            # would have to special-case, and it would waste a slot in the buffer.
            mask = np.array(self.env.action_mask_from_obs(obs))
            return int(np.random.choice(np.flatnonzero(mask)))
        return int(self._greedy_action(self.params, obs))

    def _update(self):
        # buffer_min_samples was in the config but never read: we started learning
        # as soon as one batch existed (1024) where RP-DQN waits for 2^13 = 8192
        # (AttentionModel.py gates on can_provide_sample(buffer_min_samples)).
        # Starting 8x earlier means the first updates run on a nearly-empty,
        # highly correlated buffer.
        min_samples = max(self.batch_size, self.config.get('buffer_min_samples', 0))
        if not self.buffer.can_sample(min_samples):
            return
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

        # RP-DQN soft-updates the target every gradient step (target_network_update:
        # 0.999). The hard sync stays available for ablations via target_tau=None.
        if self.target_tau is not None:
            self.target_params = self._polyak(
                self.target_params, self.params, jnp.float32(self.target_tau)
            )

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    # ------------------------------------------------------------------

    def _quality(self, mean_tour_len: float) -> float:
        """Tour length -> an increasing quality ratio the claim-2 pipeline can threshold."""
        if mean_tour_len <= 0:
            return 0.0
        return float(self.optimal_ref / mean_tour_len)

    def evaluate(self, n_episodes: int = 512) -> dict:
        """Greedy decode on fresh instances. Overridden by the vectorized subclass."""
        total_return = 0.0
        steps = 0
        for _ in range(n_episodes):
            self._key, rk = jax.random.split(self._key)
            obs, state = self.env.reset(rk)
            done = False
            ep_steps = 0
            while not done and ep_steps < self.env.max_steps:
                action = int(self._greedy_action(self.params, jnp.asarray(obs, jnp.float32)))
                self._key, sk = jax.random.split(self._key)
                obs, state, reward, done, _ = self.env.step(sk, state, jnp.int32(action))
                total_return += float(reward)
                ep_steps += 1
                done = bool(done)
            steps += ep_steps
        avg_return = total_return / max(1, n_episodes)
        return {
            'win_rate': self._quality(-avg_return),
            'avg_length': steps / max(1, n_episodes),
            'avg_return': avg_return,
        }

    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'target_params': self.target_params,
                'opt_state': self.opt_state,
                'config': self.config,
                'episode_returns': self.episode_returns,
                'episode_lengths': self.episode_lengths,
                'total_steps': self.total_steps,
            }, f)

    @classmethod
    def from_checkpoint(cls, path: str, config_overrides: Optional[dict] = None):
        """Rebuild an agent from a saved run -- Gate 1 / Gate 2 both need this."""
        with open(path, 'rb') as f:
            blob = pickle.load(f)
        config = dict(blob['config'])
        if config_overrides:
            config.update(config_overrides)
        agent = cls(config)
        agent.params = blob['params']
        agent.target_params = blob['target_params']
        agent.opt_state = blob.get('opt_state', agent.opt_state)
        agent.episode_returns = blob.get('episode_returns', [])
        agent.episode_lengths = blob.get('episode_lengths', [])
        agent.total_steps = blob.get('total_steps', 0)
        return agent
