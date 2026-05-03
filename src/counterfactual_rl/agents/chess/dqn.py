"""
ChessDQN — JAX/Flax DQN agent with Prioritized Experience Replay for Gardner chess.

Standalone implementation (does not inherit from SMAX DQN) to avoid the jaxmarl
dependency chain. Mirrors the SMAX DQN structure but adapted for single-agent chess:

  - n_agents = 1
  - action is a scalar int (not a per-agent dict)
  - observation is flat (2875,) from chess_env
  - reward is sparse ±1 at game end only
  - done is a plain bool (not dones["__all__"])
  - jax_obs NOT stored in buffer (pgx state contains obs)

Episode collection uses jax.lax.scan + jax.vmap over N_ENVS parallel environments,
keeping data on-device between steps (no Python loop per step).
"""

import os
import pickle
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import pgx

from .policies import ChessQNetwork
from counterfactual_rl.envs.chess import GardnerChessEnv, CHESS_ACTIONS
from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.metrics import MetricsLogger


class ChessDQN:
    """
    JAX DQN agent for Gardner chess (single-agent, white vs random opponent).

    Episode collection is fully vectorized: N_ENVS parallel environments run
    simultaneously via jax.vmap, with each episode collected via jax.lax.scan.
    No Python loop per step — the GPU runs entire chunks on-device.

    Usage:
        env, key, env_info = create_chess_env()
        agent = ChessDQN(env, env_info)
        agent.learn()
        metrics = agent.evaluate()
    """

    def __init__(self, env: GardnerChessEnv, env_info: Dict, config: Optional[Dict] = None):
        from .config import DEFAULT_CHESS_CONFIG
        self.config = DEFAULT_CHESS_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env = env
        self.env_info = env_info
        self._key = jax.random.PRNGKey(self.config.get('seed', 0))

        self.obs_dim = env_info['obs_dim']                       # 2875
        self.num_agents = env_info['num_agents']                 # 1
        self.actions_per_agent = env_info['actions_per_agent']   # 1225

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.05)
        self.epsilon_decay_episodes = self.config.get('epsilon_decay_episodes', 20000)
        self.epsilon = self.epsilon_start
        self.alpha = self.config.get('alpha', 0.0001)
        self.batch_size = self.config.get('B', 64)
        self.target_update_freq = self.config.get('C', 1000)
        self.n_steps_for_Q_update = self.config.get('n_steps_for_Q_update', 4)

        # Vectorized collection parameters
        self.n_envs = self.config.get('n_envs', 256)
        self.collect_steps = self.config.get('collect_steps', 256)

        # Network: conv front-end -> MLP -> (1, 1225)
        self.network = ChessQNetwork(
            hidden_dim=self.config.get('hidden_dim', 512),
            use_layer_norm=self.config.get('use_layer_norm', True),
        )
        self._key, init_key = jax.random.split(self._key)
        dummy = jnp.zeros(self.obs_dim)
        self.params = self.network.init(init_key, dummy)
        self.target_params = jax.tree.map(jnp.copy, self.params)

        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(100.0),
            optax.adam(self.alpha),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer
        per = self.config.get('PER_parameters', {})
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.get('M', 200000),
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.4),
            max_priority=per.get('maximum_priority', 1.0),
            uniform=(self.config.get('algorithm') == 'dqn-uniform'),
        )

        self.total_steps = 0
        self.episode_returns = []
        self.episode_lengths = []
        self._current_episode = 0

        self._build_jit_fns()
        self._build_vectorized_collect_fn()

    def _build_jit_fns(self):
        network = self.network
        gamma = self.gamma

        @jax.jit
        def greedy_action(params, obs, masks):
            """
            Args:
                obs:   (2875,) float32
                masks: (1, 1225) bool
            Returns:
                (1,) int — white's greedy action
            """
            q = network.apply(params, obs)          # (1, 1225)
            masked_q = jnp.where(masks, q, -jnp.inf)
            return jnp.argmax(masked_q, axis=-1)    # (1,)

        @jax.jit
        def update_step(params, target_params, opt_state,
                        states, actions, rewards, next_states, dones, next_masks, weights):
            """
            Shapes:
                states:      (B, 2875)
                actions:     (B, 1)    int32
                rewards:     (B,)
                next_states: (B, 2875)
                dones:       (B,)      float32
                next_masks:  (B, 1, 1225) bool
                weights:     (B,)
            """
            def loss_fn(p):
                q_values = jax.vmap(network.apply, in_axes=(None, 0))(p, states)
                # q_values: (B, 1, 1225)
                q_taken = jnp.take_along_axis(
                    q_values, actions[:, :, None], axis=-1
                ).squeeze(-1)               # (B, 1)
                q_taken = q_taken.sum(axis=-1)  # (B,)

                next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
                next_q = jnp.where(next_masks, next_q, -jnp.inf)
                max_next_q = next_q.max(axis=-1).sum(axis=-1)  # (B,)
                targets = rewards + gamma * max_next_q * (1.0 - dones)

                td_errors = targets - q_taken
                loss = jnp.mean(weights * td_errors ** 2)
                return loss, td_errors

            (loss, td_errors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, td_errors

        self._greedy_action = greedy_action
        self._update_step = update_step

    def _build_vectorized_collect_fn(self):
        """
        Build and JIT-compile the vectorized episode collection function.

        Architecture:
            jax.jit(jax.vmap(single_env_collect, in_axes=(None, None, 0, 0)))

        Each single_env_collect runs jax.lax.scan over collect_steps steps.
        Inside the scan:
            1. Reset if previous episode ended (jnp.where + tree_map)
            2. Epsilon-greedy white action (JAX random, no Python RNG)
            3. White's pgx.step
            4. Opponent's random pgx.step
            5. Emit transition + saved pgx.State for consequence rollouts

        Stored as self._collect_fn. Called via _run_collect_chunk().
        """
        pgx_env = self.env.pgx_env
        network = self.network
        T = self.collect_steps

        def _collect_single_env(params, epsilon, init_state, env_key):
            # _step_fn defined inside so it captures params and epsilon as closures
            def _step_fn(carry, step_key):
                state, already_done, cum_return, ep_length = carry
                k_reset, k_white, k_eps, k_opp = jax.random.split(step_key, 4)

                # Reset if previous episode ended
                reset_state = pgx_env.init(k_reset)
                state = jax.tree.map(
                    lambda r, s: jnp.where(already_done, r, s), reset_state, state
                )
                already_done = jnp.bool_(False)

                # Save state before action (current_player == 0 invariant)
                saved_state = state
                obs = state.observation.reshape(-1)    # (2875,)
                mask = state.legal_action_mask          # (1225,)

                # Epsilon-greedy action (JAX random — safe inside lax.scan)
                mask_f = mask.astype(jnp.float32)
                total = mask_f.sum()
                safe_p = jnp.where(total > 0, mask_f / total, jnp.ones(CHESS_ACTIONS) / CHESS_ACTIONS)
                random_action = jax.random.choice(k_white, CHESS_ACTIONS, p=safe_p)
                q = network.apply(params, obs)          # (1, 1225)
                greedy_action = jnp.argmax(jnp.where(mask, q[0], -jnp.inf))
                white_action = jnp.where(jax.random.uniform(k_eps) < epsilon, random_action, greedy_action)

                # White's move
                s1 = pgx_env.step(state, white_action)
                r1 = s1.rewards[0]
                done1 = s1.terminated | s1.truncated

                # Opponent random move (pgx freezes terminal states — safe to call on done s1)
                opp_f = s1.legal_action_mask.astype(jnp.float32)
                opp_total = opp_f.sum()
                safe_opp = jnp.where(opp_total > 0, opp_f / opp_total,
                                     jnp.ones(CHESS_ACTIONS) / CHESS_ACTIONS)
                opp_action = jnp.where(
                    done1,
                    jnp.int32(0),
                    jax.random.choice(k_opp, CHESS_ACTIONS, p=safe_opp),
                )
                s2 = pgx_env.step(s1, opp_action)
                r2 = jnp.where(done1, jnp.float32(0.0), s2.rewards[0])
                done2 = done1 | s2.terminated | s2.truncated

                reward = r1 + r2
                next_obs = s2.observation.reshape(-1)
                next_mask = s2.legal_action_mask

                new_cum = cum_return + reward
                new_len = ep_length + 1

                # Emit episode stats only at episode boundary (nonzero = completed episode)
                ep_return_out = jnp.where(done2, new_cum, jnp.float32(0.0))
                ep_length_out = jnp.where(done2, new_len, jnp.int32(0))

                # Reset accumulators after episode ends
                new_cum = jnp.where(done2, jnp.float32(0.0), new_cum)
                new_len = jnp.where(done2, jnp.int32(0), new_len)

                output = (
                    white_action,    # scalar int32
                    reward,          # scalar float32
                    done2,           # scalar bool
                    obs,             # (2875,)
                    next_obs,        # (2875,)
                    mask,            # (1225,)
                    next_mask,       # (1225,)
                    saved_state,     # pgx.State — for consequence rollouts
                    ep_return_out,   # float32 — nonzero only at episode end
                    ep_length_out,   # int32   — nonzero only at episode end
                )
                return (s2, done2, new_cum, new_len), output

            step_keys = jax.random.split(env_key, T)
            init_carry = (init_state, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0))
            _, outputs = jax.lax.scan(_step_fn, init_carry, step_keys)
            return outputs

        _vmapped = jax.vmap(_collect_single_env, in_axes=(None, None, 0, 0))
        self._collect_fn = jax.jit(_vmapped)

    def _run_collect_chunk(self, N_ENVS: int, T: int):
        """
        Run one vectorized collection chunk: N_ENVS envs × T steps.

        Returns a tuple of JAX arrays, each with shape (N_ENVS, T, ...):
            (actions, rewards, ep_ended, obs, next_obs, masks, next_masks,
             saved_states, ep_returns, ep_lengths)

        saved_states is a pgx.State pytree with leaves of shape (N_ENVS, T, ...).
        ep_returns and ep_lengths are nonzero only at episode boundaries.
        """
        self._key, collect_key = jax.random.split(self._key)
        env_keys = jax.random.split(collect_key, N_ENVS)
        init_states = jax.vmap(self.env.pgx_env.init)(env_keys)
        outputs = self._collect_fn(
            self.params, jnp.float32(self.epsilon), init_states, env_keys
        )
        jax.block_until_ready(outputs[0])  # block on actions array before Python processing
        return outputs

    def select_action(self, obs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy action selection (used during evaluate()).

        Args:
            obs:          (2875,) float32
            action_masks: (1, 1225) bool

        Returns:
            (1,) int — white's chosen action
        """
        if np.random.uniform() < self.epsilon:
            valid = np.where(action_masks[0])[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0
            return np.array([action])
        else:
            return np.array(self._greedy_action(
                self.params,
                jnp.array(obs),
                jnp.array(action_masks, dtype=jnp.bool_),
            ))

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        transitions, indices, is_weights = self.buffer.sample(self.batch_size)

        states = jnp.array(np.array([d['s'] for d in transitions]))
        next_states = jnp.array(np.array([d["s'"] for d in transitions]))
        actions = jnp.array(np.array([d['a'] for d in transitions]), dtype=jnp.int32)
        rewards = jnp.array(np.array([d['r'] for d in transitions]), dtype=jnp.float32)
        dones = jnp.array(np.array([d['done'] for d in transitions]), dtype=jnp.float32)
        next_masks = jnp.array(np.array([d['next_masks'] for d in transitions]), dtype=jnp.bool_)
        weights = jnp.array(is_weights, dtype=jnp.float32)

        self.params, self.opt_state, loss, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, next_states, dones, next_masks, weights,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def _add_chunk_to_buffer(self, outputs, N_ENVS: int, T: int):
        """
        Convert collected chunk to numpy and add all transitions to buffer.
        Called by learn() and overridden by ChessConsequenceDQN to also store pgx states.
        """
        actions_np  = np.array(outputs[0])    # (N, T)
        rewards_np  = np.array(outputs[1])    # (N, T)
        ep_ended_np = np.array(outputs[2])    # (N, T) bool
        obs_np      = np.array(outputs[3])    # (N, T, 2875)
        next_obs_np = np.array(outputs[4])    # (N, T, 2875)
        masks_np      = np.array(outputs[5])  # (N, T, 1225)
        next_masks_np = np.array(outputs[6])  # (N, T, 1225)

        for env_i in range(N_ENVS):
            for t in range(T):
                self.buffer.add({
                    's':          obs_np[env_i, t],
                    'a':          actions_np[env_i, t:t+1],    # (1,) to match original
                    'r':          float(rewards_np[env_i, t]),
                    "s'":         next_obs_np[env_i, t],
                    'done':       bool(ep_ended_np[env_i, t]),
                    'masks':      masks_np[env_i, t:t+1],      # (1, 1225)
                    'next_masks': next_masks_np[env_i, t:t+1],
                })

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'ChessDQN':
        """
        Train using vectorized episode collection (lax.scan + vmap).

        n_episodes here means number of collection chunks, not individual episodes.
        Each chunk collects n_envs × collect_steps transitions.
        """
        n_chunks = n_episodes or self.config['n_episodes']
        N_ENVS = self.n_envs
        T = self.collect_steps
        save_every = self.config.get('save_every', 1000)
        eval_interval = self.config.get('eval_interval', None)
        eval_episodes = self.config.get('eval_episodes', 50)
        record_interval = self.config.get('record_interval', None)

        log_env_info = {**self.env_info, 'scenario': self.env_info.get('env_name', 'gardner_chess')}
        self.metrics_logger = MetricsLogger(
            backend='JAX (Chess)',
            config=self.config,
            env_info=log_env_info,
            n_episodes=n_chunks,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_win_rate = -1.0

        if verbose:
            print(f"Training ChessDQN on Gardner chess (vectorized)")
            print(f"  N_ENVS={N_ENVS} | collect_steps={T} | {N_ENVS*T} transitions/chunk")
            print(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_episodes} eps")
            print(f"  JAX backend: {jax.default_backend()}")

        pbar = tqdm(range(n_chunks), disable=not verbose)
        for chunk_idx in pbar:
            self._current_episode = chunk_idx
            timer.begin_episode(chunk_idx)

            with timer('collect', episode=chunk_idx):
                outputs = self._run_collect_chunk(N_ENVS, T)

            with timer('buffer.add', episode=chunk_idx):
                self._add_chunk_to_buffer(outputs, N_ENVS, T)

            n_transitions = N_ENVS * T
            prev_steps = self.total_steps
            self.total_steps += n_transitions

            with timer('update', episode=chunk_idx):
                n_updates = (self.total_steps // self.n_steps_for_Q_update) - \
                            (prev_steps // self.n_steps_for_Q_update)
                for _ in range(n_updates):
                    self._update()
                if (prev_steps // self.target_update_freq) < (self.total_steps // self.target_update_freq):
                    self._update_target_network()

            # Episode stats: ep_returns/ep_lengths are nonzero only at episode ends
            ep_ret_np = np.array(outputs[8])   # (N, T)
            ep_len_np = np.array(outputs[9])   # (N, T)
            ep_ended_np = np.array(outputs[2]) # (N, T)
            for env_i in range(N_ENVS):
                for t in range(T):
                    if ep_ended_np[env_i, t]:
                        self.episode_returns.append(float(ep_ret_np[env_i, t]))
                        self.episode_lengths.append(int(ep_len_np[env_i, t]))

            # Epsilon decay by episodes completed
            decay_progress = min(1.0, len(self.episode_returns) / max(1, self.epsilon_decay_episodes))
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

            if len(self.episode_returns) >= 100:
                avg_r = np.mean(self.episode_returns[-100:])
                pbar.set_description(
                    f"eps completed: {len(self.episode_returns)} | "
                    f"avg return (100ep): {avg_r:.2f} | eps: {self.epsilon:.3f}"
                )

            if (chunk_idx + 1) % save_every == 0:
                self.save(last_path)

            if eval_interval and (chunk_idx + 1) % eval_interval == 0:
                with timer('eval', episode=chunk_idx):
                    metrics = self.evaluate(n_episodes=eval_episodes)
                model_updates = self.total_steps // self.n_steps_for_Q_update
                self.metrics_logger.log_eval(chunk_idx + 1, model_updates, self.epsilon, metrics)
                if metrics['win_rate'] > best_win_rate:
                    best_win_rate = metrics['win_rate']
                    self.save(best_path)
                    if verbose:
                        print(f"\nNew best win rate: {best_win_rate:.1%}")

            if record_interval and (chunk_idx + 1) % record_interval == 0:
                self._record_game(chunk_idx)

            timer.flush_episode()

        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"Training complete. Run saved to {self.metrics_logger.dir}")
        return self

    def evaluate(self, n_episodes: int = 50, seed: int = 42) -> Dict:
        """
        Greedy evaluation using the Python episode loop (not vectorized — correctness over speed).

        Returns:
            {'win_rate', 'draw_rate', 'loss_rate', 'avg_return', 'avg_length', 'avg_allies_alive'}
        """
        saved_epsilon = self.epsilon
        self.epsilon = 0.0
        key = jax.random.PRNGKey(seed)
        wins = draws = losses = 0
        total_return = total_length = 0.0

        for _ in range(n_episodes):
            key, reset_key = jax.random.split(key)
            obs, state = self.env.reset(reset_key)
            masks = self.env.get_legal_mask(state)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                action = self.select_action(obs, masks)
                obs, state, r, done = self.env.step(state, int(action[0]))
                masks = self.env.get_legal_mask(state)
                ep_ret += r
                ep_len += 1
            if ep_ret > 0:
                wins += 1
            elif ep_ret < 0:
                losses += 1
            else:
                draws += 1
            total_return += ep_ret
            total_length += ep_len

        self.epsilon = saved_epsilon
        return {
            'win_rate':         wins / n_episodes,
            'draw_rate':        draws / n_episodes,
            'loss_rate':        losses / n_episodes,
            'avg_return':       total_return / n_episodes,
            'avg_length':       total_length / n_episodes,
            'avg_allies_alive': draws / n_episodes,  # MetricsLogger compat column
        }

    def _record_game(self, chunk_idx: int, seed: int = 99, max_steps: int = 200):
        """
        Play one greedy game and save an SVG animation to the run directory.

        Inlines play_game() from generate_gif.py (can't import it — circular dep).
        Output: <run_dir>/game_chunk{chunk_idx:06d}.svg
        """
        pgx_env = self.env.pgx_env
        key = jax.random.PRNGKey(seed + chunk_idx)  # unique seed per recording

        saved_eps = self.epsilon
        self.epsilon = 0.0

        key, reset_key = jax.random.split(key)
        obs, state = self.env.reset(reset_key)
        masks = self.env.get_legal_mask(state)
        raw_state = pgx_env.init(reset_key)
        states = [raw_state]

        done = False
        steps = 0
        while not done and steps < max_steps:
            action = self.select_action(obs, masks)
            white_action = int(action[0])

            raw_state = pgx_env.step(raw_state, jnp.int32(white_action))
            states.append(raw_state)
            if bool(np.array(raw_state.terminated | raw_state.truncated)):
                break

            key, opp_key = jax.random.split(key)
            legal = raw_state.legal_action_mask
            legal_f = legal.astype(jnp.float32)
            safe_p = legal_f / jnp.where(legal_f.sum() > 0, legal_f.sum(), 1.0)
            opp_action = jax.random.choice(opp_key, CHESS_ACTIONS, p=safe_p)
            raw_state = pgx_env.step(raw_state, opp_action)
            states.append(raw_state)

            next_obs, next_state, reward, done = self.env.step(state, white_action)
            done = done or bool(np.array(raw_state.terminated | raw_state.truncated))
            obs, state = next_obs, next_state
            masks = self.env.get_legal_mask(state)
            steps += 1

        self.epsilon = saved_eps

        out_path = os.path.join(self.metrics_logger.dir, f"game_chunk{chunk_idx:06d}.svg")
        pgx.save_svg_animation(states, out_path, frame_duration_seconds=0.8)

    def make_policy_fn(self):
        """Return a JIT-compatible greedy policy function for consequence rollouts."""
        network = self.network
        params = self.params

        def policy_fn(obs_flat, legal_mask_1d):
            q = network.apply(params, obs_flat)   # (1, 1225)
            return jnp.argmax(jnp.where(legal_mask_1d, q[0], -jnp.inf))

        return policy_fn

    def save(self, path: str):
        checkpoint = {
            'params':          jax.tree.map(np.array, self.params),
            'target_params':   jax.tree.map(np.array, self.target_params),
            'opt_state':       jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x,
                                            self.opt_state),
            'config':          self.config,
            'env_info':        self.env_info,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'total_steps':     self.total_steps,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.params = jax.tree.map(jnp.array, checkpoint['params'])
        self.target_params = jax.tree.map(jnp.array, checkpoint['target_params'])
        self.opt_state = jax.tree.map(
            lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
            checkpoint['opt_state'],
        )
        self.episode_returns = checkpoint.get('episode_returns', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.total_steps = checkpoint.get('total_steps', 0)
        self._build_jit_fns()
        self._build_vectorized_collect_fn()
