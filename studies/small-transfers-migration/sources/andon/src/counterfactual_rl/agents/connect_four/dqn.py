"""
Connect4DQN — JAX/Flax DQN agent with Prioritized Experience Replay for Connect Four.

Standalone implementation adapted from chess/dqn.py. Uses pgx directly (no env wrapper)
since Connect Four's (6,7,2) observation is already a clean boolean array.

  - n_agents = 1 (we are always player 0)
  - action is a scalar int — column index 0-6
  - observation is flat (84,) = 6×7×2 reshaped
  - reward is sparse ±1 at game end only
  - done is a plain bool
  - pgx state stored for consequence rollouts

Episode collection uses jax.lax.scan + jax.vmap over N_ENVS parallel environments.
Each scan step = agent move + opponent random move = 1 stored transition.

Epsilon decay: chunk-based only (exploration_fraction × n_chunks).
No episode-count-based decay — episode count explodes with 256 parallel envs.
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

from .policies import Connect4QNetwork
from ..chess.array_buffer import ChessArrayReplayBuffer
from ..shared.metrics import MetricsLogger

C4_ACTIONS = 7


class Connect4DQN:
    """
    JAX DQN agent for Connect Four (single-agent, player 0 vs random opponent).

    Episode collection is fully vectorized: N_ENVS parallel environments run
    simultaneously via jax.vmap, with each episode collected via jax.lax.scan.
    No Python loop per step — the GPU runs entire chunks on-device.
    """

    def __init__(self, env_info: Dict, config: Optional[Dict] = None):
        from .config import DEFAULT_CONNECT4_CONFIG
        self.config = DEFAULT_CONNECT4_CONFIG.copy()
        if config:
            self.config.update(config)

        self.env_info = env_info
        self._key = jax.random.PRNGKey(self.config.get('seed', 0))

        self.obs_dim = env_info['obs_dim']                      # 84
        self.num_agents = env_info['num_agents']                # 1
        self.actions_per_agent = env_info['actions_per_agent']  # 7

        # pgx environment (used directly, no wrapper)
        self.pgx_env = pgx.make('connect_four')

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.05)
        self.exploration_fraction = self.config.get('exploration_fraction', 0.5)
        self.epsilon = self.epsilon_start
        self.alpha = self.config.get('alpha', 0.0001)
        self.batch_size = self.config.get('B', 64)
        self.target_update_freq = self.config.get('C', 500)
        self.n_steps_for_Q_update = self.config.get('n_steps_for_Q_update', 64)

        # Vectorized collection parameters
        self.n_envs = self.config.get('n_envs', 256)
        self.collect_steps = self.config.get('collect_steps', 42)

        # Network: conv front-end -> MLP -> (1, 7)
        self.network = Connect4QNetwork(
            hidden_dim=self.config.get('hidden_dim', 256),
            use_layer_norm=self.config.get('use_layer_norm', True),
        )
        self._key, init_key = jax.random.split(self._key)
        dummy = jnp.zeros(self.obs_dim)
        self.params = self.network.init(init_key, dummy)
        self.target_params = jax.tree.map(jnp.copy, self.params)

        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(self.alpha),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer
        per = self.config.get('PER_parameters', {})
        self.buffer = ChessArrayReplayBuffer(
            capacity=self.config.get('M', 100_000),
            obs_dim=self.obs_dim,
            mask_dim=C4_ACTIONS,
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.25),
            max_priority=per.get('maximum_priority', 1.0),
            uniform=(self.config.get('algorithm') == 'dqn-uniform'),
        )

        self.total_steps = 0
        self.episode_returns = []
        self.episode_lengths = []
        self._current_episode = 0

        self._build_jit_fns()
        self._build_vectorized_collect_fn()
        self._build_vectorized_eval_fn()

    def _build_jit_fns(self):
        network = self.network
        gamma = self.gamma

        @jax.jit
        def greedy_action(params, obs, masks):
            q = network.apply(params, obs)          # (1, 7)
            masked_q = jnp.where(masks, q, -jnp.inf)
            return jnp.argmax(masked_q, axis=-1)    # (1,)

        @jax.jit
        def update_step(params, target_params, opt_state,
                        states, actions, rewards, next_states, dones, next_masks, weights):
            def loss_fn(p):
                q_values = jax.vmap(network.apply, in_axes=(None, 0))(p, states)
                # q_values: (B, 1, 7)
                q_taken = jnp.take_along_axis(
                    q_values, actions[:, :, None], axis=-1
                ).squeeze(-1)                       # (B, 1)
                q_taken = q_taken.sum(axis=-1)      # (B,)

                next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
                next_q = jnp.where(next_masks, next_q, -jnp.inf)
                max_next_q = next_q.max(axis=-1).sum(axis=-1)  # (B,)
                targets = rewards + gamma * jnp.where(dones > 0.5, jnp.float32(0.0), max_next_q)

                td_errors = targets - q_taken
                abs_err = jnp.abs(td_errors)
                huber = jnp.where(abs_err <= 1.0, 0.5 * td_errors ** 2, abs_err - 0.5)
                loss = jnp.mean(weights * huber)
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

        Each scan step:
            1. Reset if previous episode ended
            2. Epsilon-greedy agent action (player 0)
            3. Agent's pgx.step
            4. Opponent's random pgx.step (player 1)
            5. Emit transition + saved pgx.State

        Stored as self._collect_fn. Called via _run_collect_chunk().
        """
        pgx_env = self.pgx_env
        network = self.network
        T = self.collect_steps
        opponent = self.config.get('opponent', 'random')
        if opponent == 'rule_based':
            from .opponent import rule_based_action
        elif opponent == 'mcts':
            from functools import partial
            from .opponent_mcts import mcts_action as _mcts_fn
            _mcts_action = partial(_mcts_fn, n_sims=self.config.get('mcts_n_sims', 32))

        def _collect_single_env(params, epsilon, init_state, env_key):
            def _step_fn(carry, step_key):
                state, already_done, cum_return, ep_length = carry
                k_reset, k_agent, k_eps, k_opp, k_pre = jax.random.split(step_key, 5)

                # Reset if previous episode ended
                reset_state = pgx_env.init(k_reset)
                state = jax.tree.map(
                    lambda r, s: jnp.where(already_done, r, s), reset_state, state
                )
                already_done = jnp.bool_(False)

                # Agent is always player 0. If pgx assigned opponent as first mover
                # (current_player == 1), let the opponent take one move so that the
                # main agent→opponent loop always starts on the agent's turn.
                # For ongoing games s2.current_player == 0 always, so need_opp_pre
                # is False and the jnp.where is a no-op.
                agent_player = jnp.int32(0)
                need_opp_pre = (state.current_player != agent_player)
                if opponent == 'rule_based':
                    raw_pre = rule_based_action(state, k_pre)
                elif opponent == 'mcts':
                    raw_pre = _mcts_action(state, k_pre)
                else:
                    pre_f = state.legal_action_mask.astype(jnp.float32)
                    pre_total = pre_f.sum()
                    safe_pre = jnp.where(pre_total > 0, pre_f / pre_total,
                                         jnp.ones(C4_ACTIONS) / C4_ACTIONS)
                    raw_pre = jax.random.choice(k_pre, C4_ACTIONS, p=safe_pre)
                state_after_pre = pgx_env.step(state, raw_pre)
                state = jax.tree.map(
                    lambda a, b: jnp.where(need_opp_pre, a, b), state_after_pre, state
                )

                # state.current_player == 0 == agent_player from here on.
                saved_state = state
                obs = state.observation.reshape(-1)    # (84,)
                mask = state.legal_action_mask          # (7,)

                # Epsilon-greedy action
                mask_f = mask.astype(jnp.float32)
                total = mask_f.sum()
                safe_p = jnp.where(total > 0, mask_f / total, jnp.ones(C4_ACTIONS) / C4_ACTIONS)
                random_action = jax.random.choice(k_agent, C4_ACTIONS, p=safe_p)
                q = network.apply(params, obs)          # (1, 7)
                greedy_a = jnp.argmax(jnp.where(mask, q[0], -jnp.inf))
                agent_action = jnp.where(jax.random.uniform(k_eps) < epsilon, random_action, greedy_a)

                # Agent's move
                s1 = pgx_env.step(state, agent_action)
                r1 = s1.rewards[agent_player]
                done1 = s1.terminated | s1.truncated

                # Opponent move — branch selected at build time, baked into XLA
                if opponent == 'rule_based':
                    raw_opp = rule_based_action(s1, k_opp)
                elif opponent == 'mcts':
                    raw_opp = _mcts_action(s1, k_opp)
                else:
                    opp_f = s1.legal_action_mask.astype(jnp.float32)
                    opp_total = opp_f.sum()
                    safe_opp = jnp.where(opp_total > 0, opp_f / opp_total,
                                         jnp.ones(C4_ACTIONS) / C4_ACTIONS)
                    raw_opp = jax.random.choice(k_opp, C4_ACTIONS, p=safe_opp)
                opp_action = jnp.where(done1, jnp.int32(0), raw_opp)
                s2 = pgx_env.step(s1, opp_action)
                r2 = jnp.where(done1, jnp.float32(0.0), s2.rewards[agent_player])
                done2 = done1 | s2.terminated | s2.truncated

                reward = r1 + r2
                next_obs = s2.observation.reshape(-1)
                next_mask = s2.legal_action_mask

                new_cum = cum_return + reward
                new_len = ep_length + 1

                ep_return_out = jnp.where(done2, new_cum, jnp.float32(0.0))
                ep_length_out = jnp.where(done2, new_len, jnp.int32(0))

                new_cum = jnp.where(done2, jnp.float32(0.0), new_cum)
                new_len = jnp.where(done2, jnp.int32(0), new_len)

                output = (
                    agent_action,    # scalar int32
                    reward,          # scalar float32
                    done2,           # scalar bool
                    obs,             # (84,)
                    next_obs,        # (84,)
                    mask,            # (7,)
                    next_mask,       # (7,)
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

    def _build_vectorized_eval_fn(self, max_eval_steps: int = 84):
        """
        Build JIT-compiled vectorized evaluation function (random opponent only).

        max_eval_steps=84: covers 2 full games worst-case (42 half-moves each).
        Each env plays one greedy game via lax.scan.
        """
        pgx_env = self.pgx_env
        network = self.network
        opponent = self.config.get('opponent', 'random')
        if opponent == 'rule_based':
            from .opponent import rule_based_action
        elif opponent == 'mcts':
            from functools import partial
            from .opponent_mcts import mcts_action as _mcts_fn
            _mcts_action = partial(_mcts_fn, n_sims=self.config.get('mcts_n_sims', 32))

        def eval_single_env(params, init_state, env_key):
            # Agent is always player 0. If opponent was assigned first mover by pgx,
            # let them play one move before the main agent→opponent scan loop.
            agent_player = jnp.int32(0)
            need_opp_pre = (init_state.current_player != agent_player)
            k_pre, scan_key = jax.random.split(env_key)
            step_keys = jax.random.split(scan_key, max_eval_steps)

            if opponent == 'rule_based':
                raw_pre = rule_based_action(init_state, k_pre)
            elif opponent == 'mcts':
                raw_pre = _mcts_action(init_state, k_pre)
            else:
                pre_f = init_state.legal_action_mask.astype(jnp.float32)
                pre_total = pre_f.sum()
                safe_pre = jnp.where(pre_total > 0, pre_f / pre_total,
                                     jnp.ones(C4_ACTIONS) / C4_ACTIONS)
                raw_pre = jax.random.choice(k_pre, C4_ACTIONS, p=safe_pre)
            state_after_pre = pgx_env.step(init_state, raw_pre)
            game_state = jax.tree.map(
                lambda a, b: jnp.where(need_opp_pre, a, b), state_after_pre, init_state
            )

            def step_fn(carry, step_key):
                state, done, cum_return, ep_len = carry

                # Greedy agent action (frozen to 0 once done)
                q = network.apply(params, state.observation.reshape(-1))  # (1, 7)
                greedy = jnp.argmax(jnp.where(state.legal_action_mask, q[0], -jnp.inf))
                agent_action = jnp.where(done, jnp.int32(0), greedy)

                s1 = pgx_env.step(state, agent_action)
                r1 = jnp.where(done, jnp.float32(0.0), s1.rewards[agent_player])
                done1 = done | s1.terminated | s1.truncated

                # Opponent move — branch selected at build time, baked into XLA
                if opponent == 'rule_based':
                    raw_opp = rule_based_action(s1, step_key)
                elif opponent == 'mcts':
                    raw_opp = _mcts_action(s1, step_key)
                else:
                    opp_f = s1.legal_action_mask.astype(jnp.float32)
                    opp_total = opp_f.sum()
                    safe_opp = jnp.where(opp_total > 0, opp_f / opp_total,
                                         jnp.ones(C4_ACTIONS) / C4_ACTIONS)
                    raw_opp = jax.random.choice(step_key, C4_ACTIONS, p=safe_opp)
                opp_action = jnp.where(done1, jnp.int32(0), raw_opp)

                s2 = pgx_env.step(s1, opp_action)
                r2 = jnp.where(done1, jnp.float32(0.0), s2.rewards[agent_player])
                done2 = done1 | s2.terminated | s2.truncated

                new_cum = cum_return + r1 + r2
                new_len = jnp.where(done, ep_len, ep_len + 1)
                return (s2, done2, new_cum, new_len), None

            init_carry = (game_state, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0))
            final_carry, _ = jax.lax.scan(step_fn, init_carry, step_keys)
            return final_carry[2], final_carry[3]  # cum_return, ep_len

        self._eval_fn = jax.jit(jax.vmap(eval_single_env, in_axes=(None, 0, 0)))

    def _run_collect_chunk(self, N_ENVS: int, T: int):
        self._key, collect_key = jax.random.split(self._key)
        env_keys = jax.random.split(collect_key, N_ENVS)
        init_states = jax.vmap(self.pgx_env.init)(env_keys)
        outputs = self._collect_fn(
            self.params, jnp.float32(self.epsilon), init_states, env_keys
        )
        jax.block_until_ready(outputs[0])
        return outputs

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        data, indices, is_weights = self.buffer.sample(self.batch_size)

        states      = jnp.array(data['s'])
        next_states = jnp.array(data["s'"])
        actions     = jnp.array(data['a'],    dtype=jnp.int32)
        rewards     = jnp.array(data['r'],    dtype=jnp.float32)
        dones       = jnp.array(data['done'], dtype=jnp.float32)
        next_masks  = jnp.array(data['next_masks'], dtype=jnp.bool_)
        weights     = jnp.array(is_weights,   dtype=jnp.float32)

        self.params, self.opt_state, loss, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, next_states, dones, next_masks, weights,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

    def _update_target_network(self):
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def _add_chunk_to_buffer(self, outputs, N_ENVS: int, T: int):
        n = N_ENVS * T
        self.buffer.add_batch(
            obs        = np.array(outputs[3]).reshape(n, -1),
            next_obs   = np.array(outputs[4]).reshape(n, -1),
            actions    = np.array(outputs[0]).reshape(n),
            rewards    = np.array(outputs[1]).reshape(n),
            dones      = np.array(outputs[2]).reshape(n),
            masks      = np.array(outputs[5]).reshape(n, 1, -1),
            next_masks = np.array(outputs[6]).reshape(n, 1, -1),
        )

    def learn(self, n_chunks: Optional[int] = None, verbose: bool = True) -> 'Connect4DQN':
        n_chunks = n_chunks or self.config['n_chunks']
        N_ENVS = self.n_envs
        T = self.collect_steps
        save_every = self.config.get('save_every', 10)
        eval_interval = self.config.get('eval_interval', 1)
        eval_episodes = self.config.get('eval_episodes', 200)

        # Chunk-based epsilon decay (no episode counting)
        decay_chunks = max(1, int(self.exploration_fraction * n_chunks))

        log_env_info = {**self.env_info, 'scenario': 'connect_four'}
        self.metrics_logger = MetricsLogger(
            backend='JAX (Connect Four)',
            config=self.config,
            env_info=log_env_info,
            n_episodes=n_chunks,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            run_root=os.path.dirname(os.path.abspath(__file__)),
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_win_rate = -1.0

        n_ckpts = self.config.get('n_checkpoints', 10)
        ckpt_interval = max(1, n_chunks // n_ckpts) if n_ckpts > 0 else 0
        ckpt_dir = os.path.join(self.metrics_logger.dir, 'checkpoints')
        if ckpt_interval > 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        if verbose:
            print(f"Training Connect4DQN [VECTORIZED]")
            print(f"  n_envs={N_ENVS}  collect_steps={T}  ({N_ENVS*T} transitions/chunk)")
            print(f"  Epsilon: {self.epsilon_start} → {self.epsilon_end} over {decay_chunks} chunks")
            print(f"  JAX backend: {jax.default_backend()}  |  Devices: {jax.devices()}")
            run_dir = self.metrics_logger.dir
            print(f"  Run dir: {run_dir}")

        pbar = tqdm(range(n_chunks), disable=not verbose)
        for chunk_idx in pbar:
            self._current_episode = chunk_idx
            timer.begin_episode(chunk_idx)

            with timer('collect', episode=chunk_idx):
                outputs = self._run_collect_chunk(N_ENVS, T)

            with timer('buffer.add', episode=chunk_idx):
                ep_ret_np   = np.array(outputs[8])
                ep_len_np   = np.array(outputs[9])
                ep_ended_np = np.array(outputs[2])
                self._add_chunk_to_buffer(outputs, N_ENVS, T)
                del outputs

            n_transitions = N_ENVS * T
            prev_steps = self.total_steps
            self.total_steps += n_transitions

            with timer('update', episode=chunk_idx):
                n_updates = (self.total_steps // self.n_steps_for_Q_update) - \
                            (prev_steps // self.n_steps_for_Q_update)
                q_start = prev_steps // self.n_steps_for_Q_update
                target_freq_q = max(1, self.target_update_freq // self.n_steps_for_Q_update)
                for i in range(n_updates):
                    self._update()
                    if (q_start + i) % target_freq_q == 0:
                        self._update_target_network()

            for env_i in range(N_ENVS):
                for t in range(T):
                    if ep_ended_np[env_i, t]:
                        self.episode_returns.append(float(ep_ret_np[env_i, t]))
                        self.episode_lengths.append(int(ep_len_np[env_i, t]))

            # Chunk-based epsilon decay
            decay_progress = min(1.0, chunk_idx / decay_chunks)
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

            if len(self.episode_returns) >= 100:
                avg_r = np.mean(self.episode_returns[-100:])
                pbar.set_description(
                    f"chunk={chunk_idx} | AvgR(100)={avg_r:.2f} | ε={self.epsilon:.3f}"
                )

            if (chunk_idx + 1) % save_every == 0:
                self.save(last_path)

            if ckpt_interval > 0 and (chunk_idx + 1) % ckpt_interval == 0:
                self.save(os.path.join(ckpt_dir, f'ckpt_{chunk_idx+1:07d}.pkl'))

            if eval_interval and (chunk_idx + 1) % eval_interval == 0:
                with timer('eval', episode=chunk_idx):
                    metrics = self.evaluate(n_episodes=eval_episodes, seed=chunk_idx)
                model_updates = self.total_steps // self.n_steps_for_Q_update
                self.metrics_logger.log_eval(chunk_idx + 1, model_updates, self.epsilon, metrics)
                if metrics['win_rate'] > best_win_rate:
                    best_win_rate = metrics['win_rate']
                    self.save(best_path)
                    if verbose:
                        print(f"\nNew best win rate: {best_win_rate:.1%}")

            timer.flush_episode()

        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"Training complete. Run saved to {self.metrics_logger.dir}")
        return self

    def evaluate(self, n_episodes: int = 200, seed: int = 42) -> Dict:
        """Greedy evaluation — agent plays ~50% as first mover, ~50% as second."""
        key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(key, n_episodes)
        _, subkey = jax.random.split(key)
        eval_keys = jax.random.split(subkey, n_episodes)

        init_states = jax.vmap(self.pgx_env.init)(init_keys)
        # Which games have agent going first (current_player == 0 == agent_player)?
        agent_goes_first = np.array(init_states.current_player) == 0

        returns, lengths = self._eval_fn(self.params, init_states, eval_keys)
        jax.block_until_ready(returns)

        returns_np = np.array(returns)
        lengths_np = np.array(lengths)
        wins   = int((returns_np > 0).sum())
        losses = int((returns_np < 0).sum())
        draws  = n_episodes - wins - losses

        first_ret  = returns_np[agent_goes_first]
        second_ret = returns_np[~agent_goes_first]
        wr_first  = float((first_ret  > 0).mean()) if len(first_ret)  > 0 else float('nan')
        wr_second = float((second_ret > 0).mean()) if len(second_ret) > 0 else float('nan')

        return {
            'win_rate':        wins / n_episodes,
            'draw_rate':       draws / n_episodes,
            'loss_rate':       losses / n_episodes,
            'avg_return':      float(returns_np.mean()),
            'avg_length':      float(lengths_np.mean()),
            'win_rate_first':  wr_first,
            'win_rate_second': wr_second,
        }

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
            'epsilon':         self.epsilon,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self._build_jit_fns()
        self._build_vectorized_collect_fn()
        self._build_vectorized_eval_fn()
