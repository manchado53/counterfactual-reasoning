"""
ChessConsequenceDQN — Algorithm 2 for Gardner chess.

Extends ChessDQN with consequence-weighted prioritized replay.
The algorithm is identical to the SMAX version; only the rollout engine
adapts to chess's alternating-turn structure:

  - Each "horizon step" = white move + opponent response (one full move pair)
  - first_action is a scalar int32 (not a per-agent array)
  - actions_array shape is (B, K) — not (B, K, n_agents)
  - JAX state stored in buffer; jax_obs NOT stored (pgx state carries obs)
  - Opponent inside the JIT rollout uses greedy argmax on legal_action_mask
    (deterministic, avoids RNG dependency on stored state)

Key invariant: states stored in the buffer always have current_player == 0.
"""

import os
import numpy as np
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import pgx
from tqdm import tqdm

from .dqn import ChessDQN
from .array_buffer import ChessArrayReplayBuffer
from counterfactual_rl.utils.action_selection import (
    beam_search_top_k_joint_actions,
    convert_mask_to_indices,
)
from counterfactual_rl.analysis.metrics import compute_consequence_metric
from ..shared.consequence_diagnostics import ConsequenceDiagnostics


class ChessConsequenceDQN(ChessDQN):
    """
    Consequence-weighted DQN (Algorithm 2) for Gardner chess.

    Inherits all ChessDQN internals and overrides:
    - Buffer: ConsequenceReplayBuffer (Equations 2-4 priorities)
    - _update(): adds consequence scoring before Q-update
    - learn(): stores JAX pgx.State in buffer for counterfactual rollouts
    """

    def __init__(self, env, env_info: Dict, config: Optional[Dict] = None):
        super().__init__(env, env_info, config)

        per_params = self.config.get('PER_parameters', {})
        self.buffer = ChessArrayReplayBuffer(
            capacity=self.config.get('M', 200000),
            obs_dim=self.obs_dim,
            eps=per_params.get('eps', 0.01),
            beta=per_params.get('beta', 0.25),
            max_priority=per_params.get('maximum_priority', 1.0),
            store_consequences=True,
            mu=self.config.get('mu', 0.5),
            priority_mixing=self.config.get('priority_mixing', 'additive'),
            mu_c=self.config.get('mu_c', 1.0),
            mu_delta=self.config.get('mu_delta', 1.0),
        )

        self.score_interval = self.config.get('score_interval', 200)
        self.n_score_sample = self.config.get('n_score_sample', 128)
        self.consequence_metric = self.config.get('consequence_metric', 'wasserstein')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')

        self.cf_horizon = self.config.get('cf_horizon', 10)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 16)
        self.cf_top_k = self.config.get('cf_top_k', 10)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)

        self.q_update_count = 0
        self.diagnostics_enabled = self.config.get('diagnostics_enabled', False)
        self._compiled_batched_fn = None

    def _build_batched_rollout_fn(self):
        """
        Build triple-vmapped JIT function for batched chess consequence scoring.

        Parallelism:
            vmap over B transitions (states)
              vmap over K actions (different white first moves)
                vmap over N rollouts (different RNG keys for opponent)
                  lax.scan over H horizon steps

        Each horizon step = white greedy move + opponent greedy move.
        Opponent uses greedy argmax on Q-network (deterministic, JIT-compatible).
        State invariant maintained: every carry state has current_player == 0.
        """
        pgx_env = pgx.make("gardner_chess")
        network = self.network
        horizon = self.cf_horizon
        gamma = self.cf_gamma

        def policy_fn(params, obs_flat, legal_mask_1d):
            """Greedy white policy. Returns scalar int32 action."""
            q = network.apply(params, obs_flat)   # (1, 1225)
            return jnp.argmax(jnp.where(legal_mask_1d, q[0], -jnp.inf))

        def opp_step(state, rng_key):
            """Opponent (black) takes greedy move w.r.t. its own Q-network.
            Using greedy argmax over legal actions keeps it deterministic
            and JIT-compatible without needing separate opponent params."""
            # Simple greedy: pick highest-index legal action for opponent
            # (random in distribution, deterministic per key)
            logits = jax.random.normal(rng_key, (1225,))
            masked = jnp.where(state.legal_action_mask, logits, -jnp.inf)
            return pgx_env.step(state, jnp.argmax(masked))

        def single_rollout(params, state, first_action, rng_key):
            """
            One rollout from 'state' (current_player==0) with white's first_action.

            Args:
                params:       network params (passed dynamically, not captured)
                state:        pgx.State with current_player == 0
                first_action: int32 scalar — white's first move
                rng_key:      PRNGKey for all stochastic decisions in this rollout

            Returns:
                Cumulative discounted return for white (float32 scalar)
            """
            # White's first move
            s1 = pgx_env.step(state, first_action)
            r1 = s1.rewards[0]
            done1 = s1.terminated | s1.truncated

            # Opponent responds
            rng_key, opp_key = jax.random.split(rng_key)
            s2 = jax.lax.cond(done1, lambda: s1, lambda: opp_step(s1, opp_key))
            r2 = jnp.where(done1, 0.0, s2.rewards[0])
            done2 = done1 | s2.terminated | s2.truncated

            init_carry = (s2, rng_key, r1 + r2, jnp.float32(gamma), done2)

            def scan_step(carry, _):
                s, key, cum, disc, done = carry
                key, opp_k = jax.random.split(key)

                # White greedy (frozen to action 0 if already done)
                aw = jax.lax.cond(
                    done,
                    lambda: jnp.int32(0),
                    lambda: policy_fn(params, s.observation.reshape(-1), s.legal_action_mask),
                )
                sw = jax.lax.cond(done, lambda: s, lambda: pgx_env.step(s, aw))
                rw = jnp.where(done, 0.0, sw.rewards[0])
                dw = done | sw.terminated | sw.truncated

                # Opponent responds
                so = jax.lax.cond(dw, lambda: sw, lambda: opp_step(sw, opp_k))
                ro = jnp.where(dw, 0.0, so.rewards[0])
                do = dw | so.terminated | so.truncated

                new_cum = cum + disc * (rw + ro)
                new_disc = jnp.where(do, disc, disc * gamma)
                return (so, key, new_cum, new_disc, do), None

            final_carry, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=horizon - 1)
            return final_carry[2]  # cumulative return

        # Triple vmap: B transitions x K actions x N rollouts
        f = jax.vmap(single_rollout, in_axes=(None, None, None, 0))  # N
        f = jax.vmap(f,              in_axes=(None, None, 0, 0))      # K
        f = jax.vmap(f,              in_axes=(None, 0, 0, 0))         # B
        # Signature: (params, states[B], actions[B,K], keys[B,K,N,2]) -> (B,K,N)
        self._compiled_batched_fn = jax.jit(f)

    def _score_buffer_transitions(self):
        """
        Algorithm 2, lines 11-12: Batched consequence scoring for chess.

        Chess-specific changes vs SMAX:
        - valid actions come from jax_state.legal_action_mask directly
        - actual_action is a 1-tuple (n_agents=1)
        - actions_array is (B, K) scalars, not (B, K, n_agents)
        - jax_obs not used (pgx state carries observation)
        """
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        timer = self.metrics_logger.timer
        ep = self._current_episode

        with timer('update.scoring.sample', episode=ep):
            data, indices = self.buffer.sample_uniform(n_score)

        with timer('update.scoring.beam', episode=ep):
            all_actions = []
            all_action_probs = []
            all_actual_actions = []
            valid_indices = []
            valid_states = []

            for i, idx in enumerate(indices):
                jax_state = self.buffer.get_jax_state(idx)
                if jax_state is None:
                    continue

                actual_action = (int(data['a'][i, 0]),)  # 1-tuple

                # Valid actions from stored pgx state (n_agents=1 wrapper)
                legal_mask = np.array(jax_state.legal_action_mask)
                valid_actions_wrapped = [[j for j, v in enumerate(legal_mask) if v]]

                actions_to_eval, joint_probs = beam_search_top_k_joint_actions(
                    valid_actions=valid_actions_wrapped,
                    k=self.cf_top_k,
                    return_probs=True,
                )

                # Ensure actual action is included
                if actual_action not in actions_to_eval:
                    actions_to_eval = [actual_action] + actions_to_eval[:self.cf_top_k - 1]
                    if actual_action not in joint_probs:
                        min_prob = min(joint_probs.values()) if joint_probs else 0.01
                        joint_probs[actual_action] = min_prob * 0.5

                while len(actions_to_eval) < self.cf_top_k:
                    actions_to_eval.append(actual_action)

                valid_indices.append(i)
                valid_states.append(jax_state)
                all_actions.append(actions_to_eval)
                all_action_probs.append(joint_probs)
                all_actual_actions.append(actual_action)

        if not valid_states:
            return

        B = len(valid_states)
        K = self.cf_top_k
        N = self.cf_n_rollouts

        with timer('update.scoring.stack', episode=ep):
            batched_states = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *valid_states)

            # actions_array: (B, K) scalars — chess actions are single ints
            actions_array = jnp.array(
                [[a[0] for a in trans_actions] for trans_actions in all_actions],
                dtype=jnp.int32,
            )

            self._key, subkey = jax.random.split(self._key)
            keys_flat = jax.random.split(subkey, B * K * N)
            keys_array = keys_flat.reshape(B, K, N, 2)

            if self._compiled_batched_fn is None:
                print("Compiling batched rollout function (one-time cost)...")
                self._build_batched_rollout_fn()

        with timer('update.scoring.rollouts', episode=ep, batch_size=B):
            returns_array = self._compiled_batched_fn(
                self.params, batched_states, actions_array, keys_array
            )
            returns_array = jax.block_until_ready(returns_array)
            returns_np = np.array(returns_array)  # (B, K, N)

        with timer('update.scoring.metrics', episode=ep):
            scores = np.zeros(B)
            for i in range(B):
                actual_action = all_actual_actions[i]
                actions_list = all_actions[i]
                action_probs = all_action_probs[i]

                return_distributions = {}
                for j, action_tuple in enumerate(actions_list):
                    if action_tuple not in return_distributions:
                        return_distributions[action_tuple] = returns_np[i, j]

                scores[i] = compute_consequence_metric(
                    actual_action,
                    return_distributions,
                    metric=self.consequence_metric,
                    action_probs=action_probs,
                    aggregation=self.consequence_aggregation,
                )

            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        with timer('update.scoring.buffer_update', episode=ep):
            scored_buffer_indices = indices[np.array(valid_indices)]
            self.buffer.update_consequence_scores(scored_buffer_indices, scores)

            if self.diagnostics_enabled:
                episode_steps = np.zeros(len(valid_indices), dtype=int)
                self.diagnostics.log_scoring_pass(
                    self.q_update_count, scores, returns_np,
                    all_actual_actions, all_actions, all_action_probs, self.buffer,
                    episode_steps=episode_steps,
                )

    def _update(self):
        """Perform update with consequence scoring (Algorithm 2, lines 10-18)."""
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1
        ep = self._current_episode
        timer = self.metrics_logger.timer

        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        with timer('update.q_update', episode=ep):
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

    def _add_chunk_to_buffer(self, outputs, N_ENVS: int, T: int):
        """
        Override of ChessDQN._add_chunk_to_buffer.
        Also stores pgx.State per transition for consequence rollouts.

        pgx states are converted from JAX to numpy in one bulk call, then
        reshaped to (N*T, ...) and stored via slice assignment — no per-transition loop.
        """
        n = N_ENVS * T
        saved_states = outputs[7]  # pgx.State pytree, leaves (N_ENVS, T, ...)

        # One bulk GPU→CPU copy + reshape per leaf (not one copy per transition)
        states_flat = jax.tree.map(
            lambda x: np.array(x).reshape(n, *x.shape[2:]),
            saved_states,
        )

        self.buffer.add_batch(
            obs        = np.array(outputs[3]).reshape(n, -1),
            next_obs   = np.array(outputs[4]).reshape(n, -1),
            actions    = np.array(outputs[0]).reshape(n),
            rewards    = np.array(outputs[1]).reshape(n),
            dones      = np.array(outputs[2]).reshape(n),
            masks      = np.array(outputs[5]).reshape(n, 1, -1),
            next_masks = np.array(outputs[6]).reshape(n, 1, -1),
            states_flat = states_flat,
        )

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'ChessConsequenceDQN':
        """
        Train Consequence-weighted DQN on Gardner chess (Algorithm 2), vectorized.

        Uses the inherited _run_collect_chunk() + _add_chunk_to_buffer() (overridden above
        to also store pgx.State per transition). Consequence scoring (_score_buffer_transitions)
        runs every score_interval Q-updates, unchanged from the non-vectorized version.
        """
        n_chunks = n_episodes or self.config['n_episodes']
        N_ENVS = self.n_envs
        T = self.collect_steps
        save_every = self.config.get('save_every', 1000)
        eval_interval = self.config.get('eval_interval', None)
        eval_episodes = self.config.get('eval_episodes', 50)
        record_interval = self.config.get('record_interval', None)

        log_env_info = {**self.env_info, 'scenario': self.env_info.get('env_name', 'gardner_chess')}
        self.metrics_logger = self._make_metrics_logger(
            log_env_info, n_chunks, eval_interval, eval_episodes
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        self.diagnostics = ConsequenceDiagnostics(
            self.metrics_logger.dir, metric_name=self.consequence_metric,
            plot_interval=self.config.get('diagnostics_plot_interval', 100),
            n_step_slices=self.config.get('diagnostics_n_step_slices', 10),
            n_scatter_snapshots=self.config.get('diagnostics_n_scatter_snapshots', 10),
        )

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_win_rate = -1.0

        n_ckpts = self.config.get('n_checkpoints', 100)
        ckpt_interval = max(1, n_chunks // n_ckpts) if n_ckpts > 0 else 0
        ckpt_dir = os.path.join(self.metrics_logger.dir, 'checkpoints')
        if ckpt_interval > 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        if verbose:
            print(f"Training ChessConsequenceDQN on Gardner chess (vectorized)")
            print(f"  N_ENVS={N_ENVS} | collect_steps={T} | {N_ENVS*T} transitions/chunk")
            print(f"  JAX backend: {jax.default_backend()}")
            print(f"  Metric: {self.consequence_metric} | Mixing: {self.config.get('priority_mixing')}")
            print(f"  CF rollouts: {self.cf_n_rollouts} | Horizon: {self.cf_horizon} | Top-K: {self.cf_top_k}")

        pbar = tqdm(range(n_chunks), disable=not verbose)
        for chunk_idx in pbar:
            self._current_episode = chunk_idx
            timer.begin_episode(chunk_idx)

            with timer('collect', episode=chunk_idx):
                outputs = self._run_collect_chunk(N_ENVS, T)

            with timer('buffer.add', episode=chunk_idx):
                # Extract episode stats to numpy before buffer add so we can
                # free outputs (and its GPU memory) before consequence scoring.
                ep_ret_np   = np.array(outputs[8])   # (N, T)
                ep_len_np   = np.array(outputs[9])   # (N, T)
                ep_ended_np = np.array(outputs[2])   # (N, T)
                self._add_chunk_to_buffer(outputs, N_ENVS, T)
                del outputs  # free GPU memory before _update() / consequence rollouts

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

            # Episode stats (already extracted to numpy above)
            for env_i in range(N_ENVS):
                for t in range(T):
                    if ep_ended_np[env_i, t]:
                        self.episode_returns.append(float(ep_ret_np[env_i, t]))
                        self.episode_lengths.append(int(ep_len_np[env_i, t]))

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

            if ckpt_interval > 0 and (chunk_idx + 1) % ckpt_interval == 0:
                self.save(os.path.join(ckpt_dir, f'ckpt_{chunk_idx+1:07d}.pkl'))

            if eval_interval and (chunk_idx + 1) % eval_interval == 0:
                with timer('eval', episode=chunk_idx):
                    metrics = self.evaluate(n_episodes=eval_episodes)
                self.metrics_logger.log_eval(
                    chunk_idx + 1, self.q_update_count, self.epsilon, metrics
                )
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
        self.diagnostics.close()
        self.metrics_logger.close()
        if verbose:
            print(f"Training complete. Run saved to {self.metrics_logger.dir}")
        return self

    def _make_metrics_logger(self, env_info, n_episodes, eval_interval, eval_episodes):
        """Create MetricsLogger with chess-compatible env_info."""
        from ..shared.metrics import MetricsLogger
        return MetricsLogger(
            backend='JAX (Chess Consequence)',
            config=self.config,
            env_info=env_info,
            n_episodes=n_episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )
