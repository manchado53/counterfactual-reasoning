"""
Vectorized counterfactual rollout analysis for Gardner chess (pgx).

Identifies consequential moments in a chess game by running counterfactual
rollouts at every white move and comparing return distributions across
K candidate moves.

Rollout structure per step:
  1. Fix white's first move to a candidate action
  2. Opponent responds (stochastic — source of rollout variance)
  3. lax.scan for (horizon - 1) more move-pairs:
       white greedy → opponent stochastic
  4. Collect discounted return

Three rollout_policy modes control opponent stochasticity and white's greedy source:
  'random'            — white: greedy over random logits, opponent: uniform categorical
  'baseline_vs_random'— white: greedy from AlphaZero baseline, opponent: uniform categorical
  'baseline'          — white: greedy from AlphaZero baseline, opponent: categorical from baseline

White's subsequent moves are deterministic greedy in all modes — the opponent's
stochasticity is sufficient to produce a meaningful return distribution across N rollouts.
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import subprocess
import numpy as np
import jax
import jax.numpy as jnp
import pgx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from counterfactual_rl.envs.chess import GardnerChessEnv, CHESS_ACTIONS
from counterfactual_rl.utils.chess_data_structures import ChessConsequenceRecord
from counterfactual_rl.utils.action_selection import (
    beam_search_top_k_joint_actions,
    convert_mask_to_indices,
)
from counterfactual_rl.analysis.metrics import compute_all_consequence_metrics


class ChessCounterfactualAnalyzer:
    """
    Vectorized counterfactual analyzer for Gardner chess.

    Evaluates consequential moments in a chess game by running counterfactual
    rollouts at every white move. Uses JAX vmap/scan/jit to parallelise all
    rollout computation.

    Example:
        >>> import jax
        >>> from counterfactual_rl.training.pgx.dqn_jax.chess_env import GardnerChessEnv
        >>> env = GardnerChessEnv(seed=0)
        >>> analyzer = ChessCounterfactualAnalyzer(env, rollout_policy='baseline',
        ...                                        horizon=10, n_rollouts=32, top_k=10)
        >>> key = jax.random.PRNGKey(42)
        >>> records = analyzer.evaluate_episode(key)
        >>> analyzer.save_plots(records, save_dir='runs/chess_cf/')
    """

    def __init__(
        self,
        env: GardnerChessEnv,
        rollout_policy: str = 'random',
        horizon: int = 10,
        n_rollouts: int = 32,
        gamma: float = 0.99,
        top_k: int = 10,
        aggregation: str = 'mean',
        store_states: bool = False,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            env:            GardnerChessEnv wrapper instance.
            rollout_policy: 'random' | 'baseline_vs_random' | 'baseline'.
                            Controls opponent policy and white's greedy logit source.
            horizon:        Number of full move-pairs (white + opponent) per rollout.
            n_rollouts:     Number of independent rollouts per candidate move.
            gamma:          Discount factor for rollout returns.
            top_k:          Number of candidate moves to evaluate at each step.
            aggregation:    How to aggregate divergences across alternatives.
                            'mean' | 'max' | 'weighted_mean'.
            store_states:   If True, attach raw pgx.State to each record.
            log_file:       Optional path for log output.
        """
        if rollout_policy not in ('random', 'baseline_vs_random', 'baseline'):
            raise ValueError(
                f"rollout_policy must be 'random', 'baseline_vs_random', or 'baseline', "
                f"got '{rollout_policy}'"
            )

        self.env = env
        self.pgx_env = env._env          # raw pgx env for JIT-traceable rollouts
        self.rollout_policy = rollout_policy
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.top_k = top_k
        self.aggregation = aggregation
        self.store_states = store_states

        # Reuse the baseline model already loaded in the env (if available),
        # otherwise load it. GardnerChessEnv(opponent='baseline') pre-loads it.
        self._baseline_model = None
        if rollout_policy in ('baseline_vs_random', 'baseline'):
            if env._opponent_model is not None:
                self._baseline_model = env._opponent_model
            else:
                self._baseline_model = pgx.make_baseline_model("gardner_chess_v0")

        # Compiled rollout function — built lazily on first call
        self._compiled_rollout_fn = None

        self._setup_logging(log_file)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self, log_file: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"chess_cf_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        log_path = log_file or os.path.join(self.run_dir, "chess_counterfactual.log")
        self.logger = logging.getLogger(f"ChessCounterfactualAnalyzer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        self._log("=" * 70)
        self._log("Chess Counterfactual Analysis")
        self._log(f"rollout_policy={self.rollout_policy}, horizon={self.horizon}, "
                  f"n_rollouts={self.n_rollouts}, top_k={self.top_k}, gamma={self.gamma}")
        self._log("=" * 70)

    def _log(self, msg: str, level: str = "info"):
        getattr(self.logger, level)(msg)

    # ------------------------------------------------------------------
    # Build compiled rollout function
    # ------------------------------------------------------------------

    def _build_rollout_fn(self):
        """
        Build a JIT-compiled, double-vmapped rollout function.

        Signature:
            compiled_fn(state, actions[K], keys[K, N, 2]) -> returns[K, N]

        Inner vmap: N rollouts (different RNG keys, same action)
        Outer vmap: K candidate actions
        lax.scan:   horizon - 1 move-pairs inside each rollout
        """
        pgx_env = self.pgx_env
        horizon = self.horizon
        gamma = self.gamma
        rollout_policy = self.rollout_policy
        baseline_model = self._baseline_model

        # --- Policy closures (captured at compile time) ---

        def _white_move(state, key):
            """White: greedy argmax. Key unused for random mode (logits are random)."""
            if rollout_policy == 'random':
                logits = jax.random.normal(key, (CHESS_ACTIONS,))
            else:
                logits, _ = baseline_model(state.observation[None])  # (1,5,5,115)
                logits = logits[0]
            return jnp.argmax(jnp.where(state.legal_action_mask, logits, -jnp.inf))

        def _opp_move(state, key):
            """Opponent: stochastic categorical. Source of rollout variance."""
            if rollout_policy == 'baseline':
                logits, _ = baseline_model(state.observation[None])  # (1,5,5,115)
                logits = logits[0]
                masked = jnp.where(state.legal_action_mask, logits, -jnp.inf)
            else:
                # Uniform over legal moves
                masked = jnp.where(
                    state.legal_action_mask,
                    jnp.zeros(CHESS_ACTIONS),
                    -jnp.inf,
                )
            return jax.random.categorical(key, masked)

        def single_rollout(state, first_action, rng_key):
            """
            One rollout from state (current_player==0) with white's first_action.

            Returns cumulative discounted return for white (float32 scalar).
            """
            # White's counterfactual first move
            s1 = pgx_env.step(state, first_action)
            r1 = s1.rewards[0]
            done1 = s1.terminated | s1.truncated

            # Opponent responds to white's first move
            rng_key, opp_key = jax.random.split(rng_key)
            s2 = jax.lax.cond(done1, lambda: s1, lambda: pgx_env.step(s1, _opp_move(s1, opp_key)))
            r2 = jnp.where(done1, 0.0, s2.rewards[0])
            done2 = done1 | s2.terminated | s2.truncated

            init_carry = (s2, rng_key, r1 + r2, jnp.float32(gamma), done2)

            def scan_step(carry, _):
                s, key, cum, disc, done = carry
                key, wk, ok = jax.random.split(key, 3)

                # White greedy (frozen to action 0 if already done — masked out anyway)
                aw = jax.lax.cond(
                    done,
                    lambda: jnp.int32(0),
                    lambda: _white_move(s, wk),
                )
                sw = jax.lax.cond(done, lambda: s, lambda: pgx_env.step(s, aw))
                rw = jnp.where(done, 0.0, sw.rewards[0])
                dw = done | sw.terminated | sw.truncated

                # Opponent responds
                so = jax.lax.cond(dw, lambda: sw, lambda: pgx_env.step(sw, _opp_move(sw, ok)))
                ro = jnp.where(dw, 0.0, so.rewards[0])
                do = dw | so.terminated | so.truncated

                new_cum = cum + disc * (rw + ro)
                new_disc = jnp.where(do, disc, disc * gamma)
                return (so, key, new_cum, new_disc, do), None

            final_carry, _ = jax.lax.scan(
                scan_step, init_carry, xs=None, length=horizon - 1
            )
            return final_carry[2]  # cumulative return

        # Inner vmap: N rollouts (same action, different keys)
        f = jax.vmap(single_rollout, in_axes=(None, None, 0))
        # Outer vmap: K candidate actions
        f = jax.vmap(f, in_axes=(None, 0, 0))
        self._compiled_rollout_fn = jax.jit(f)
        self._log("Rollout function compiled.")

    # ------------------------------------------------------------------
    # Rollout execution
    # ------------------------------------------------------------------

    def _run_rollouts(
        self,
        key: jax.Array,
        state,
        actions_to_eval: List[int],
    ) -> np.ndarray:
        """
        Run all counterfactual rollouts in one compiled JAX call.

        Args:
            key:             RNG key.
            state:           pgx.State at the decision point (current_player==0).
            actions_to_eval: List of K move indices to evaluate.

        Returns:
            np.ndarray of shape (K, N) — discounted returns per action per rollout.
        """
        K = len(actions_to_eval)
        N = self.n_rollouts

        actions_array = jnp.array(actions_to_eval, dtype=jnp.int32)  # (K,)

        key, subkey = jax.random.split(key)
        action_keys = jax.random.split(subkey, K)                     # (K, 2)
        all_keys = jax.vmap(lambda k: jax.random.split(k, N))(action_keys)  # (K, N, 2)

        if self._compiled_rollout_fn is None:
            self._log("Compiling rollout function (one-time cost)...")
            print("Compiling rollout function (one-time cost)...")
            self._build_rollout_fn()

        returns_array = self._compiled_rollout_fn(state, actions_array, all_keys)
        returns_array = jax.block_until_ready(returns_array)
        return np.array(returns_array)  # (K, N)

    # ------------------------------------------------------------------
    # Episode evaluation
    # ------------------------------------------------------------------

    def evaluate_episode(
        self,
        key: jax.Array,
        max_steps: int = 200,
        verbose: bool = True,
    ) -> List[ChessConsequenceRecord]:
        """
        Evaluate consequential moments across one full chess game.

        At every white move, runs K × N counterfactual rollouts and computes
        all four consequence metrics. Returns one ChessConsequenceRecord per move.

        Args:
            key:       JAX PRNGKey.
            max_steps: Maximum number of white moves before stopping.
            verbose:   Print progress to stdout.

        Returns:
            List[ChessConsequenceRecord] — one record per white move.
        """
        key, reset_key = jax.random.split(key)
        obs, state = self.env.reset(reset_key)

        records: List[ChessConsequenceRecord] = []
        # Half-move states: initial position + state after every half-move
        # (white's move and black's response are separate frames)
        half_move_states = [state]
        episode_return = 0.0
        step = 0
        done = False
        t_start = time.time()

        self._log(f"\nEpisode start (max_steps={max_steps})")

        while not done and step < max_steps:
            saved_state = state
            saved_obs = obs

            # --- Select white's actual move (epsilon-greedy not needed here;
            #     we just follow the env's built-in white policy or pick randomly) ---
            legal_mask = np.array(state.legal_action_mask)
            legal_indices = np.where(legal_mask)[0]

            key, action_key = jax.random.split(key)
            if self._baseline_model is not None:
                # White follows the baseline model (greedy argmax over its logits)
                logits, _ = self._baseline_model(state.observation[None])  # (1,5,5,115) → (1,1225)
                logits = logits[0]
                masked = jnp.where(state.legal_action_mask, logits, -jnp.inf)
                actual_action = int(jnp.argmax(masked))
            else:
                # Fallback: random legal move (useful for quick testing)
                actual_action = int(jax.random.choice(
                    action_key,
                    jnp.array(legal_indices),
                ))

            # --- Select top-K candidate moves to evaluate ---
            valid_actions_wrapped = [legal_indices.tolist()]
            actions_1tuples, _ = beam_search_top_k_joint_actions(
                valid_actions=valid_actions_wrapped,
                k=self.top_k,
                return_probs=True,
            )
            actions_to_eval = [a[0] for a in actions_1tuples]

            # Ensure actual action is included
            if actual_action not in actions_to_eval:
                actions_to_eval = [actual_action] + actions_to_eval[:self.top_k - 1]

            # Always pad to exactly top_k — constant K means jax.jit never retraces
            # between steps (duplicate entries just get deduplicated in return_distributions)
            while len(actions_to_eval) < self.top_k:
                actions_to_eval.append(actual_action)

            # --- Run counterfactual rollouts ---
            key, rollout_key = jax.random.split(key)
            t0 = time.time()
            returns_np = self._run_rollouts(rollout_key, saved_state, actions_to_eval)
            self._log(f"  step={step}: rollouts in {time.time()-t0:.2f}s, "
                      f"K={len(actions_to_eval)}, N={self.n_rollouts}")

            # --- Build return distributions dict ---
            return_distributions: dict = {}
            for i, move in enumerate(actions_to_eval):
                if move not in return_distributions:
                    return_distributions[move] = returns_np[i]

            # --- Compute all 4 consequence metrics ---
            actual_tuple = (actual_action,)
            dist_tupled = {(m,): v for m, v in return_distributions.items()}

            all_metrics = compute_all_consequence_metrics(
                actual_tuple,
                dist_tupled,
                aggregation=self.aggregation,
            )

            kl_score    = all_metrics['kl_divergence'][0]
            kl_divs     = {k[0]: v for k, v in all_metrics['kl_divergence'][1].items()}
            jsd_score   = all_metrics['jensen_shannon'][0]
            jsd_divs    = {k[0]: v for k, v in all_metrics['jensen_shannon'][1].items()}
            tv_score    = all_metrics['total_variation'][0]
            tv_dists    = {k[0]: v for k, v in all_metrics['total_variation'][1].items()}
            w_score     = all_metrics['wasserstein'][0]
            w_dists     = {k[0]: v for k, v in all_metrics['wasserstein'][1].items()}

            self._log(f"  step={step}: KL={kl_score:.4f}, JSD={jsd_score:.4f}, "
                      f"TV={tv_score:.4f}, W={w_score:.4f}")

            record = ChessConsequenceRecord(
                obs=np.array(saved_obs),
                action=actual_action,
                timestep=step,
                episode_return=episode_return,
                kl_score=kl_score,
                kl_divergences=kl_divs,
                return_distributions=return_distributions,
                jsd_score=jsd_score,
                jsd_divergences=jsd_divs,
                tv_score=tv_score,
                tv_distances=tv_dists,
                wasserstein_score=w_score,
                wasserstein_distances=w_dists,
                pgx_state=saved_state if self.store_states else None,
            )
            records.append(record)

            # --- Step environment, capturing both half-moves as separate frames ---
            s1 = self.pgx_env.step(saved_state, jnp.int32(actual_action))
            reward = float(np.array(s1.rewards[0]))
            done = bool(np.array(s1.terminated | s1.truncated))
            half_move_states.append(s1)   # frame: board after white's move

            if not done:
                self.env._rng, opp_k = jax.random.split(self.env._rng)
                black_action = self.env._opponent_action(s1, opp_k)
                s2 = self.pgx_env.step(s1, black_action)
                reward += float(np.array(s2.rewards[0]))
                done = bool(np.array(s2.terminated | s2.truncated))
                half_move_states.append(s2)  # frame: board after black's response
                state = s2
            else:
                state = s1

            obs = GardnerChessEnv._obs(state)
            episode_return += reward
            step += 1

            if verbose:
                print(f"  step={step-1:3d}  action={actual_action:4d}  "
                      f"KL={kl_score:.3f}  WS={w_score:.3f}  "
                      f"reward={reward:+.1f}  done={done}")

        self._last_half_move_states = half_move_states  # used by save_game

        elapsed = time.time() - t_start
        self._log(f"Episode done: {len(records)} moves in {elapsed:.1f}s, "
                  f"total_return={episode_return:.2f}")
        if verbose:
            print(f"Episode complete: {len(records)} moves, "
                  f"return={episode_return:.2f}, {elapsed:.1f}s")

        return records

    def evaluate_multiple_episodes(
        self,
        key: jax.Array,
        n_episodes: int = 20,
        max_steps: int = 200,
        verbose: bool = True,
    ) -> List[ChessConsequenceRecord]:
        """
        Evaluate consequential moments across multiple games.

        All records from all episodes are concatenated into a single flat list.
        Useful for averaging patterns across many games (e.g., "is move 15
        consistently more consequential than move 5?").

        Args:
            key:         JAX PRNGKey.
            n_episodes:  Number of games to play.
            max_steps:   Maximum white moves per game.
            verbose:     Print per-episode summary.

        Returns:
            List[ChessConsequenceRecord] — one record per white move, all episodes.
        """
        all_records: List[ChessConsequenceRecord] = []
        for ep in range(n_episodes):
            key, ep_key = jax.random.split(key)
            ep_records = self.evaluate_episode(ep_key, max_steps=max_steps, verbose=False)
            all_records.extend(ep_records)
            if verbose:
                avg_ws = np.mean([r.wasserstein_score for r in ep_records
                                  if r.wasserstein_score is not None])
                print(f"Episode {ep+1}/{n_episodes}: {len(ep_records)} moves, "
                      f"avg WS={avg_ws:.4f}")
        return all_records

    # ------------------------------------------------------------------
    # Game recording
    # ------------------------------------------------------------------

    def save_game(
        self,
        records: List[ChessConsequenceRecord],
        save_path: Optional[str] = None,
        frame_duration: float = 1.5,
        save_gif: bool = True,
        gif_scale: float = 2.0,
    ):
        """
        Save the game as an animated SVG and optionally a GIF.

        Uses half-move states captured during evaluate_episode (one frame per
        half-move: white's move AND black's response are separate frames).
        Falls back to per-white-move states from records if evaluate_episode
        has not been called on this instance yet.

        Args:
            records:        List of ChessConsequenceRecord from evaluate_episode().
            save_path:      Output path prefix (without extension). Defaults to
                            run_dir/game. Produces game.svg (always) and game.gif
                            (if save_gif=True and ImageMagick is available).
            frame_duration: Seconds per frame (1.5s gives a comfortable viewing pace).
            save_gif:       Also produce a GIF via ImageMagick convert.
            gif_scale:      SVG scale factor for GIF frames (controls resolution).
        """
        # Prefer half-move states (2× more frames, shows both sides' moves)
        states = getattr(self, '_last_half_move_states', None)
        if not states:
            states = [r.pgx_state for r in records if r.pgx_state is not None]
        if not states:
            print("No states found. Call evaluate_episode first (store_states=True "
                  "is only needed for the fallback path).")
            return

        base = save_path or os.path.join(self.run_dir, "game")
        svg_path = base if base.endswith('.svg') else base + '.svg'

        pgx.save_svg_animation(states, svg_path, frame_duration_seconds=frame_duration)
        n = len(states)
        print(f"SVG animation saved to {svg_path}  ({n} frames, {frame_duration}s/frame)")
        self._log(f"SVG animation saved to {svg_path}")

        if save_gif:
            gif_path = base.rstrip('.svg') + '.gif' if base.endswith('.svg') \
                       else base + '.gif'
            self._save_gif(states, gif_path, frame_duration, gif_scale)

    def _save_gif(self, states, out_path: str, frame_duration: float, scale: float):
        """Convert states to GIF via ImageMagick. Skips gracefully if unavailable."""
        import tempfile
        delay_cs = int(frame_duration * 100)
        try:
            subprocess.run(["convert", "--version"], check=True,
                           capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ImageMagick 'convert' not found — skipping GIF export. "
                  "Install with: sudo apt-get install imagemagick")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            png_paths = []
            for i, state in enumerate(states):
                svg_f = os.path.join(tmpdir, f"frame_{i:04d}.svg")
                png_f = os.path.join(tmpdir, f"frame_{i:04d}.png")
                pgx.save_svg(state, svg_f, scale=scale)
                subprocess.run(
                    ["convert", "-background", "white", "-flatten", svg_f, png_f],
                    check=True, capture_output=True,
                )
                png_paths.append(png_f)

            cmd = (["convert", "-delay", str(delay_cs), "-loop", "0"]
                   + png_paths + [out_path])
            subprocess.run(cmd, check=True, capture_output=True)

        print(f"GIF saved to {out_path}  ({len(states)} frames, {frame_duration}s/frame)")
        self._log(f"GIF saved to {out_path}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def save_plots(
        self,
        records: List[ChessConsequenceRecord],
        save_dir: Optional[str] = None,
    ):
        """
        Generate and save all consequence analysis plots.

        Args:
            records:  List of ChessConsequenceRecord from evaluate_episode().
            save_dir: Directory to save plots. Defaults to self.run_dir.
        """
        from counterfactual_rl.visualization.chess_plots import ChessConsequencePlotter

        out_dir = save_dir or self.run_dir
        os.makedirs(out_dir, exist_ok=True)

        plotter = ChessConsequencePlotter()
        plotter.plot_comprehensive(
            records,
            save_path=os.path.join(out_dir, "analysis_comprehensive.png"),
        )
        plotter.print_statistics(records)
        self._log(f"Plots saved to {out_dir}")
        print(f"Plots saved to {out_dir}")
