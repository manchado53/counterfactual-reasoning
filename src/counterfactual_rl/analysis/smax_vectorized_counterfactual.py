"""
Vectorized counterfactual rollout analysis for SMAX (JaxMARL) environments.

Uses JAX primitives (vmap, lax.scan, jit) to parallelize rollout computation:
- Inner loop (horizon steps) -> jax.lax.scan  (sequential dependency)
- Middle loop (n_rollouts)   -> jax.vmap       (embarrassingly parallel)
- Outer loop (actions)       -> jax.vmap       (independent evaluations)

The entire rollout computation is JIT-compiled into a single XLA kernel,
replacing thousands of sequential Python->JAX round-trips.

This is a standalone file that parallels smax_counterfactual.py.
The original file is not modified.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import logging
from datetime import datetime
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from counterfactual_rl.utils.smax_data_structures import SMAXConsequenceRecord
from counterfactual_rl.utils.smax_utils import (
    action_dict_to_tuple,
    get_valid_actions_mask,
    get_agent_names,
    sum_rewards,
)
from counterfactual_rl.utils.smax_jax_utils import (
    jax_actions_array_to_dict,
    jax_sum_rewards,
)
from counterfactual_rl.utils.action_selection import (
    beam_search_top_k_joint_actions,
    convert_mask_to_indices,
)
from counterfactual_rl.analysis.metrics import compute_all_consequence_metrics


class SMAXVectorizedCounterfactualAnalyzer:
    """
    Vectorized counterfactual analyzer for SMAX (JaxMARL) environments.

    Same public interface as SMAXCounterfactualAnalyzer but uses JAX
    vmap/scan/jit to parallelize all rollout computation into a single
    compiled kernel.

    Requirements:
        - policy_fn must be JIT-compatible (no variable-length ops like
          jnp.where(mask)[0]). Use make_jax_random_policy() from
          smax_jax_utils.py for a JIT-safe random policy.

    Example:
        >>> from jaxmarl import make
        >>> from jaxmarl.environments.smax import map_name_to_scenario
        >>> from counterfactual_rl.utils.smax_jax_utils import make_jax_random_policy
        >>>
        >>> scenario = map_name_to_scenario("3m")
        >>> env = make("HeuristicEnemySMAX", scenario=scenario)
        >>> policy_fn = make_jax_random_policy(list(env.agents))
        >>>
        >>> analyzer = SMAXVectorizedCounterfactualAnalyzer(
        ...     env=env,
        ...     policy_fn=policy_fn,
        ...     top_k=20,
        ...     horizon=20,
        ...     n_rollouts=48
        ... )
        >>> key = jax.random.PRNGKey(42)
        >>> records = analyzer.evaluate_episode(key)
    """

    def __init__(
        self,
        env,  # SMAX environment
        policy_fn: Callable,  # (key, obs, avail_actions) -> action_dict (must be JIT-safe)
        get_action_probs_fn: Optional[Callable] = None,
        horizon: int = 20,
        n_rollouts: int = 48,
        gamma: float = 0.99,
        top_k: int = 20,
        log_file: Optional[str] = None,
        store_states: bool = False,
        aggregation: str = 'weighted_mean',  # 'max', 'mean', or 'weighted_mean'
    ):
        self.env = env
        self.policy_fn = policy_fn
        self.get_action_probs_fn = get_action_probs_fn
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.top_k = top_k
        self.store_states = store_states
        self.aggregation = aggregation

        # Extract environment info
        self.agent_names = get_agent_names(env)
        self.n_agents = len(self.agent_names)
        self.n_actions = env.action_space(self.agent_names[0]).n

        # Lazy-built compiled rollout function (built on first use)
        self._compiled_rollout_fn = None

        # State sequence from most recent evaluate_episode() call
        # List of (step_key, state, action_dict) tuples for SMAXVisualizer
        self.last_state_seq = None

        # Setup logging
        self.logger = None
        self.run_dir = None
        self.log_file = log_file
        self._setup_logging(log_file)

    # ------------------------------------------------------------------
    # Logging (same as original)
    # ------------------------------------------------------------------

    def _setup_logging(self, log_file: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"smax_vectorized_run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        if log_file is None:
            log_filename = "smax_vectorized_counterfactual_analysis.log"
        else:
            log_filename = os.path.basename(log_file)

        self.log_file = os.path.join(self.run_dir, log_filename)
        self.logger = logging.getLogger(f"SMAXVectorizedAnalyzer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self._log("=" * 80)
        self._log("SMAX Vectorized Counterfactual Analysis Started")
        self._log(f"Configuration: n_agents={self.n_agents}, n_actions={self.n_actions}")
        self._log(f"Parameters: horizon={self.horizon}, n_rollouts={self.n_rollouts}, top_k={self.top_k}")
        self._log(f"Gamma={self.gamma}, Aggregation={self.aggregation}")
        self._log("=" * 80)

    def _log(self, message: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(message)

    # ------------------------------------------------------------------
    # Vectorized rollout core
    # ------------------------------------------------------------------

    def _build_vectorized_rollout_fn(self):
        """
        Build a JIT-compiled, double-vmapped rollout function.

        Returns a function:
            compiled_fn(state, first_actions_array, rng_keys) -> returns_array

        Where:
            state: single SMAX State (broadcast to all rollouts)
            first_actions_array: (n_actions, n_agents) int32 array
            rng_keys: (n_actions, n_rollouts, 2) PRNGKey array

        Returns:
            (n_actions, n_rollouts) float array of discounted returns
        """
        # Capture references for use inside closures (all static during tracing)
        env = self.env
        agent_names = self.agent_names
        horizon = self.horizon
        gamma = self.gamma
        policy_fn = self.policy_fn

        def single_rollout(state, first_action_array, rng_key):
            """
            One rollout: take first_action, then follow policy for (horizon-1) steps.
            Returns scalar discounted return.
            """
            # Convert action array to dict for env.step
            action_dict = jax_actions_array_to_dict(first_action_array, agent_names)

            # Take counterfactual first action
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = env.step(step_key, state, action_dict)

            first_reward = jax_sum_rewards(rewards, agent_names)
            done = dones["__all__"]

            # lax.scan for remaining (horizon - 1) steps
            # Carry: (state, obs, rng_key, cumulative_return, discount, done_flag)
            init_carry = (state, obs, rng_key, first_reward, jnp.float32(gamma), done)

            def scan_step(carry, _unused):
                state_c, obs_c, key_c, cum_return, discount, done_flag = carry

                # Get available actions
                avail_actions = env.get_avail_actions(state_c)

                # Get policy action
                key_c, policy_key, step_key = jax.random.split(key_c, 3)
                action_dict = policy_fn(policy_key, obs_c, avail_actions)

                # Step environment
                new_obs, new_state, rewards, dones, infos = env.step(
                    step_key, state_c, action_dict
                )

                step_reward = jax_sum_rewards(rewards, agent_names)

                # Done masking: zero out reward if already done.
                # lax.scan can't break, so we mask instead.
                masked_reward = jnp.where(done_flag, 0.0, step_reward)
                new_cum_return = cum_return + discount * masked_reward

                # Freeze discount once done
                new_discount = jnp.where(done_flag, discount, discount * gamma)

                # Once done, stay done (env auto-resets but we mask rewards)
                new_done = jnp.logical_or(done_flag, dones["__all__"])

                new_carry = (new_state, new_obs, key_c, new_cum_return, new_discount, new_done)
                return new_carry, None  # No per-step output needed

            final_carry, _ = jax.lax.scan(
                scan_step,
                init_carry,
                xs=None,
                length=horizon - 1
            )

            total_return = final_carry[3]  # cum_return from carry
            return total_return

        # Inner vmap: parallelize over n_rollouts (differ only in RNG key)
        batched_over_rollouts = jax.vmap(
            single_rollout,
            in_axes=(None, None, 0)  # state=broadcast, action=broadcast, key=batched
        )

        # Outer vmap: parallelize over actions (differ in first action + keys)
        batched_over_actions_and_rollouts = jax.vmap(
            batched_over_rollouts,
            in_axes=(None, 0, 0)  # state=broadcast, actions=batched, keys=batched
        )

        # JIT compile the entire double-vmapped function
        compiled_fn = jax.jit(batched_over_actions_and_rollouts)

        return compiled_fn

    def _perform_vectorized_rollouts(
        self,
        key: jax.Array,
        state: Any,
        actions_to_evaluate: List[Tuple[int, ...]],
    ) -> jnp.ndarray:
        """
        Run all counterfactual rollouts in one vectorized JAX call.

        Args:
            key: RNG key
            state: SMAX State at the decision point
            actions_to_evaluate: List of joint action tuples

        Returns:
            jnp.ndarray of shape (n_actions, n_rollouts)
        """
        n_actions = len(actions_to_evaluate)

        # Convert action tuples to JAX array: (n_actions, n_agents)
        actions_array = jnp.array(actions_to_evaluate, dtype=jnp.int32)

        # Pre-split RNG keys: (n_actions, n_rollouts, 2)
        key, subkey = jax.random.split(key)
        action_keys = jax.random.split(subkey, n_actions)  # (n_actions, 2)
        all_keys = jax.vmap(
            lambda k: jax.random.split(k, self.n_rollouts)
        )(action_keys)  # (n_actions, n_rollouts, 2)

        # Lazy-build the compiled function on first call
        if self._compiled_rollout_fn is None:
            self._log("Compiling vectorized rollout function (one-time cost)...")
            print("Compiling vectorized rollout function (one-time cost)...")
            self._compiled_rollout_fn = self._build_vectorized_rollout_fn()

        # Execute all rollouts in one compiled call
        returns_array = self._compiled_rollout_fn(state, actions_array, all_keys)

        return returns_array

    # ------------------------------------------------------------------
    # Public API (same interface as SMAXCounterfactualAnalyzer)
    # ------------------------------------------------------------------

    def perform_counterfactual_rollouts(
        self,
        key: jax.Array,
        state: Any,
        obs: Dict[str, jnp.ndarray],
        actual_action: Tuple[int, ...],
        current_timestep: int = 0,
        verbose: bool = False
    ) -> Tuple[Dict[Tuple[int, ...], np.ndarray], Dict[Tuple[int, ...], float]]:
        """
        Perform counterfactual rollouts for top-K joint actions (vectorized).

        Same interface as SMAXCounterfactualAnalyzer.perform_counterfactual_rollouts,
        but also returns action probabilities for weighted aggregation.

        Returns:
            Tuple of (return_distributions, joint_action_probs)
        """
        return_distributions: Dict[Tuple[int, ...], np.ndarray] = {}

        self._log(f"Starting vectorized counterfactual rollouts at step {current_timestep}")

        # --- Python-side: beam search (runs once, negligible cost) ---
        valid_actions_mask = get_valid_actions_mask(self.env, state)

        per_agent_probs = None
        if self.get_action_probs_fn is not None:
            per_agent_probs = self.get_action_probs_fn(obs, valid_actions_mask)
            self._log("Action probabilities computed from policy")
        else:
            self._log("Using uniform action probabilities")

        valid_actions = convert_mask_to_indices(valid_actions_mask)

        # Get top-K actions WITH their joint probabilities
        actions_to_evaluate, joint_action_probs = beam_search_top_k_joint_actions(
            valid_actions=valid_actions,
            action_probs=per_agent_probs,
            k=self.top_k,
            return_probs=True
        )

        # Ensure actual action is included
        if actual_action not in actions_to_evaluate:
            actions_to_evaluate = [actual_action] + actions_to_evaluate[:self.top_k - 1]
            # Recompute probs to include actual action (assign it small prob if not in top-K)
            if actual_action not in joint_action_probs:
                # Assign minimum probability (it wasn't in top-K, so it's less likely)
                min_prob = min(joint_action_probs.values()) if joint_action_probs else 0.01
                joint_action_probs[actual_action] = min_prob * 0.5
            self._log(f"Added actual action {actual_action} to evaluation set")

        self._log(f"Evaluating {len(actions_to_evaluate)} joint actions x {self.n_rollouts} rollouts")

        # --- JAX-side: all rollouts in one vectorized call ---
        key, rollout_key = jax.random.split(key)

        t0 = time.time()
        returns_array = self._perform_vectorized_rollouts(
            key=rollout_key,
            state=state,
            actions_to_evaluate=actions_to_evaluate,
        )
        # Block until JAX computation completes (for accurate timing)
        returns_array = jax.block_until_ready(returns_array)
        t1 = time.time()

        self._log(f"Vectorized rollouts completed in {t1 - t0:.3f}s")

        # --- Python-side: convert JAX output back to dict format ---
        returns_np = np.array(returns_array)  # (n_actions, n_rollouts)

        for i, action in enumerate(actions_to_evaluate):
            return_distributions[action] = returns_np[i]
            mean_ret = returns_np[i].mean()
            std_ret = returns_np[i].std()
            prob = joint_action_probs.get(action, 0.0)
            self._log(f"  Action {action}: mean={mean_ret:.4f}, std={std_ret:.4f}, prob={prob:.4f}")

        return return_distributions, joint_action_probs

    def evaluate_episode(
        self,
        key: jax.Array,
        max_steps: int = 100,
        verbose: bool = True
    ) -> List[SMAXConsequenceRecord]:
        """
        Evaluate consequential states for a single episode.

        Same interface as SMAXCounterfactualAnalyzer.evaluate_episode.
        """
        key, reset_key = jax.random.split(key)
        obs, state = self.env.reset(reset_key)

        done = False
        step = 0
        records: List[SMAXConsequenceRecord] = []
        state_seq = []  # Collect (key, state, actions) for video replay
        episode_return = 0.0
        episode_start_time = time.time()

        self._log("\n" + "=" * 80)
        self._log(f"Starting Episode Evaluation (max_steps={max_steps})")
        self._log("=" * 80)

        pbar = tqdm(total=max_steps, desc="Episode steps", disable=not verbose)

        while not done and step < max_steps:
            pbar.update(1)
            self._log(f"\n{'=' * 40} STEP {step} {'=' * 40}")

            # Save state (JAX states are immutable)
            saved_state = state
            saved_obs = obs

            # Get available actions
            avail_actions = self.env.get_avail_actions(state)

            # Get action from policy
            key, policy_key = jax.random.split(key)
            action_dict = self.policy_fn(policy_key, obs, avail_actions)
            action_tuple = action_dict_to_tuple(action_dict, self.agent_names)

            self._log(f"Policy selected action: {action_tuple}")

            # Perform vectorized counterfactual analysis
            key, counterfactual_key = jax.random.split(key)
            counterfactual_start = time.time()

            return_distributions, joint_action_probs = self.perform_counterfactual_rollouts(
                key=counterfactual_key,
                state=saved_state,
                obs=saved_obs,
                actual_action=action_tuple,
                current_timestep=step,
                verbose=verbose
            )

            counterfactual_duration = time.time() - counterfactual_start
            self._log(f"Counterfactual analysis took {counterfactual_duration:.2f}s")

            # Compute consequence metrics (scipy/numpy, stays on CPU)
            # Pass action probabilities for weighted aggregation
            all_metrics = compute_all_consequence_metrics(
                action_tuple,
                return_distributions,
                action_probs=joint_action_probs,
                aggregation=self.aggregation
            )

            kl_score = all_metrics['kl_divergence'][0]
            kl_divergences = all_metrics['kl_divergence'][1]
            jsd_score = all_metrics['jensen_shannon'][0]
            jsd_divergences = all_metrics['jensen_shannon'][1]
            tv_score = all_metrics['total_variation'][0]
            tv_distances = all_metrics['total_variation'][1]
            wasserstein_score = all_metrics['wasserstein'][0]
            wasserstein_distances = all_metrics['wasserstein'][1]

            self._log(f"Scores: KL={kl_score:.6f}, JSD={jsd_score:.6f}, TV={tv_score:.6f}, W={wasserstein_score:.6f}")

            # Create record
            record = SMAXConsequenceRecord(
                obs={k: np.array(v) for k, v in saved_obs.items()},
                action=action_tuple,
                timestep=step,
                episode_return=episode_return,
                state=saved_state if self.store_states else None,
                rng_key=counterfactual_key if self.store_states else None,
                kl_score=kl_score,
                kl_divergences=kl_divergences,
                return_distributions=return_distributions,
                jsd_score=jsd_score,
                jsd_divergences=jsd_divergences,
                tv_score=tv_score,
                tv_distances=tv_distances,
                wasserstein_score=wasserstein_score,
                wasserstein_distances=wasserstein_distances,
            )
            records.append(record)

            # Record state for video replay (before stepping)
            key, step_key = jax.random.split(key)
            state_seq.append((step_key, saved_state, action_dict))

            # Execute actual step in main trajectory
            obs, state, rewards, dones, infos = self.env.step(step_key, state, action_dict)

            total_reward = sum_rewards(rewards, self.agent_names)
            episode_return += total_reward
            done = dones.get("__all__", False)

            self._log(f"Actual step: reward={total_reward:.4f}, done={done}")

            if verbose:
                pbar.set_postfix({
                    'action': str(action_tuple),
                    'KL': f"{kl_score:.4f}"
                })

            step += 1

        pbar.close()

        # Store state sequence for video replay
        self.last_state_seq = state_seq

        total_time = time.time() - episode_start_time
        self._log("\n" + "=" * 80)
        self._log(f"Episode complete: {len(records)} records in {total_time:.2f}s")
        self._log(f"Episode return: {episode_return:.4f}")
        self._log("=" * 80)

        return records

    def evaluate_multiple_episodes(
        self,
        key: jax.Array,
        n_episodes: int = 20,
        max_steps: int = 100,
        verbose: bool = True
    ) -> List[SMAXConsequenceRecord]:
        """
        Evaluate consequential states across multiple episodes.

        Same interface as SMAXCounterfactualAnalyzer.evaluate_multiple_episodes.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SMAX Vectorized Counterfactual Analysis")
            print(f"Agents: {self.n_agents}, Actions/agent: {self.n_actions}")
            print(f"Top-K: {self.top_k}, Horizon: {self.horizon}, Rollouts: {self.n_rollouts}")
            print(f"{'=' * 60}\n")

        all_records: List[SMAXConsequenceRecord] = []

        for episode in tqdm(range(n_episodes), desc="Episodes", disable=not verbose):
            key, episode_key = jax.random.split(key)
            episode_records = self.evaluate_episode(
                key=episode_key,
                max_steps=max_steps,
                verbose=False
            )
            all_records.extend(episode_records)

            if verbose:
                avg_kl = np.mean([r.kl_score for r in episode_records]) if episode_records else 0.0
                print(f"Episode {episode + 1}/{n_episodes}: {len(episode_records)} steps, avg KL={avg_kl:.4f}")

        if verbose:
            print(f"\nTotal records collected: {len(all_records)}")

        return all_records

    # ------------------------------------------------------------------
    # Video and plot saving
    # ------------------------------------------------------------------

    def save_video(self, save_path: str, state_seq=None):
        """
        Save episode replay as GIF using JaxMARL's SMAXVisualizer.

        Args:
            save_path: Path to save the GIF (e.g., "episode_replay.gif").
            state_seq: List of (key, state, actions) tuples. If None, uses
                       the state sequence from the most recent evaluate_episode() call.
        """
        from jaxmarl.viz.visualizer import SMAXVisualizer

        seq = state_seq if state_seq is not None else self.last_state_seq
        if seq is None or len(seq) == 0:
            print("Warning: No state sequence available. Run evaluate_episode() first.")
            return

        viz = SMAXVisualizer(self.env, state_seq=seq)
        viz.animate(view=False, save_fname=save_path)
        self._log(f"Episode replay saved to: {save_path}")
        print(f"Episode replay GIF saved to: {save_path}")

    def save_plots(
        self,
        records: List[SMAXConsequenceRecord],
        save_dir: str,
        n_enemies: int,
    ):
        """
        Generate and save all consequence analysis plots.

        Args:
            records: List of SMAXConsequenceRecord from evaluate_episode().
            save_dir: Directory to save plots into.
            n_enemies: Number of enemies in the scenario (for action labeling).
        """
        from counterfactual_rl.visualization.smax_plots import SMAXConsequencePlotter

        os.makedirs(save_dir, exist_ok=True)

        plotter = SMAXConsequencePlotter(n_enemies=n_enemies)

        # Comprehensive multi-panel figure
        comprehensive_path = os.path.join(save_dir, "analysis_comprehensive.png")
        plotter.plot_comprehensive(records, save_path=comprehensive_path)

        # Individual plots
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        plotter.plot_histogram(records, ax=ax_hist)
        fig_hist.savefig(
            os.path.join(save_dir, "consequence_histogram.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_hist)

        fig_time, ax_time = plt.subplots(figsize=(12, 6))
        plotter.plot_consequence_over_time(records, ax=ax_time)
        fig_time.savefig(
            os.path.join(save_dir, "consequence_over_time.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_time)

        top_n = min(6, len(records))
        if top_n > 0:
            fig_returns = plotter.plot_return_distributions(records, top_n=top_n)
            fig_returns.savefig(
                os.path.join(save_dir, "return_distributions.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig_returns)

        # Print statistics to console
        plotter.print_statistics(records)

        self._log(f"All plots saved to: {save_dir}")
        print(f"All analysis plots saved to: {save_dir}")
