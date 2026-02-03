"""
Counterfactual rollout analysis for SMAX (JaxMARL) environments.

Key differences from SMAC version:
1. No wrapper needed - work directly with SMAX functional API
2. No StateManager needed - state is just data, save it directly
3. Explicit RNG key management throughout
4. Single environment - no need for separate rollout env
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

from counterfactual_rl.utils.smax_data_structures import SMAXConsequenceRecord
from counterfactual_rl.utils.smax_utils import (
    tuple_to_action_dict,
    action_dict_to_tuple,
    get_valid_actions_mask,
    get_agent_names,
    sum_rewards,
)
from counterfactual_rl.utils.action_selection import (
    beam_search_top_k_joint_actions,
    convert_mask_to_indices,
)
from counterfactual_rl.analysis.metrics import compute_all_consequence_metrics


class SMAXCounterfactualAnalyzer:
    """
    Counterfactual analyzer for SMAX (JaxMARL) environments.

    Leverages SMAX's functional API with explicit state passing for
    efficient counterfactual rollouts. State can be "saved" by simply
    keeping a reference - no replay strategy needed.

    Example:
        >>> from jaxmarl import make
        >>> from jaxmarl.environments.smax import map_name_to_scenario
        >>>
        >>> scenario = map_name_to_scenario("3m")
        >>> env = make("SMAX", scenario=scenario)
        >>>
        >>> def random_policy(key, obs, avail_actions):
        ...     # Your policy here
        ...     pass
        >>>
        >>> analyzer = SMAXCounterfactualAnalyzer(
        ...     env=env,
        ...     policy_fn=random_policy,
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
        policy_fn: Callable,  # (key, obs, avail_actions) -> action_dict
        get_action_probs_fn: Optional[Callable] = None,
        horizon: int = 20,
        n_rollouts: int = 48,
        gamma: float = 0.99,
        top_k: int = 20,
        log_file: Optional[str] = None,
        store_states: bool = False,
    ):
        """
        Initialize SMAX counterfactual analyzer.

        Args:
            env: SMAX environment (from jaxmarl.make())
            policy_fn: Function (key, obs, avail_actions) -> action_dict
                       Takes RNG key, observation dict, and available actions,
                       returns action dict for all agents
            get_action_probs_fn: Optional function (obs, masks) -> List[Dict[int, float]]
                                 Returns per-agent action probabilities for beam search.
                                 If None, uses uniform distribution.
            horizon: Number of steps to roll out policy
            n_rollouts: Number of rollouts per action for distribution estimation
            gamma: Discount factor
            top_k: Number of top joint actions to evaluate
            log_file: Path to log file (optional)
            store_states: Whether to store full JAX states in records
        """
        self.env = env
        self.policy_fn = policy_fn
        self.get_action_probs_fn = get_action_probs_fn
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.top_k = top_k
        self.store_states = store_states

        # Extract environment info
        self.agent_names = get_agent_names(env)
        self.n_agents = len(self.agent_names)

        # Get action space size (assuming homogeneous agents)
        self.n_actions = env.action_space(self.agent_names[0]).n

        # Setup logging
        self.logger = None
        self.run_dir = None
        self.log_file = log_file
        self._setup_logging(log_file)

    def _setup_logging(self, log_file: Optional[str] = None):
        """Setup file-based logging for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("runs", f"smax_run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        if log_file is None:
            log_filename = "smax_counterfactual_analysis.log"
        else:
            log_filename = os.path.basename(log_file)

        self.log_file = os.path.join(self.run_dir, log_filename)
        self.logger = logging.getLogger(f"SMAXCounterfactualAnalyzer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self._log("=" * 80)
        self._log("SMAX Counterfactual Analysis Started")
        self._log(f"Configuration: n_agents={self.n_agents}, n_actions={self.n_actions}")
        self._log(f"Parameters: horizon={self.horizon}, n_rollouts={self.n_rollouts}, top_k={self.top_k}")
        self._log(f"Gamma={self.gamma}")
        self._log("=" * 80)

    def _log(self, message: str, level: str = "info"):
        """Log message to file if logging is enabled."""
        if self.logger:
            getattr(self.logger, level)(message)

    def _rollout_single_action(
        self,
        key: jax.Array,
        state: Any,
        obs: Dict[str, jnp.ndarray],
        first_action: Tuple[int, ...]
    ) -> float:
        """
        Perform a single rollout from saved state with given first action.

        Args:
            key: RNG key for this rollout
            state: Saved JAX state to start from
            obs: Observations at saved state (unused, we use state directly)
            first_action: First action to take (as tuple)

        Returns:
            Discounted return for this rollout
        """
        # Convert first action to dict
        action_dict = tuple_to_action_dict(first_action, self.agent_names)

        # Take first action (counterfactual action)
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = self.env.step(step_key, state, action_dict)

        # Sum rewards across agents for team reward
        total_reward = sum_rewards(rewards, self.agent_names)
        total_return = total_reward
        discount = self.gamma

        # Check if episode ended
        done = dones.get("__all__", False)

        # Continue rollout with policy for remaining horizon
        for step in range(self.horizon - 1):
            if done:
                break

            # Get available actions
            avail_actions = self.env.get_avail_actions(state)

            # Get action from policy
            key, policy_key = jax.random.split(key)
            action_dict = self.policy_fn(policy_key, obs, avail_actions)

            # Step environment
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, action_dict)

            # Accumulate discounted reward
            total_reward = sum_rewards(rewards, self.agent_names)
            total_return += discount * total_reward
            discount *= self.gamma

            done = dones.get("__all__", False)

        return total_return

    def perform_counterfactual_rollouts(
        self,
        key: jax.Array,
        state: Any,
        obs: Dict[str, jnp.ndarray],
        actual_action: Tuple[int, ...],
        current_timestep: int = 0,
        verbose: bool = False
    ) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Perform counterfactual rollouts for top-K most probable joint actions.

        Args:
            key: RNG key for rollouts
            state: Saved JAX state (can be used directly - no restoration needed!)
            obs: Current observations
            actual_action: The actual action taken by policy
            current_timestep: Current timestep (for logging)
            verbose: Whether to print progress

        Returns:
            Dictionary mapping joint_action (tuple) -> array of returns
        """
        return_distributions: Dict[Tuple[int, ...], np.ndarray] = {}

        self._log(f"Starting counterfactual rollouts at step {current_timestep}")

        # Get valid action masks
        valid_actions_mask = get_valid_actions_mask(self.env, state)

        # Get action probabilities for beam search
        action_probs = None
        if self.get_action_probs_fn is not None:
            action_probs = self.get_action_probs_fn(obs, valid_actions_mask)
            self._log("Action probabilities computed")
        else:
            self._log("Using uniform action probabilities")

        # Convert masks to indices
        valid_actions = convert_mask_to_indices(valid_actions_mask)

        # Beam search for top-K actions
        actions_to_evaluate = beam_search_top_k_joint_actions(
            valid_actions=valid_actions,
            action_probs=action_probs,
            k=self.top_k
        )

        # Ensure actual action is included
        if actual_action not in actions_to_evaluate:
            actions_to_evaluate = [actual_action] + actions_to_evaluate[:self.top_k - 1]
            self._log(f"Added actual action {actual_action} to evaluation set")

        self._log(f"Evaluating {len(actions_to_evaluate)} joint actions")

        # Split key for all rollouts upfront
        key, rollout_key = jax.random.split(key)

        # Evaluate each action
        iterator = tqdm(actions_to_evaluate, desc="Joint actions", leave=False, disable=not verbose)
        for action_idx, joint_action in enumerate(iterator):
            returns = []
            self._log(f"\n--- Action {action_idx + 1}/{len(actions_to_evaluate)}: {joint_action} ---")

            # Split keys for all rollouts of this action
            rollout_keys = jax.random.split(rollout_key, self.n_rollouts + 1)
            rollout_key = rollout_keys[0]  # For next action
            action_keys = rollout_keys[1:]  # For this action's rollouts

            for rollout_idx in range(self.n_rollouts):
                # KEY SIMPLIFICATION: Just use the saved state directly!
                # No reset, no replay - state is immutable JAX data
                rollout_return = self._rollout_single_action(
                    key=action_keys[rollout_idx],
                    state=state,  # Directly use saved state
                    obs=obs,
                    first_action=joint_action
                )
                returns.append(rollout_return)
                self._log(f"  Rollout {rollout_idx + 1}/{self.n_rollouts}: return={rollout_return:.4f}")

            mean_return = np.mean(returns)
            std_return = np.std(returns)
            self._log(f"  Action {joint_action} summary: mean={mean_return:.4f}, std={std_return:.4f}")

            return_distributions[joint_action] = np.array(returns)

        return return_distributions

    def evaluate_episode(
        self,
        key: jax.Array,
        max_steps: int = 100,
        verbose: bool = True
    ) -> List[SMAXConsequenceRecord]:
        """
        Evaluate consequential states for a single episode.

        Args:
            key: RNG key for the episode
            max_steps: Maximum steps per episode
            verbose: Whether to print progress

        Returns:
            List of SMAXConsequenceRecord objects
        """
        # Split key for reset and steps
        key, reset_key = jax.random.split(key)

        # Reset environment
        obs, state = self.env.reset(reset_key)

        done = False
        step = 0
        records: List[SMAXConsequenceRecord] = []
        episode_return = 0.0
        episode_start_time = time.time()

        self._log("\n" + "=" * 80)
        self._log(f"Starting Episode Evaluation (max_steps={max_steps})")
        self._log("=" * 80)

        pbar = tqdm(total=max_steps, desc="Episode steps", disable=not verbose)

        while not done and step < max_steps:
            pbar.update(1)
            self._log(f"\n{'=' * 40} STEP {step} {'=' * 40}")

            # SAVE STATE: Just keep references - JAX states are immutable!
            saved_state = state
            saved_obs = obs

            # Get available actions
            avail_actions = self.env.get_avail_actions(state)

            # Get action from policy
            key, policy_key = jax.random.split(key)
            action_dict = self.policy_fn(policy_key, obs, avail_actions)
            action_tuple = action_dict_to_tuple(action_dict, self.agent_names)

            self._log(f"Policy selected action: {action_tuple}")

            # Perform counterfactual analysis
            key, counterfactual_key = jax.random.split(key)
            counterfactual_start = time.time()

            return_distributions = self.perform_counterfactual_rollouts(
                key=counterfactual_key,
                state=saved_state,
                obs=saved_obs,
                actual_action=action_tuple,
                current_timestep=step,
                verbose=verbose
            )

            counterfactual_duration = time.time() - counterfactual_start
            self._log(f"Counterfactual analysis took {counterfactual_duration:.2f}s")

            # Compute consequence metrics
            all_metrics = compute_all_consequence_metrics(action_tuple, return_distributions)

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

            # Execute actual step in main trajectory
            key, step_key = jax.random.split(key)
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

        Args:
            key: RNG key
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            verbose: Whether to print progress

        Returns:
            List of all SMAXConsequenceRecord objects
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SMAX Counterfactual Analysis")
            print(f"Agents: {self.n_agents}, Actions/agent: {self.n_actions}")
            print(f"Top-K: {self.top_k}, Horizon: {self.horizon}, Rollouts: {self.n_rollouts}")
            print(f"{'=' * 60}\n")

        all_records: List[SMAXConsequenceRecord] = []

        for episode in tqdm(range(n_episodes), desc="Episodes", disable=not verbose):
            key, episode_key = jax.random.split(key)
            episode_records = self.evaluate_episode(
                key=episode_key,
                max_steps=max_steps,
                verbose=False  # Don't show per-step progress for multiple episodes
            )
            all_records.extend(episode_records)

            if verbose:
                avg_kl = np.mean([r.kl_score for r in episode_records]) if episode_records else 0.0
                print(f"Episode {episode + 1}/{n_episodes}: {len(episode_records)} steps, avg KL={avg_kl:.4f}")

        if verbose:
            print(f"\nTotal records collected: {len(all_records)}")

        return all_records
