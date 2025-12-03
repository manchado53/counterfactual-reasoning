"""
Counterfactual rollout analysis for consequential states.

Environment-agnostic implementation that works with any registered environment.
"""

from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from counterfactual_rl.environments.base import StateManager
from counterfactual_rl.utils.data_structures import ConsequenceRecord
from counterfactual_rl.analysis.metrics import (
    compute_kl_divergence_kde,
    compute_jensen_shannon_divergence,
    compute_total_variation,
    compute_wasserstein_distance
)


class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis to identify consequential states.

    This class is environment-agnostic and works with any environment
    that has a corresponding StateManager implementation.

    Example:
        # For FrozenLake
        from counterfactual_rl.environments import registry
        state_manager = registry.get_state_manager("FrozenLake-v1")
        analyzer = CounterfactualAnalyzer(model, env, state_manager)

        # For Taxi-v3 (once registered)
        state_manager = registry.get_state_manager("Taxi-v3")
        analyzer = CounterfactualAnalyzer(model, env, state_manager)
        # Same analyzer code works!
    """

    def __init__(
        self,
        model: PPO,
        env: gym.Env,
        state_manager: StateManager,
        horizon: int = 20,
        n_rollouts: int = 48,
        gamma: float = 0.99,
        deterministic: bool = True
    ):
        """
        Initialize counterfactual analyzer.

        Args:
            model: Trained PPO model
            env: Gymnasium environment
            state_manager: StateManager instance for cloning/restoration
                          (environment-agnostic, works with any registered environment)
            horizon: Number of steps to roll out policy
            n_rollouts: Number of rollouts per action for distribution estimation
            gamma: Discount factor
            deterministic: Whether to use deterministic policy during rollouts.
                          True = argmax (same state -> same action, good for deterministic envs)
                          False = sample (stochastic, adds exploration, good for stochastic envs)
        """
        self.model = model
        self.env = env
        self.state_manager = state_manager
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.deterministic = deterministic
        self.action_space_size = env.action_space.n

    def perform_counterfactual_rollouts(
        self,
        state_dict: Dict
    ) -> Dict[int, np.ndarray]:
        """
        Perform counterfactual rollouts for all possible actions from a given state.

        For each action a:
        1. Restore the environment to the saved state
        2. Execute action a
        3. Roll out the policy for H steps
        4. Record the discounted return
        5. Repeat n_rollouts times to build a distribution

        Args:
            state_dict: Saved environment state

        Returns:
            Dictionary mapping action -> array of returns (shape: n_rollouts)
        """
        return_distributions = {}

        for action in range(self.action_space_size):
            returns = []

            for _ in range(self.n_rollouts):
                # Restore to the original state
                self.state_manager.restore_state(self.env, state_dict)

                # Execute the counterfactual action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Compute discounted return starting with this immediate reward
                total_return = reward
                discount = self.gamma

                # Roll out policy for remaining horizon
                if not done:
                    for step in range(self.horizon - 1):
                        # Get action from policy
                        action_pred, _ = self.model.predict(obs, deterministic=self.deterministic)
                        action_pred = int(action_pred)  # Convert numpy array to int

                        # Step environment
                        obs, reward, terminated, truncated, info = self.env.step(action_pred)
                        done = terminated or truncated

                        # Accumulate discounted reward
                        total_return += discount * reward
                        discount *= self.gamma

                        if done:
                            break

                returns.append(total_return)

            return_distributions[action] = np.array(returns)

        return return_distributions

    def compute_consequence_score(
        self,
        action: int,
        return_distributions: Dict[int, np.ndarray]
    ) -> tuple:
        """
        Compute consequence score for a state-action pair.

        The consequence score is the maximum KL divergence between the
        return distribution of the chosen action and any alternative action.

        Args:
            action: The action that was taken
            return_distributions: Return distributions for all actions

        Returns:
            Tuple of (consequence_score, kl_divergences_dict)
        """
        kl_divergences = {}
        chosen_returns = return_distributions[action]

        for alt_action in range(self.action_space_size):
            if alt_action != action:
                alt_returns = return_distributions[alt_action]
                kl_div = compute_kl_divergence_kde(chosen_returns, alt_returns)
                kl_divergences[alt_action] = kl_div

        # Consequence score = maximum KL divergence
        consequence_score = max(kl_divergences.values()) if kl_divergences else 0.0

        return consequence_score, kl_divergences

    def compute_all_metrics(
        self,
        action: int,
        return_distributions: Dict[int, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Compute all distributional metrics for a state-action pair.

        Computes KL divergence, Jensen-Shannon divergence, Total Variation distance,
        and Wasserstein distance between the chosen action's return distribution
        and all alternative actions.

        Args:
            action: The action that was taken
            return_distributions: Return distributions for all actions

        Returns:
            Dictionary with structure:
            {
                'kl': {'score': max_kl, 'divergences': {alt_action: kl_value, ...}},
                'jsd': {'score': max_jsd, 'divergences': {alt_action: jsd_value, ...}},
                'tv': {'score': max_tv, 'distances': {alt_action: tv_value, ...}},
                'wasserstein': {'score': max_wass, 'distances': {alt_action: wass_value, ...}}
            }
        """
        chosen_returns = return_distributions[action]

        # Initialize storage for all metrics
        kl_divergences = {}
        jsd_divergences = {}
        tv_distances = {}
        wasserstein_distances = {}

        # Compute all metrics for each alternative action
        for alt_action in range(self.action_space_size):
            if alt_action != action:
                alt_returns = return_distributions[alt_action]

                # KL divergence (asymmetric, unbounded)
                kl_div = compute_kl_divergence_kde(chosen_returns, alt_returns)
                kl_divergences[alt_action] = kl_div

                # Jensen-Shannon divergence (symmetric, bounded [0, ln(2)])
                jsd = compute_jensen_shannon_divergence(chosen_returns, alt_returns)
                jsd_divergences[alt_action] = jsd

                # Total Variation distance (symmetric, bounded [0, 1])
                tv = compute_total_variation(chosen_returns, alt_returns)
                tv_distances[alt_action] = tv

                # Wasserstein distance (symmetric, unbounded but interpretable)
                wass = compute_wasserstein_distance(chosen_returns, alt_returns)
                wasserstein_distances[alt_action] = wass

        # Compute consequence scores (maximum across alternatives)
        kl_score = max(kl_divergences.values()) if kl_divergences else 0.0
        jsd_score = max(jsd_divergences.values()) if jsd_divergences else 0.0
        tv_score = max(tv_distances.values()) if tv_distances else 0.0
        wasserstein_score = max(wasserstein_distances.values()) if wasserstein_distances else 0.0

        return {
            'kl': {'score': kl_score, 'divergences': kl_divergences},
            'jsd': {'score': jsd_score, 'divergences': jsd_divergences},
            'tv': {'score': tv_score, 'distances': tv_distances},
            'wasserstein': {'score': wasserstein_score, 'distances': wasserstein_distances}
        }

    def evaluate_episode(
        self,
        max_steps: int = 100,
        verbose: bool = False,
        compute_all_metrics: bool = False
    ) -> List[ConsequenceRecord]:
        """
        Evaluate consequential states for a single episode.

        Args:
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            compute_all_metrics: If True, compute all metrics (KL, JSD, TV, Wasserstein).
                                If False, compute only KL divergence (faster).

        Returns:
            List of ConsequenceRecord objects for the episode
        """
        obs, info = self.env.reset()
        done = False
        step = 0
        records = []

        while not done and step < max_steps:
            # Save current state
            current_state = self.state_manager.clone_state(self.env)
            current_position = obs

            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action = int(action)  # Convert numpy array to int

            # Perform counterfactual rollouts for all actions
            return_distributions = self.perform_counterfactual_rollouts(current_state)

            # Compute metrics
            if compute_all_metrics:
                # Compute all distributional metrics
                all_metrics = self.compute_all_metrics(action, return_distributions)

                # Extract KL as primary
                consequence_score = all_metrics['kl']['score']
                kl_divergences = all_metrics['kl']['divergences']

                # Extract additional metrics
                jsd_score = all_metrics['jsd']['score']
                jsd_divergences = all_metrics['jsd']['divergences']
                tv_score = all_metrics['tv']['score']
                tv_distances = all_metrics['tv']['distances']
                wasserstein_score = all_metrics['wasserstein']['score']
                wasserstein_distances = all_metrics['wasserstein']['distances']
            else:
                # Compute only KL divergence (default, faster)
                consequence_score, kl_divergences = self.compute_consequence_score(
                    action, return_distributions
                )
                jsd_score = jsd_divergences = None
                tv_score = tv_distances = None
                wasserstein_score = wasserstein_distances = None

            # BUG FIX: Use current_position (saved before rollouts) instead of getting state from env
            # After perform_counterfactual_rollouts(), self.env is in an arbitrary state!
            # current_position was saved at line 261 BEFORE rollouts, so it has the correct state
            state_value = current_position
            
            # Get position info from StateManager (but restore state first to get correct position)
            self.state_manager.restore_state(self.env, current_state)
            state_info = self.state_manager.get_state_info(self.env)
            position = state_info.get('position', None)

            # Create record
            record = ConsequenceRecord(
                state=state_value,
                action=action,
                position=position,
                consequence_score=consequence_score,
                kl_divergences=kl_divergences,
                return_distributions=return_distributions,
                jsd_score=jsd_score,
                jsd_divergences=jsd_divergences,
                tv_score=tv_score,
                tv_distances=tv_distances,
                wasserstein_score=wasserstein_score,
                wasserstein_distances=wasserstein_distances
            )
            records.append(record)

            if verbose:
                if compute_all_metrics:
                    print(f"  Step {step}: state={state_value}, action={action}, "
                          f"KL={consequence_score:.4f}, JSD={jsd_score:.4f}, "
                          f"TV={tv_score:.4f}, W={wasserstein_score:.4f}")
                else:
                    print(f"  Step {step}: state={state_value}, action={action}, "
                          f"consequence={consequence_score:.4f}")

            # Execute the chosen action in the actual environment
            # IMPORTANT: Restore state before stepping (counterfactual_rollouts leaves env in arbitrary state)
            self.state_manager.restore_state(self.env, current_state)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step += 1

        return records

    def evaluate_multiple_episodes(
        self,
        n_episodes: int = 20,
        verbose: bool = True,
        compute_all_metrics: bool = False
    ) -> List[ConsequenceRecord]:
        """
        Evaluate consequential states across multiple episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            verbose: Whether to print progress
            compute_all_metrics: If True, compute all metrics (KL, JSD, TV, Wasserstein).
                                If False, compute only KL divergence (faster).

        Returns:
            List of all ConsequenceRecord objects
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Consequential States")
            print(f"Episodes: {n_episodes}, Horizon: {self.horizon}, "
                  f"Rollouts/action: {self.n_rollouts}")
            if compute_all_metrics:
                print(f"Metrics: KL divergence, Jensen-Shannon, Total Variation, Wasserstein")
            else:
                print(f"Metric: KL divergence only")
            print(f"{'='*60}\n")

        all_records = []

        for episode in range(n_episodes):
            if verbose:
                print(f"Episode {episode + 1}/{n_episodes}...", end=" ")

            episode_records = self.evaluate_episode(
                verbose=False,
                compute_all_metrics=compute_all_metrics
            )
            all_records.extend(episode_records)

            if verbose:
                print(f"Recorded {len(episode_records)} state-action pairs")

        if verbose:
            print(f"\nTotal records collected: {len(all_records)}")

        return all_records
