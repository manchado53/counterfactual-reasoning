"""
Counterfactual rollout analysis for MultiDiscrete action spaces (e.g., SMAC).

Uses beam search to efficiently select top-K most probable joint actions
without exhaustive enumeration of the combinatorial action space.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm
import logging
from datetime import datetime

from counterfactual_rl.environments.base import StateManager
from counterfactual_rl.utils.data_structures import ConsequenceRecord
from counterfactual_rl.utils.action_selection import beam_search_top_k_joint_actions
from counterfactual_rl.analysis.metrics import (
    compute_kl_divergence_kde,
    compute_jensen_shannon_divergence,
    compute_total_variation,
    compute_wasserstein_distance
)


class MultiDiscreteCounterfactualAnalyzer:
    """
    Counterfactual analyzer for MultiDiscrete action spaces (e.g., SMAC).
    
    Uses beam search to efficiently select top-K most probable joint actions
    for counterfactual rollouts.
    
    Example:
        >>> analyzer = MultiDiscreteCounterfactualAnalyzer(
        ...     model=ppo_model,
        ...     env=smac_wrapper,
        ...     state_manager=smac_state_manager,
        ...     get_valid_actions_fn=lambda: get_valid_actions(smac_env),
        ...     get_action_probs_fn=lambda obs: get_action_probs(ppo, obs),  # optional
        ...     top_k=20
        ... )
        >>> records = analyzer.evaluate_episode()
    """
    
    def __init__(
        self,
        model,  # Your custom PPO model
        env: gym.Env,
        state_manager: StateManager,
        get_valid_actions_fn: Callable[[], List[List[int]]],
        get_action_probs_fn: Optional[Callable[[np.ndarray], List[Dict[int, float]]]] = None,
        n_agents: int = None,
        n_actions: int = None,
        horizon: int = 20,
        n_rollouts: int = 48,
        gamma: float = 0.99,
        deterministic: bool = True,
        top_k: int = 20, 
        log_file: str = None
    ):
        """
        Initialize MultiDiscrete counterfactual analyzer.
        
        Args:
            model: Trained policy model with .predict(obs, deterministic) method
            env: Environment with MultiDiscrete action space
            state_manager: StateManager for cloning/restoring environment state
            get_valid_actions_fn: Function that returns valid action indices per agent.
                                 Returns List[List[int]], e.g., [[0,1,4], [2,3], [1,4,5]]
            get_action_probs_fn: Optional function that takes obs and returns action probs.
                                Returns List[Dict[int, float]], e.g., [{0: 0.5, 1: 0.3}, ...]
                                If None, uses uniform probability over valid actions.
            n_agents: Number of agents (optional, for reference)
            n_actions: Number of actions per agent (optional, for reference)
            horizon: Number of steps to roll out policy
            n_rollouts: Number of rollouts per action for distribution estimation
            gamma: Discount factor
            deterministic: Whether to use deterministic policy during rollouts
            top_k: Number of top joint actions to evaluate (default: 20)
        """
        self.model = model
        self.env = env
        self.state_manager = state_manager
        self.get_valid_actions_fn = get_valid_actions_fn
        self.get_action_probs_fn = get_action_probs_fn
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.deterministic = deterministic
        self.top_k = top_k
        
        # Setup file logging
        self.log_file = log_file
        self.logger = None
        self.setup_logging(log_file)
    
    def setup_logging(self, log_file: str = None):
        """Setup file-based logging for analysis."""
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"counterfactual_analysis_{timestamp}.log"
        
        self.log_file = log_file
        self.logger = logging.getLogger(f"CounterfactualAnalyzer_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info("="*80)
        self.logger.info("Counterfactual Analysis Started")
        self.logger.info(f"Configuration: n_agents={self.n_agents}, n_actions={self.n_actions}")
        self.logger.info(f"Parameters: horizon={self.horizon}, n_rollouts={self.n_rollouts}, top_k={self.top_k}")
        self.logger.info(f"Gamma={self.gamma}, Deterministic={self.deterministic}")
        self.logger.info("="*80)
    
    def _log(self, message: str, level: str = "info"):
        """Log message to file if logging is enabled."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
    
    
    def perform_counterfactual_rollouts(
        self,
        state_dict: Dict,
        obs: np.ndarray,
        verbose: bool = False
    ) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Perform counterfactual rollouts for top-K most probable joint actions.
        
        Args:
            state_dict: Saved environment state
            obs: Current observation (needed to compute action probabilities)
            verbose: Whether to print progress
        
        Returns:
            Dictionary mapping joint_action (tuple) -> array of returns (shape: n_rollouts)
        """
        return_distributions = {}
        
        self._log("Starting counterfactual rollouts")
        
        # Get valid actions for each agent
        valid_actions = self.get_valid_actions_fn()
        self._log(f"Valid actions per agent: {[len(v) for v in valid_actions]}")
        
        # Get action probabilities (optional - if None, beam search uses uniform)
        action_probs = None
        if self.get_action_probs_fn is not None:
            action_probs = self.get_action_probs_fn(obs)
            self._log("Action probabilities computed")
        else:
            self._log("Using uniform action probabilities")
        
        # Use beam search to find top-K joint actions
        actions_to_evaluate = beam_search_top_k_joint_actions(
            valid_actions=valid_actions,
            action_probs=action_probs,
            k=self.top_k
        )
        
        self._log(f"Beam search returned {len(actions_to_evaluate)} joint actions to evaluate")
        self._log(f"Actions: {actions_to_evaluate}")
        
        # Perform rollouts for each selected joint action
        for action_idx, joint_action in enumerate(tqdm(actions_to_evaluate, desc="Joint actions", leave=False, disable=not verbose)):
            returns = []
            self._log(f"\n--- Action {action_idx+1}/{len(actions_to_evaluate)}: {joint_action} ---")
            
            for rollout_idx in range(self.n_rollouts):
                self._log(f"  Rollout {rollout_idx+1}/{self.n_rollouts}")
                
                # Restore to the original state
                self.state_manager.restore_state(self.env, state_dict)
                
                # Execute the counterfactual joint action
                obs_next, reward, terminated, truncated, info = self.env.step(list(joint_action))
                done = terminated or truncated
                
                self._log(f"    Initial step: reward={reward:.4f}, done={done}")
                
                # Compute discounted return
                total_return = reward
                discount = self.gamma
                step_rewards = [reward]
                
                # Roll out policy for remaining horizon
                if not done:
                    for step in range(self.horizon - 1):
                        action_pred, _ = self.model.predict(obs_next, deterministic=self.deterministic)
                        obs_next, reward, terminated, truncated, info = self.env.step(list(action_pred))
                        done = terminated or truncated
                        
                        step_rewards.append(reward)
                        total_return += discount * reward
                        discount *= self.gamma
                        
                        self._log(f"    Step {step+1}: action={tuple(action_pred)}, reward={reward:.4f}, done={done}")
                        
                        if done:
                            self._log(f"    Episode terminated at step {step+1}")
                            break
                
                returns.append(total_return)
                self._log(f"    Total return: {total_return:.4f} (from {len(step_rewards)} rewards: {[f'{r:.3f}' for r in step_rewards]})")
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            self._log(f"  Action {joint_action} summary: mean_return={mean_return:.4f}, std={std_return:.4f}")
            self._log(f"  All returns: {[f'{r:.3f}' for r in returns]}")
            
            return_distributions[joint_action] = np.array(returns)
        
        return return_distributions
    
    def compute_consequence_score(
        self,
        action: Tuple[int, ...],
        return_distributions: Dict[Tuple[int, ...], np.ndarray]
    ) -> Tuple[float, Dict[Tuple[int, ...], float]]:
        """
        Compute consequence score for a state-action pair.
        
        The consequence score is the maximum KL divergence between the
        return distribution of the chosen action and any alternative action.
        """
        kl_divergences = {}
        chosen_returns = return_distributions.get(action)
        
        # If chosen action not in top-K, we can't compute score
        if chosen_returns is None:
            return 0.0, {}
        
        for alt_action in return_distributions.keys():
            if alt_action != action:
                alt_returns = return_distributions[alt_action]
                kl_div = compute_kl_divergence_kde(chosen_returns, alt_returns)
                kl_divergences[alt_action] = kl_div
        
        consequence_score = max(kl_divergences.values()) if kl_divergences else 0.0
        return consequence_score, kl_divergences
    
    def evaluate_episode(
        self,
        max_steps: int = 100,
        verbose: bool = True
    ) -> List[ConsequenceRecord]:
        """
        Evaluate consequential states for a single episode.
        
        Args:
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
        
        Returns:
            List of ConsequenceRecord objects for the episode
        """
        obs, info = self.env.reset()
        done = False
        step = 0
        records = []
        
        self._log("\n" + "="*80)
        self._log(f"Starting Episode Evaluation (max_steps={max_steps})")
        self._log("="*80)
        
        pbar = tqdm(total=max_steps, desc="Episode steps", disable=not verbose)
        while not done and step < max_steps:
            pbar.update(1)
            
            self._log(f"\n{'='*40} STEP {step} {'='*40}")
            
            # Save current state
            current_state = self.state_manager.clone_state(self.env)
            self._log(f"State cloned at step {step}")
            
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action_tuple = tuple(action)
            
            self._log(f"Policy selected action: {action_tuple}")
            self._log(f"Starting counterfactual analysis: {self.top_k} actions x {self.n_rollouts} rollouts")
            
            # Perform counterfactual rollouts for top-K actions
            return_distributions = self.perform_counterfactual_rollouts(current_state, obs, verbose=verbose)
            
            self._log(f"Evaluated {len(return_distributions)} joint actions")
            
            # Compute consequence score
            consequence_score, kl_divergences = self.compute_consequence_score(
                action_tuple, return_distributions
            )
            
            self._log(f"Consequence score: {consequence_score:.6f}")
            if kl_divergences:
                self._log("KL divergences to alternative actions:")
                for alt_action, kl in kl_divergences.items():
                    self._log(f"  {alt_action}: {kl:.6f}")
            
            if verbose:
                pbar.set_postfix({
                    'action': str(action_tuple),
                    'consequence': f"{consequence_score:.4f}"
                })
            
            # Restore state before getting info
            self.state_manager.restore_state(self.env, current_state)
            state_info = self.state_manager.get_state_info(self.env)
            position = state_info.get('position', None)
            
            # Create record
            record = ConsequenceRecord(
                state=obs,
                action=action_tuple,
                position=position,
                consequence_score=consequence_score,
                kl_divergences=kl_divergences,
                return_distributions=return_distributions
            )
            records.append(record)
            self._log(f"Record created and saved")
            
            # Execute the chosen action
            self.state_manager.restore_state(self.env, current_state)
            obs, reward, terminated, truncated, info = self.env.step(list(action))
            done = terminated or truncated
            
            self._log(f"Actual step executed: reward={reward:.4f}, done={done}")
            if done:
                self._log(f"Episode terminated at step {step}")
            
            step += 1
        
        pbar.close()
        
        self._log("\n" + "="*80)
        self._log(f"Episode Evaluation Complete: {len(records)} records collected")
        self._log("="*80)
        
        if self.logger:
            # Log summary statistics
            if records:
                scores = [r.consequence_score for r in records]
                self._log(f"\nSummary Statistics:")
                self._log(f"  Mean consequence score: {np.mean(scores):.6f}")
                self._log(f"  Max consequence score: {np.max(scores):.6f}")
                self._log(f"  Min consequence score: {np.min(scores):.6f}")
                self._log(f"  Std consequence score: {np.std(scores):.6f}")
        
        return records
    
    def evaluate_multiple_episodes(
        self,
        n_episodes: int = 20,
        verbose: bool = True
    ) -> List[ConsequenceRecord]:
        """Evaluate consequential states across multiple episodes."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"MultiDiscrete Counterfactual Analysis")
            print(f"Agents: {self.n_agents}, Actions/agent: {self.n_actions}")
            print(f"Top-K: {self.top_k}, Horizon: {self.horizon}, Rollouts: {self.n_rollouts}")
            print(f"{'='*60}\n")
        
        all_records = []
        
        for episode in tqdm(range(n_episodes), desc="Episodes", disable=not verbose):
            episode_records = self.evaluate_episode(verbose=verbose)
            all_records.extend(episode_records)
        
        if verbose:
            print(f"\nTotal records collected: {len(all_records)}")
        
        return all_records
