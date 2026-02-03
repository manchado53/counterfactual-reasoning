"""
Example: SMAX counterfactual analysis with random policy.

Demonstrates usage of SMAXCounterfactualAnalyzer with a simple random policy.

Usage:
    python -m counterfactual_rl.simulations.smax_random_policy
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario

from counterfactual_rl.analysis.smax_counterfactual import SMAXCounterfactualAnalyzer
from counterfactual_rl.utils.smax_utils import get_agent_names


# Configuration
SCENARIO = "3m"  # 3 marines vs 3 marines
MAX_STEPS = 50
HORIZON = 15
N_ROLLOUTS = 10
TOP_K = 5
SEED = 42


def create_random_policy(env):
    """
    Create a random policy function for SMAX.

    The policy samples uniformly from valid actions for each agent.

    Args:
        env: SMAX environment

    Returns:
        Function (key, obs, avail_actions) -> action_dict
    """
    agent_names = get_agent_names(env)

    def random_policy(key, obs, avail_actions):
        """Sample random valid actions for all agents."""
        actions = {}
        keys = jax.random.split(key, len(agent_names))

        for i, agent in enumerate(agent_names):
            mask = avail_actions[agent]
            # Find valid action indices
            valid_indices = jnp.where(mask == 1)[0]
            # Sample uniformly from valid actions
            action_idx = jax.random.choice(keys[i], valid_indices)
            actions[agent] = action_idx

        return actions

    return random_policy


def main():
    """Run SMAX counterfactual analysis with random policy."""
    print("=" * 60)
    print("SMAX Counterfactual Analysis - Random Policy Demo")
    print("=" * 60)

    # Create SMAX environment
    print(f"\nCreating SMAX environment with scenario: {SCENARIO}")
    scenario = map_name_to_scenario(SCENARIO)
    env = make("HeuristicEnemySMAX", scenario=scenario)

    print(f"  Agents: {env.agents}")
    print(f"  Num agents: {len(env.agents)}")

    # Create random policy
    print("\nCreating random policy...")
    policy_fn = create_random_policy(env)

    # Create analyzer
    print(f"\nInitializing SMAXCounterfactualAnalyzer:")
    print(f"  horizon={HORIZON}, n_rollouts={N_ROLLOUTS}, top_k={TOP_K}")

    analyzer = SMAXCounterfactualAnalyzer(
        env=env,
        policy_fn=policy_fn,
        get_action_probs_fn=None,  # Uniform for random policy
        horizon=HORIZON,
        n_rollouts=N_ROLLOUTS,
        top_k=TOP_K,
        store_states=False
    )

    # Run analysis
    print(f"\nRunning counterfactual analysis (max_steps={MAX_STEPS})...")
    key = jax.random.PRNGKey(SEED)
    records = analyzer.evaluate_episode(key=key, max_steps=MAX_STEPS, verbose=True)

    # Display results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    print(f"\nCollected {len(records)} consequence records")

    if records:
        kl_scores = [r.kl_score for r in records]
        jsd_scores = [r.jsd_score for r in records if r.jsd_score is not None]

        print(f"\nKL Divergence Statistics:")
        print(f"  Mean: {np.mean(kl_scores):.4f}")
        print(f"  Std:  {np.std(kl_scores):.4f}")
        print(f"  Max:  {np.max(kl_scores):.4f}")
        print(f"  Min:  {np.min(kl_scores):.4f}")

        if jsd_scores:
            print(f"\nJensen-Shannon Divergence Statistics:")
            print(f"  Mean: {np.mean(jsd_scores):.4f}")
            print(f"  Max:  {np.max(jsd_scores):.4f}")

        # Find most consequential moment
        most_consequential = max(records, key=lambda r: r.kl_score)
        print(f"\nMost Consequential Moment:")
        print(f"  Timestep: {most_consequential.timestep}")
        print(f"  Action: {most_consequential.action}")
        print(f"  KL Score: {most_consequential.kl_score:.4f}")

        # Show alternative that would have been most different
        alt_action, alt_kl = most_consequential.get_most_different_action()
        if alt_action:
            print(f"  Most different alternative: {alt_action} (KL={alt_kl:.4f})")

    print(f"\nResults saved to: {analyzer.run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
