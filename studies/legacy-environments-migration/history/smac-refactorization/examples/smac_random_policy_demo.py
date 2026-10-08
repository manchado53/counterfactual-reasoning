"""
SMAC Counterfactual Analysis with Random Policy

This example demonstrates how to use the CounterfactualAnalyzer with SMAC
using a random policy as a baseline. This is useful for:
1. Testing the analysis pipeline
2. Understanding what "non-consequential" decisions look like
3. Debugging before using a trained policy

Note: The CounterfactualAnalyzer still needs to be updated to support
      MultiDiscrete action spaces. This script shows the intended usage.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smac.env import StarCraft2Env
import numpy as np

from counterfactual_rl.environments.smac import CentralizedSmacWrapper, SmacStateManager
from counterfactual_rl.policies import RandomPolicy

# Set SC2 Path
os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'


def test_random_policy_smac():
    """
    Test the random policy with SMAC environment.
    
    This demonstrates:
    1. Creating a wrapped SMAC environment
    2. Using the RandomPolicy
    3. Running episodes with action masking
    """
    print("="*60)
    print("Testing Random Policy with SMAC")
    print("="*60)
    
    # 1. Create SMAC environment
    print("\n1. Creating SMAC environment (3m map)...")
    smac_env = StarCraft2Env(map_name="3m")
    wrapped_env = CentralizedSmacWrapper(smac_env, use_state=True)
    
    print(f"   Environment created!")
    print(f"   - Agents: {wrapped_env.n_agents}")
    print(f"   - Actions per agent: {wrapped_env.n_actions_per_agent}")
    print(f"   - Action space: {wrapped_env.action_space}")
    print(f"   - Observation space: {wrapped_env.observation_space}")
    
    # 2. Create random policy
    print("\n2. Creating random policy...")
    policy = RandomPolicy(smac_env)
    print("   Random policy created!")
    
    # 3. Run a few episodes
    print("\n3. Running episodes with random policy...")
    n_episodes = 3
    
    for episode in range(n_episodes):
        obs, info = wrapped_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\n   Episode {episode + 1}:")
        
        while not done and step < 50:  # Limit steps
            # Get action from random policy
            actions, _ = policy.predict(obs, deterministic=False)
            
            # Step environment
            obs, reward, terminated, truncated, info = wrapped_env.step(actions)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
            
            if step % 10 == 0:
                print(f"     Step {step}: reward={reward:.2f}, total={total_reward:.2f}")
        
        print(f"   Episode {episode + 1} finished: {step} steps, total reward: {total_reward:.2f}")
    
    # 4. Close environment
    smac_env.close()
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)



if __name__ == "__main__":
    # Test random policy
    test_random_policy_smac()
    
