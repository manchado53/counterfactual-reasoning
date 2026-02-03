import sys
import os

import sys
import os

# Absolute paths
sys.path.insert(0, '/home/ad.msoe.edu/manchadoa/UR-RL/playing-with-smac/smac/agents')
sys.path.insert(0, '/home/ad.msoe.edu/manchadoa/UR-RL/playing-with-smac/smac/agents/PPO_one_action')
sys.path.insert(0, '/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/src')
os.environ['SC2PATH'] = '/home/ad.msoe.edu/manchadoa/UR-RL/playing-with-smac/StarCraftII'

import numpy as np
import matplotlib.pyplot as plt
from smac.env import StarCraft2Env
from counterfactual_rl.environments.smac import CentralizedSmacWrapper, SmacStateManager
from counterfactual_rl.analysis import MultiDiscreteCounterfactualAnalyzer
from counterfactual_rl.visualization.smac_plots import SmacConsequencePlotter
from utils import get_valid_actions
from random_policy import RandomPolicy

# Configuration
MAP_NAME = "3m"
MAX_STEPS = 300
HORIZON = 300
N_ROLLOUTS = 10
TOP_K = 2
LOG_FILE = "smac_counterfactual_analysis-random.log"


def main():
    # Create SMAC environment
    smac_env = StarCraft2Env(map_name=MAP_NAME, debug=True)
    env = CentralizedSmacWrapper(smac_env, use_state=True)
    
    # Create random policy
    policy = RandomPolicy(env=smac_env)
    
    # Create analyzer
    analyzer = MultiDiscreteCounterfactualAnalyzer(
        model=policy,
        env=env,
        state_manager=SmacStateManager,
        get_valid_actions_mask_fn=lambda: get_valid_actions(smac_env),
        get_action_probs_fn=None,
        n_agents=env.n_agents,
        n_actions=env.n_actions_per_agent,
        horizon=HORIZON,
        n_rollouts=N_ROLLOUTS,
        top_k=TOP_K,
        deterministic=False,
        log_file=LOG_FILE,
    )
    
    # Run analysis
    print("Running counterfactual analysis...")
    records = analyzer.evaluate_episode(max_steps=MAX_STEPS, verbose=True)
    print(f"\nCollected {len(records)} records")
    
    # Show ALL consequence scores
    if records:
        kl_scores = [r.kl_score for r in records]
        jsd_scores = [r.jsd_score for r in records]
        tv_scores = [r.tv_score for r in records]
        w_scores = [r.wasserstein_score for r in records]
        
        print("\nConsequence Scores Summary:")
        print(f"  KL Divergence   - Mean: {np.mean(kl_scores):.4f}, Max: {np.max(kl_scores):.4f}")
        print(f"  Jensen-Shannon  - Mean: {np.mean(jsd_scores):.4f}, Max: {np.max(jsd_scores):.4f}")
        print(f"  Total Variation - Mean: {np.mean(tv_scores):.4f}, Max: {np.max(tv_scores):.4f}")
        print(f"  Wasserstein     - Mean: {np.mean(w_scores):.4f}, Max: {np.max(w_scores):.4f}")
        
        most_consequential = max(records, key=lambda r: r.kl_score)
        print(f"\nMost consequential step (KL={most_consequential.kl_score:.4f}):")
        print(f"  Action: {most_consequential.action}")
        print(f"  JSD: {most_consequential.jsd_score:.4f}")
        print(f"  TV: {most_consequential.tv_score:.4f}")
        print(f"  W: {most_consequential.wasserstein_score:.4f}")
    
    # Generate plots and save in run directory
    plotter = SmacConsequencePlotter()
    plotter.plot_histogram(records)
    plt.savefig(os.path.join(analyzer.run_dir, 'consequence_histogram.png'))
    plt.close()
    
    plotter.plot_consequence_over_time(records)
    plt.savefig(os.path.join(analyzer.run_dir, 'consequence_over_time.png'))
    plt.close()
    
    plotter.plot_return_distributions(records, top_n=len(records))
    plt.savefig(os.path.join(analyzer.run_dir, 'return_distributions.png'))
    plt.close()
    
    plotter.print_statistics(records, top_n=5)
    
    # Cleanup
    smac_env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()