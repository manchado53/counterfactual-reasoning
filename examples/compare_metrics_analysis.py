"""
Compare all metrics (KL, JSD, TV, Wasserstein) on FrozenLake counterfactual analysis

This script runs a full analysis with all metrics enabled and generates:
1. Side-by-side heatmaps for all metrics
2. Correlation plots showing agreement between metrics
3. Statistical comparison of metric rankings
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from counterfactual_rl.environments import FrozenLakeStateManager
from counterfactual_rl.analysis import CounterfactualAnalyzer
from counterfactual_rl.visualization import ConsequencePlotter

# Configuration
ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False
MAP_NAME = "4x4"
MODEL_PATH = "models/ppo_nonslippery_demo"
N_EPISODES = 20
HORIZON = 20
N_ROLLOUTS = 50

print("Multi-Metric Counterfactual Analysis")
print("="*70)
print(f"Environment: {ENV_ID} (slippery={IS_SLIPPERY})")
print(f"Episodes: {N_EPISODES}, Horizon: {HORIZON}, Rollouts: {N_ROLLOUTS}")
print(f"Metrics: KL divergence, Jensen-Shannon, Total Variation, Wasserstein")
print("="*70)

# Load model and create environment
env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY, map_name=MAP_NAME)
model = PPO.load(MODEL_PATH, env=env)
print("[OK] Model loaded")

# Create analyzer with ALL METRICS enabled
state_manager = FrozenLakeStateManager()
analyzer = CounterfactualAnalyzer(
    model=model,
    env=env,
    state_manager=state_manager,
    horizon=HORIZON,
    n_rollouts=N_ROLLOUTS,
    gamma=0.99,
    deterministic=True
)
print("[OK] Analyzer created")

# Run analysis with ALL METRICS
print("\nRunning counterfactual analysis with all metrics...")
print("(This will take longer than KL-only analysis)")
records = analyzer.evaluate_multiple_episodes(
    n_episodes=N_EPISODES,
    verbose=True,
    compute_all_metrics=True  # ENABLE ALL METRICS
)

print(f"\n[OK] Analysis complete - {len(records)} state-action pairs analyzed")

# Create visualization
plotter = ConsequencePlotter()

# 1. Print comprehensive statistics
print("\n" + "="*70)
print("MULTI-METRIC STATISTICS")
print("="*70)
plotter.print_statistics(records, top_n=5)

# 2. Generate side-by-side heatmap comparison
print("\nGenerating metric comparison heatmaps...")
fig_heatmaps = plotter.plot_metric_comparison_heatmaps(
    records,
    save_path='results/metric_comparison_heatmaps.png',
    show=False
)
print("[OK] Saved to: results/metric_comparison_heatmaps.png")

# 3. Generate correlation plots
print("\nGenerating metric correlation analysis...")
fig_correlation = plotter.plot_metric_correlation(
    records,
    save_path='results/metric_correlation.png',
    show=False
)
print("[OK] Saved to: results/metric_correlation.png")

# 4. Analyze ranking agreement between metrics
print("\n" + "="*70)
print("METRIC RANKING COMPARISON")
print("="*70)

# Get top-10 states by each metric
kl_scores = np.array([r.consequence_score for r in records])
jsd_scores = np.array([r.jsd_score for r in records])
tv_scores = np.array([r.tv_score for r in records])
wass_scores = np.array([r.wasserstein_score for r in records])

# Filter finite scores only
finite_mask = np.isfinite(kl_scores)
finite_records = [r for i, r in enumerate(records) if finite_mask[i]]
finite_kl = kl_scores[finite_mask]
finite_jsd = jsd_scores[finite_mask]
finite_tv = tv_scores[finite_mask]
finite_wass = wass_scores[finite_mask]

if len(finite_records) > 0:
    # Get top-10 indices for each metric
    top_n = min(10, len(finite_records))

    top_kl_indices = set(np.argsort(finite_kl)[-top_n:][::-1])
    top_jsd_indices = set(np.argsort(finite_jsd)[-top_n:][::-1])
    top_tv_indices = set(np.argsort(finite_tv)[-top_n:][::-1])
    top_wass_indices = set(np.argsort(finite_wass)[-top_n:][::-1])

    # Compute agreement (Jaccard similarity)
    def jaccard_similarity(set_a, set_b):
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    print(f"\nTop-{top_n} State Agreement (Jaccard similarity):")
    print(f"  KL vs JSD:         {jaccard_similarity(top_kl_indices, top_jsd_indices):.3f}")
    print(f"  KL vs TV:          {jaccard_similarity(top_kl_indices, top_tv_indices):.3f}")
    print(f"  KL vs Wasserstein: {jaccard_similarity(top_kl_indices, top_wass_indices):.3f}")
    print(f"  JSD vs TV:         {jaccard_similarity(top_jsd_indices, top_tv_indices):.3f}")
    print(f"  JSD vs Wasserstein:{jaccard_similarity(top_jsd_indices, top_wass_indices):.3f}")
    print(f"  TV vs Wasserstein: {jaccard_similarity(top_tv_indices, top_wass_indices):.3f}")

    # States that ALL metrics agree are consequential
    all_agree = top_kl_indices & top_jsd_indices & top_tv_indices & top_wass_indices
    print(f"\nStates in top-{top_n} for ALL metrics: {len(all_agree)}/{top_n}")

    if len(all_agree) > 0:
        print("\nUniversally consequential states:")
        for idx in sorted(all_agree):
            record = finite_records[idx]
            print(f"  State {record.state} at {record.position}: "
                  f"KL={record.consequence_score:.2f}, JSD={record.jsd_score:.2f}, "
                  f"TV={record.tv_score:.2f}, W={record.wasserstein_score:.2f}")

    # States where metrics disagree significantly
    kl_only = top_kl_indices - (top_jsd_indices | top_tv_indices | top_wass_indices)
    if len(kl_only) > 0:
        print(f"\nStates consequential ONLY by KL: {len(kl_only)}")
        for idx in sorted(kl_only):
            record = finite_records[idx]
            print(f"  State {record.state} at {record.position}: "
                  f"KL={record.consequence_score:.2f}, JSD={record.jsd_score:.2f}, "
                  f"TV={record.tv_score:.2f}, W={record.wasserstein_score:.2f}")

# 5. Correlation coefficients
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

print("\nPearson correlation between metrics (finite scores only):")
print(f"  KL vs JSD:         {np.corrcoef(finite_kl, finite_jsd)[0,1]:.3f}")
print(f"  KL vs TV:          {np.corrcoef(finite_kl, finite_tv)[0,1]:.3f}")
print(f"  KL vs Wasserstein: {np.corrcoef(finite_kl, finite_wass)[0,1]:.3f}")
print(f"  JSD vs TV:         {np.corrcoef(finite_jsd, finite_tv)[0,1]:.3f}")
print(f"  JSD vs Wasserstein:{np.corrcoef(finite_jsd, finite_wass)[0,1]:.3f}")
print(f"  TV vs Wasserstein: {np.corrcoef(finite_tv, finite_wass)[0,1]:.3f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - results/metric_comparison_heatmaps.png")
print("  - results/metric_correlation.png")
print("\nKey insights:")
print("  - High correlation suggests metrics agree on consequential states")
print("  - Low correlation suggests metrics capture different aspects")
print("  - KL can be infinite; other metrics are bounded")
print("  - JSD/TV/Wasserstein handle edge cases more gracefully")

env.close()
