"""
Visualize Consequential Timesteps in SMAC

This script demonstrates how to visualize which timesteps were most consequential
during SMAC gameplay. It creates several plots:
1. Consequence scores over time
2. Heatmap of critical moments
3. Action distribution at critical timesteps
4. Comparison of actual vs alternative actions

Note: Requires CounterfactualAnalyzer to support MultiDiscrete actions.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smac.env import StarCraft2Env

from counterfactual_rl.environments.smac import CentralizedSmacWrapper, SmacStateManager
from counterfactual_rl.policies import RandomPolicy
# from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer  # TODO: Update for MultiDiscrete

# Set SC2 Path
os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_consequence_timeline(records, save_path=None):
    """
    Plot consequence scores over time.
    
    Args:
        records: List of ConsequenceRecord objects from analyzer
        save_path: Optional path to save the plot
    """
    timesteps = list(range(len(records)))
    consequence_scores = [r.consequence_score for r in records]
    
    plt.figure(figsize=(14, 6))
    
    # Plot consequence scores
    plt.plot(timesteps, consequence_scores, 'b-', linewidth=2, label='Consequence Score')
    
    # Highlight critical timesteps (top 20%)
    threshold = np.percentile(consequence_scores, 80)
    critical_mask = np.array(consequence_scores) > threshold
    critical_timesteps = np.array(timesteps)[critical_mask]
    critical_scores = np.array(consequence_scores)[critical_mask]
    
    plt.scatter(critical_timesteps, critical_scores, 
                color='red', s=100, zorder=5, 
                label=f'Critical Moments (>{threshold:.2f})')
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'80th Percentile')
    
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Consequence Score', fontsize=12)
    plt.title('Consequential Timesteps in SMAC Episode', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timeline plot to {save_path}")
    
    plt.show()


def plot_consequence_heatmap(records, save_path=None):
    """
    Create a heatmap showing consequence scores and actions.
    
    Args:
        records: List of ConsequenceRecord objects
        save_path: Optional path to save the plot
    """
    n_timesteps = len(records)
    n_agents = len(records[0].action) if hasattr(records[0].action, '__len__') else 1
    
    # Create matrices for visualization
    consequence_matrix = np.zeros((n_agents + 1, n_timesteps))
    action_matrix = np.zeros((n_agents, n_timesteps))
    
    for t, record in enumerate(records):
        consequence_matrix[0, t] = record.consequence_score
        
        if hasattr(record.action, '__len__'):
            for agent_id, action in enumerate(record.action):
                action_matrix[agent_id, t] = action
        else:
            action_matrix[0, t] = record.action
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
                                     gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot 1: Consequence scores
    sns.heatmap(consequence_matrix[:1, :], 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Consequence Score'},
                ax=ax1,
                yticklabels=['Overall'])
    ax1.set_xlabel('')
    ax1.set_title('Consequence Scores Over Time', fontsize=14, fontweight='bold')
    
    # Plot 2: Agent actions
    sns.heatmap(action_matrix, 
                cmap='viridis', 
                cbar_kws={'label': 'Action ID'},
                ax=ax2,
                yticklabels=[f'Agent {i}' for i in range(n_agents)])
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_title('Agent Actions Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()


def plot_critical_moments_analysis(records, top_k=5, save_path=None):
    """
    Analyze and visualize the top-k most critical moments.
    
    Args:
        records: List of ConsequenceRecord objects
        top_k: Number of top critical moments to analyze
        save_path: Optional path to save the plot
    """
    # Sort by consequence score
    sorted_records = sorted(enumerate(records), 
                           key=lambda x: x[1].consequence_score, 
                           reverse=True)
    
    top_moments = sorted_records[:top_k]
    
    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 5))
    if top_k == 1:
        axes = [axes]
    
    for idx, (timestep, record) in enumerate(top_moments):
        ax = axes[idx]
        
        # Get return distributions
        return_dists = record.return_distributions
        
        # Plot return distributions for actual and alternative actions
        actions = list(return_dists.keys())
        actual_action = tuple(record.action) if hasattr(record.action, '__len__') else record.action
        
        # Create violin plot of returns
        data_to_plot = []
        labels = []
        colors = []
        
        for action in actions[:10]:  # Limit to 10 actions for visibility
            returns = return_dists[action]
            data_to_plot.append(returns)
            
            if action == actual_action:
                labels.append(f'Actual\n{action}')
                colors.append('red')
            else:
                labels.append(f'{action}')
                colors.append('blue')
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                              showmeans=True, showmedians=True)
        
        # Color the actual action differently
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Return')
        ax.set_title(f'Timestep {timestep}\nScore: {record.consequence_score:.3f}',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Top {top_k} Most Consequential Moments', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved critical moments analysis to {save_path}")
    
    plt.show()


def plot_action_distribution(records, save_path=None):
    """
    Plot distribution of actions taken at different consequence levels.
    
    Args:
        records: List of ConsequenceRecord objects
        save_path: Optional path to save the plot
    """
    # Categorize timesteps by consequence score
    low_consequence = []
    medium_consequence = []
    high_consequence = []
    
    threshold_low = np.percentile([r.consequence_score for r in records], 33)
    threshold_high = np.percentile([r.consequence_score for r in records], 67)
    
    for record in records:
        if hasattr(record.action, '__len__'):
            actions = record.action
        else:
            actions = [record.action]
        
        if record.consequence_score < threshold_low:
            low_consequence.extend(actions)
        elif record.consequence_score < threshold_high:
            medium_consequence.extend(actions)
        else:
            high_consequence.extend(actions)
    
    # Create histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, data, title, color in zip(axes, 
                                       [low_consequence, medium_consequence, high_consequence],
                                       ['Low Consequence', 'Medium Consequence', 'High Consequence'],
                                       ['green', 'orange', 'red']):
        ax.hist(data, bins=range(10), alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel('Action ID')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Action Distribution by Consequence Level', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved action distribution to {save_path}")
    
    plt.show()


def create_summary_report(records, save_path=None):
    """
    Create a text summary report of the analysis.
    
    Args:
        records: List of ConsequenceRecord objects
        save_path: Optional path to save the report
    """
    report = []
    report.append("="*60)
    report.append("SMAC Counterfactual Analysis Summary")
    report.append("="*60)
    report.append("")
    
    # Overall statistics
    consequence_scores = [r.consequence_score for r in records]
    report.append(f"Total Timesteps: {len(records)}")
    report.append(f"Average Consequence Score: {np.mean(consequence_scores):.4f}")
    report.append(f"Max Consequence Score: {np.max(consequence_scores):.4f}")
    report.append(f"Min Consequence Score: {np.min(consequence_scores):.4f}")
    report.append(f"Std Dev: {np.std(consequence_scores):.4f}")
    report.append("")
    
    # Critical moments
    threshold = np.percentile(consequence_scores, 80)
    critical_count = sum(1 for s in consequence_scores if s > threshold)
    report.append(f"Critical Timesteps (>80th percentile): {critical_count}")
    report.append(f"Critical Threshold: {threshold:.4f}")
    report.append("")
    
    # Top 5 most consequential moments
    sorted_records = sorted(enumerate(records), 
                           key=lambda x: x[1].consequence_score, 
                           reverse=True)
    
    report.append("Top 5 Most Consequential Moments:")
    report.append("-" * 60)
    for rank, (timestep, record) in enumerate(sorted_records[:5], 1):
        report.append(f"{rank}. Timestep {timestep}:")
        report.append(f"   Consequence Score: {record.consequence_score:.4f}")
        report.append(f"   Action Taken: {record.action}")
        report.append("")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved report to {save_path}")
    
    return report_text


def visualize_all(records, output_dir='results'):
    """
    Create all visualizations and save to output directory.
    
    Args:
        records: List of ConsequenceRecord objects
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    print("="*60)
    
    # 1. Timeline plot
    print("\n1. Creating consequence timeline...")
    plot_consequence_timeline(records, 
                             save_path=os.path.join(output_dir, 'timeline.png'))
    
    # 2. Heatmap
    print("\n2. Creating consequence heatmap...")
    plot_consequence_heatmap(records,
                            save_path=os.path.join(output_dir, 'heatmap.png'))
    
    # 3. Critical moments analysis
    print("\n3. Analyzing critical moments...")
    plot_critical_moments_analysis(records, top_k=5,
                                  save_path=os.path.join(output_dir, 'critical_moments.png'))
    
    # 4. Action distribution
    print("\n4. Plotting action distribution...")
    plot_action_distribution(records,
                            save_path=os.path.join(output_dir, 'action_distribution.png'))
    
    # 5. Summary report
    print("\n5. Creating summary report...")
    create_summary_report(records,
                         save_path=os.path.join(output_dir, 'summary_report.txt'))
    
    print("\n" + "="*60)
    print(f"All visualizations saved to {output_dir}/")
    print("="*60)


# Example usage (once CounterfactualAnalyzer is updated):
def example_usage():
    """
    Example of how to use the visualization functions.
    
    NOTE: This requires CounterfactualAnalyzer to support MultiDiscrete actions.
    """
    print("\n" + "="*60)
    print("Example: Visualizing Consequential Timesteps")
    print("="*60)
    print("\nOnce CounterfactualAnalyzer is updated, use it like this:")
    print("""
from counterfactual_rl.analysis.counterfactual import CounterfactualAnalyzer

# Setup
smac_env = StarCraft2Env(map_name="3m")
wrapped_env = CentralizedSmacWrapper(smac_env, use_state=True)
policy = RandomPolicy(smac_env)
state_manager = SmacStateManager()

# Create analyzer
analyzer = CounterfactualAnalyzer(
    model=policy,
    env=wrapped_env,
    state_manager=state_manager,
    horizon=20,
    n_rollouts=48,
    n_counterfactual_samples=10
)

# Run analysis
records = analyzer.evaluate_episode(max_steps=50, verbose=True)

# Visualize results
visualize_all(records, output_dir='smac_analysis_results')
    """)


if __name__ == "__main__":
    example_usage()
