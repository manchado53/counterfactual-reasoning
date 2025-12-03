"""
Visualization tools for consequential states analysis
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from counterfactual_rl.utils.data_structures import ConsequenceRecord


class ConsequencePlotter:
    """
    Creates visualizations for consequential states analysis.

    Generates:
    1. Heatmap of average consequence scores per grid cell
    2. Histogram of all consequence scores
    3. Top-N consequential states visualization
    
    Supports both grid-based environments (FrozenLake, Taxi) and others.
    """

    def __init__(
        self,
        grid_shape: Optional[Tuple[int, int]] = None,
        map_labels: Optional[List[List[str]]] = None
    ):
        """
        Initialize plotter.

        Args:
            grid_shape: (rows, cols) for grid environments. If None, defaults to (4, 4) for FrozenLake.
            map_labels: Optional custom map labels. If None, uses FrozenLake default.
        """
        if grid_shape is None:
            grid_shape = (4, 4)  # Default to FrozenLake
        
        self.grid_rows, self.grid_cols = grid_shape
        self.grid_size = self.grid_rows * self.grid_cols  # Total number of cells

        # Default FrozenLake 4x4 map
        if map_labels is None and grid_shape == (4, 4):
            self.map_labels = [
                ['S', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'H'],
                ['F', 'F', 'F', 'H'],
                ['H', 'F', 'F', 'G']
            ]
        elif map_labels is None:
            # Generate generic labels for non-standard grids
            self.map_labels = [[' ' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        else:
            self.map_labels = map_labels

    def plot_heatmap(
        self,
        records: List[ConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Average Consequence Score per Grid Cell"
    ) -> plt.Axes:
        """
        Plot heatmap of average consequence scores.

        Args:
            records: List of ConsequenceRecord objects
            ax: Matplotlib axes. If None, creates new figure.
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # Define placeholder for infinite values (visualization only)
        INF_PLACEHOLDER = -1

        # Aggregate scores by position
        # Use a dictionary to collect all scores per cell
        cell_scores = {}
        for record in records:
            row, col = record.position
            key = (row, col)
            if key not in cell_scores:
                cell_scores[key] = []
            # Replace infinity with placeholder for visualization
            viz_score = INF_PLACEHOLDER if np.isinf(record.consequence_score) else record.consequence_score
            cell_scores[key].append(viz_score)

        # Compute average for each cell
        avg_score_grid = np.zeros((self.grid_rows, self.grid_cols))
        for (row, col), scores_list in cell_scores.items():
            if len(scores_list) > 0:
                avg_score_grid[row, col] = np.mean(scores_list)

        # Create heatmap
        sns.heatmap(
            avg_score_grid,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Avg Consequence Score (∞ shown as -1)'},
            ax=ax,
            vmin=-1
        )

        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Add map labels
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                ax.text(
                    j + 0.5, i + 0.85,
                    self.map_labels[i][j],
                    ha='center', va='center',
                    color='blue', fontsize=10, fontweight='bold'
                )

        return ax

    def plot_histogram(
        self,
        records: List[ConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Distribution of Consequence Scores",
        bins: int = 30
    ) -> plt.Axes:
        """
        Plot histogram of consequence scores.

        Args:
            records: List of ConsequenceRecord objects
            ax: Matplotlib axes. If None, creates new figure.
            title: Plot title
            bins: Number of histogram bins

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        scores = np.array([r.consequence_score for r in records])

        # Separate finite and infinite scores
        finite_scores = scores[np.isfinite(scores)]
        n_infinite = np.sum(np.isinf(scores))

        # Plot histogram of finite scores only
        if len(finite_scores) > 0:
            ax.hist(finite_scores, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')

            # Add mean and median lines for finite scores
            finite_mean = finite_scores.mean()
            finite_median = np.median(finite_scores)

            ax.axvline(finite_mean, color='red', linestyle='--',
                       label=f'Mean: {finite_mean:.3f}')
            ax.axvline(finite_median, color='green', linestyle='--',
                       label=f'Median: {finite_median:.3f}')

        # Add note about infinite values if present
        if n_infinite > 0:
            ax.text(0.98, 0.98, f'Note: {n_infinite} infinite values\nexcluded from histogram',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)

        ax.set_xlabel('Consequence Score (max KL divergence)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_top_states(
        self,
        records: List[ConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        top_n: int = 5,
        title: str = "Top-5 Most Consequential States"
    ) -> plt.Axes:
        """
        Plot visualization highlighting top-N consequential states.

        Args:
            records: List of ConsequenceRecord objects
            ax: Matplotlib axes. If None, creates new figure.
            top_n: Number of top states to highlight
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # Define placeholder for infinite values (visualization only)
        INF_PLACEHOLDER = -1

        scores = np.array([r.consequence_score for r in records])
        # Replace infinities with placeholder for ranking purposes
        scores_for_ranking = np.array([INF_PLACEHOLDER if np.isinf(s) else s for s in scores])

        # Find top-N most consequential records
        top_indices = np.argsort(scores_for_ranking)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        # Create grid for visualization
        viz_grid = np.zeros((self.grid_rows, self.grid_cols))

        for idx, record in enumerate(top_records):
            row, col = record.position
            viz_grid[row, col] = top_n - idx  # Rank: 1st=5, 2nd=4, etc.

        # Plot
        im = ax.imshow(viz_grid, cmap='Reds', vmin=0, vmax=top_n)

        # Add grid lines
        for i in range(self.grid_rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Annotate with ranks and scores
        for idx, record in enumerate(top_records):
            row, col = record.position
            rank = idx + 1
            score = record.consequence_score
            # Display placeholder if infinite
            score_display = "-1 (∞)" if np.isinf(score) else f'{score:.3f}'

            ax.text(col, row - 0.15, f'#{rank}',
                    ha='center', va='center',
                    color='darkred', fontsize=12, fontweight='bold')
            ax.text(col, row + 0.15, score_display,
                    ha='center', va='center',
                    color='black', fontsize=9)

            # Add map label
            ax.text(col, row + 0.35, self.map_labels[row][col],
                    ha='center', va='center',
                    color='blue', fontsize=10, fontweight='bold')

        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(self.grid_cols))
        ax.set_yticks(range(self.grid_rows))

        return ax

    def plot_return_distributions(
        self,
        records: List[ConsequenceRecord],
        top_n: int = 6,
        figsize: Optional[tuple] = None,
        title: str = "Return Distributions for Most Consequential States"
    ) -> plt.Figure:
        """
        Plot return distributions for all 4 actions at the most consequential states.

        Creates overlaid histogram plots showing the distribution of returns for each
        possible action at consequential states. This visualization reveals why certain
        states are consequential by showing how different actions lead to different
        outcome distributions.

        The plots show:
        - X-axis: Return values
        - Y-axis: Count/frequency
        - Overlaid histograms with transparency for each action (color-coded)
        - Dashed vertical lines showing mean returns for each action

        Args:
            records: List of ConsequenceRecord objects
            top_n: Number of top consequential states to visualize
            figsize: Figure size (width, height). If None, auto-computed based on top_n.
            title: Main figure title

        Returns:
            Matplotlib figure
        """
        # Get top-N most consequential records
        scores = np.array([r.consequence_score for r in records])
        # Replace infinities with placeholder for ranking
        scores_for_ranking = np.array([-1 if np.isinf(s) else s for s in scores])
        top_indices = np.argsort(scores_for_ranking)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        # Auto-compute figure size if not provided
        if figsize is None:
            ncols = min(3, top_n)
            nrows = int(np.ceil(top_n / ncols))
            figsize = (6 * ncols, 5 * nrows)

        # Create figure and subplots
        ncols = min(3, top_n)
        nrows = int(np.ceil(top_n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes for easier indexing
        if top_n == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if top_n > 1 else [axes]

        # Action names and colors
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        action_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']  # blue, green, red, orange

        # Plot each state
        for idx, record in enumerate(top_records):
            ax = axes[idx]

            # Plot overlaid histograms for each action
            for action in range(4):
                if action in record.return_distributions:
                    returns = record.return_distributions[action]

                    # Plot histogram with transparency for overlay
                    ax.hist(
                        returns,
                        bins=20,
                        alpha=0.5,
                        color=action_colors[action],
                        label=action_names[action],
                        edgecolor='black',
                        linewidth=0.5,
                        density=False  # Show counts, not density
                    )

                    # Add mean line for this action
                    mean_val = returns.mean()
                    ax.axvline(
                        mean_val,
                        color=action_colors[action],
                        linestyle='--',
                        linewidth=2,
                        alpha=0.8
                    )

            # Formatting
            ax.set_xlabel('Return Value')
            ax.set_ylabel('Count')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, axis='both')

            # Title with state info
            row, col = record.position
            map_label = self.map_labels[row][col]
            actual_action = action_names[record.action]
            # Display placeholder if infinite
            score_display = "-1 (∞)" if np.isinf(record.consequence_score) else f'{record.consequence_score:.2f}'
            ax.set_title(
                f'Rank #{idx+1}: ({row},{col}) [{map_label}]\n'
                f'Action: {actual_action}, Score: {score_display}',
                fontsize=10,
                fontweight='bold'
            )

        # Hide unused subplots
        for idx in range(top_n, len(axes)):
            axes[idx].set_visible(False)

        # Main title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig

    def plot_comprehensive(
        self,
        records: List[ConsequenceRecord],
        slippery: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive visualization with all plots.

        Args:
            records: List of ConsequenceRecord objects
            slippery: Whether environment was slippery (for title)
            save_path: Optional path to save figure
            show: Whether to display the figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 5))

        # Create subplots
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        # Plot heatmap (uses -1 placeholder for infinity)
        self.plot_heatmap(
            records, ax=ax1,
            title=f'Average Consequence Score per Grid Cell\n(Slippery={slippery}, ∞ shown as -1)'
        )

        # Plot histogram
        self.plot_histogram(records, ax=ax2)

        # Plot top states (uses -1 placeholder for infinity)
        self.plot_top_states(records, ax=ax3,
                            title=f'Top-5 Most Consequential States\n(Slippery={slippery}, ∞ shown as -1)')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def print_statistics(
        self,
        records: List[ConsequenceRecord],
        top_n: int = 5
    ):
        """
        Print detailed statistics about consequence scores and all metrics if available.

        Args:
            records: List of ConsequenceRecord objects
            top_n: Number of top states to display
        """
        # Check if multiple metrics are available
        has_all_metrics = (len(records) > 0 and
                          records[0].jsd_score is not None and
                          records[0].tv_score is not None and
                          records[0].wasserstein_score is not None)

        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)

        # Helper function to print metric statistics
        def print_metric_stats(metric_name, scores_array, indent="  "):
            finite_scores = scores_array[np.isfinite(scores_array)]
            n_infinite = np.sum(np.isinf(scores_array))

            if len(finite_scores) > 0:
                print(f"{indent}{metric_name}:")
                print(f"{indent}  Mean:   {finite_scores.mean():.4f}")
                print(f"{indent}  Median: {np.median(finite_scores):.4f}")
                print(f"{indent}  Std:    {finite_scores.std():.4f}")
                print(f"{indent}  Min:    {finite_scores.min():.4f}")
                print(f"{indent}  Max:    {finite_scores.max():.4f}")
            if n_infinite > 0:
                print(f"{indent}  Infinite: {n_infinite}/{len(scores_array)} ({n_infinite/len(scores_array)*100:.1f}%)")

        # Print statistics for all available metrics
        if has_all_metrics:
            print("\n[All Metrics Computed]")
            print("\nKL Divergence (primary metric):")
            kl_scores = np.array([r.consequence_score for r in records])
            print_metric_stats("", kl_scores, indent="  ")

            print("\nJensen-Shannon Divergence:")
            jsd_scores = np.array([r.jsd_score for r in records])
            print_metric_stats("", jsd_scores, indent="  ")

            print("\nTotal Variation Distance:")
            tv_scores = np.array([r.tv_score for r in records])
            print_metric_stats("", tv_scores, indent="  ")

            print("\nWasserstein Distance:")
            wasserstein_scores = np.array([r.wasserstein_score for r in records])
            print_metric_stats("", wasserstein_scores, indent="  ")
        else:
            print("\n[KL Divergence Only]")
            kl_scores = np.array([r.consequence_score for r in records])
            print_metric_stats("Consequence Score", kl_scores)

        # Find top-N
        scores = np.array([r.consequence_score for r in records])
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        print(f"\nTop-{top_n} Most Consequential State-Action Pairs (by KL):")
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']

        for idx, record in enumerate(top_records):
            print(f"  #{idx+1}: Position {record.position}, "
                  f"Action: {action_names[record.action]}")

            if has_all_metrics:
                print(f"       KL: {record.consequence_score:.4f}, "
                      f"JSD: {record.jsd_score:.4f}, "
                      f"TV: {record.tv_score:.4f}, "
                      f"W: {record.wasserstein_score:.4f}")
            else:
                print(f"       Score: {record.consequence_score:.4f}")

            # Show which alternative action had highest KL
            if record.kl_divergences:
                max_alt = max(record.kl_divergences.items(), key=lambda x: x[1])
                print(f"       Max KL vs {action_names[max_alt[0]]}: {max_alt[1]:.4f}")

        # Grid cell analysis (excluding infinite values like heatmap)
        cell_scores_grid = {}
        for record in records:
            row, col = record.position
            key = (row, col)
            if key not in cell_scores_grid:
                cell_scores_grid[key] = []
            if np.isfinite(record.consequence_score):
                cell_scores_grid[key].append(record.consequence_score)

        # Compute averages
        avg_score_grid = np.zeros((self.grid_rows, self.grid_cols))
        count_grid = np.zeros((self.grid_rows, self.grid_cols))
        for (row, col), scores_list in cell_scores_grid.items():
            count_grid[row, col] = len(scores_list)
            if len(scores_list) > 0:
                avg_score_grid[row, col] = np.mean(scores_list)

        print("\nGrid Cell Analysis (avg finite score):")
        for i in range(self.grid_rows):
            row_str = "  "
            for j in range(self.grid_cols):
                score = avg_score_grid[i, j]
                visits = int(count_grid[i, j])
                if visits > 0:
                    row_str += f"{self.map_labels[i][j]}:{score:.3f}({visits:2d})  "
                else:
                    row_str += f"{self.map_labels[i][j]}:---  ({0:2d})  "
            print(row_str)
        print("  (Format: Cell:AvgScore(FiniteVisits), --- means all scores were infinite)")

    def plot_metric_comparison_heatmaps(
        self,
        records: List[ConsequenceRecord],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot side-by-side heatmaps comparing all metrics.

        Creates a 2x2 grid of heatmaps showing average consequence scores
        for each metric (KL, JSD, TV, Wasserstein) across grid positions.
        
        Infinite values are replaced with a placeholder (100.0) for visualization only,
        while actual records maintain their true infinity values.

        Args:
            records: List of ConsequenceRecord objects (with all metrics computed)
            save_path: Optional path to save figure
            show: Whether to display the figure

        Returns:
            Matplotlib figure
        """
        # Check if all metrics are available
        if len(records) == 0 or records[0].jsd_score is None:
            print("Warning: All metrics must be computed. Use compute_all_metrics=True.")
            return None

        # Define placeholder for infinite values (visualization only)
        INF_PLACEHOLDER = -1
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Metric Comparison: Consequence Scores by Position\n' + 
                     f'(∞ displayed as {INF_PLACEHOLDER} for visualization)',
                     fontsize=16, fontweight='bold')

        # Define metrics to plot
        metrics = [
            ('KL Divergence', 'consequence_score', 'YlOrRd'),
            ('Jensen-Shannon Divergence', 'jsd_score', 'YlGnBu'),
            ('Total Variation Distance', 'tv_score', 'PuRd'),
            ('Wasserstein Distance', 'wasserstein_score', 'YlOrBr')
        ]

        for idx, (metric_name, attr_name, cmap) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Aggregate scores by position
            cell_scores = {}
            for record in records:
                row, col = record.position
                key = (row, col)
                if key not in cell_scores:
                    cell_scores[key] = []

                score = getattr(record, attr_name)
                # Replace infinity with placeholder ONLY for visualization
                viz_score = INF_PLACEHOLDER if np.isinf(score) else score
                cell_scores[key].append(viz_score)

            # Compute average for each cell
            avg_score_grid = np.zeros((self.grid_rows, self.grid_cols))
            for (row, col), scores_list in cell_scores.items():
                if len(scores_list) > 0:
                    avg_score_grid[row, col] = np.mean(scores_list)

            # Create heatmap
            sns.heatmap(
                avg_score_grid,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                cbar_kws={'label': f'Avg {metric_name}'},
                ax=ax,
                vmin=0
            )

            ax.set_title(f'{metric_name}', fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

            # Add map labels
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    ax.text(
                        j + 0.5, i + 0.85,
                        self.map_labels[i][j],
                        ha='center', va='center',
                        color='blue', fontsize=10, fontweight='bold'
                    )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metric comparison heatmaps to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig

    def plot_metric_correlation(
        self,
        records: List[ConsequenceRecord],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot scatter plots showing correlation between different metrics.

        Creates a grid of scatter plots comparing each pair of metrics
        to analyze whether they agree on consequentiality rankings.

        Args:
            records: List of ConsequenceRecord objects (with all metrics computed)
            save_path: Optional path to save figure
            show: Whether to display the figure

        Returns:
            Matplotlib figure
        """
        # Check if all metrics are available
        if len(records) == 0 or records[0].jsd_score is None:
            print("Warning: All metrics must be computed. Use compute_all_metrics=True.")
            return None

        # Extract finite scores for all metrics
        kl_scores = np.array([r.consequence_score for r in records if np.isfinite(r.consequence_score)])
        jsd_scores = np.array([r.jsd_score for r in records if np.isfinite(r.consequence_score)])
        tv_scores = np.array([r.tv_score for r in records if np.isfinite(r.consequence_score)])
        wass_scores = np.array([r.wasserstein_score for r in records if np.isfinite(r.consequence_score)])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Metric Correlation Analysis', fontsize=16, fontweight='bold')

        # Define pairs to plot
        pairs = [
            ('KL', kl_scores, 'JSD', jsd_scores),
            ('KL', kl_scores, 'TV', tv_scores),
            ('KL', kl_scores, 'Wasserstein', wass_scores),
            ('JSD', jsd_scores, 'TV', tv_scores),
            ('JSD', jsd_scores, 'Wasserstein', wass_scores),
            ('TV', tv_scores, 'Wasserstein', wass_scores)
        ]

        for idx, (name_x, scores_x, name_y, scores_y) in enumerate(pairs):
            ax = axes[idx // 3, idx % 3]

            # Scatter plot
            ax.scatter(scores_x, scores_y, alpha=0.5, s=50, edgecolors='black', linewidths=0.5)

            # Compute correlation
            correlation = np.corrcoef(scores_x, scores_y)[0, 1]

            # Add regression line
            z = np.polyfit(scores_x, scores_y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(scores_x.min(), scores_x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(f'{name_x} Score')
            ax.set_ylabel(f'{name_y} Score')
            ax.set_title(f'{name_x} vs {name_y}\nCorrelation: {correlation:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metric correlation plots to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig
