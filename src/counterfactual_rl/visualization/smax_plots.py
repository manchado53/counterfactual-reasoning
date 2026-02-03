"""
Visualization tools for SMAX (JaxMARL) multi-agent counterfactual analysis.

Adapted from smac_plots.py but uses SMAXConsequenceRecord instead of
SmacConsequenceRecord. Original smac_plots.py is untouched.
"""

from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from counterfactual_rl.utils.smax_data_structures import SMAXConsequenceRecord
from counterfactual_rl.utils.action_names import format_joint_action


class SMAXConsequencePlotter:
    """
    Visualization tools for SMAX counterfactual analysis.

    Adapted from SmacConsequencePlotter for JaxMARL SMAX environments.
    """

    def __init__(self, n_enemies: int):
        """Initialize SMAX plotter.

        Args:
            n_enemies: Number of enemies in the scenario (for action name translation).
        """
        self.n_enemies = n_enemies

    def plot_histogram(
        self,
        records: List[SMAXConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Distribution of Consequence Scores",
    ) -> plt.Axes:
        """
        Plot KDE of consequence scores.

        Args:
            records: List of SMAXConsequenceRecord objects.
            ax: Matplotlib axes. If None, creates new figure.
            title: Plot title.

        Returns:
            Matplotlib axes.
        """
        from scipy.stats import gaussian_kde

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        scores = np.array([r.kl_score for r in records])

        finite_scores = scores[np.isfinite(scores)]
        n_infinite = np.sum(np.isinf(scores))

        if len(finite_scores) > 1:
            kde = gaussian_kde(finite_scores, bw_method="scott")

            x_min, x_max = finite_scores.min(), finite_scores.max()
            margin = 0.1 * (x_max - x_min) if x_max > x_min else 0.5
            x_plot = np.linspace(x_min - margin, x_max + margin, 200)

            ax.fill_between(x_plot, kde(x_plot), alpha=0.3, color="steelblue")
            ax.plot(x_plot, kde(x_plot), color="steelblue", linewidth=2, label="KDE")

            finite_mean = finite_scores.mean()
            finite_median = np.median(finite_scores)

            ax.axvline(
                finite_mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {finite_mean:.3f}",
            )
            ax.axvline(
                finite_median,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {finite_median:.3f}",
            )

        if n_infinite > 0:
            ax.text(
                0.98,
                0.98,
                f"Note: {n_infinite} infinite values excluded",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9,
            )

        ax.set_xlabel("Consequence Score (max KL divergence)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_return_distributions(
        self,
        records: List[SMAXConsequenceRecord],
        top_n: int = 6,
        figsize: Optional[tuple] = None,
        title: str = "Return Distributions for Most Consequential States",
    ) -> plt.Figure:
        """
        Plot return distributions for all joint actions at the most consequential states.

        Args:
            records: List of SMAXConsequenceRecord objects.
            top_n: Number of top consequential states to visualize.
            figsize: Figure size. If None, auto-computed.
            title: Main figure title.

        Returns:
            Matplotlib figure.
        """
        scores = np.array([r.kl_score for r in records])
        scores_for_ranking = np.array([-1 if np.isinf(s) else s for s in scores])
        top_indices = np.argsort(scores_for_ranking)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        if figsize is None:
            ncols = min(3, top_n)
            nrows = int(np.ceil(top_n / ncols))
            figsize = (6 * ncols, 5 * nrows)

        ncols = min(3, top_n)
        nrows = int(np.ceil(top_n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if top_n == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if top_n > 1 else [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, record in enumerate(top_records):
            ax = axes[idx]
            actions = list(record.return_distributions.keys())

            for action_idx, action in enumerate(actions):
                returns = record.return_distributions[action]

                action_label = format_joint_action(action, self.n_enemies)
                if action == record.action:
                    action_label += " (chosen)"

                ax.hist(
                    returns,
                    bins=20,
                    alpha=0.5,
                    color=colors[action_idx % len(colors)],
                    label=action_label,
                    edgecolor="black",
                    linewidth=0.5,
                    density=False,
                )

                mean_val = returns.mean()
                ax.axvline(
                    mean_val,
                    color=colors[action_idx % len(colors)],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )

            ax.set_xlabel("Return Value")
            ax.set_ylabel("Count")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3, axis="both")

            score_display = (
                "-1 (inf)"
                if np.isinf(record.kl_score)
                else f"{record.kl_score:.2f}"
            )
            action_names = format_joint_action(record.action, self.n_enemies)
            ax.set_title(
                f"Rank #{idx + 1}: t={record.timestep}\n"
                f"{action_names}\nScore: {score_display}",
                fontsize=9,
                fontweight="bold",
            )

        for idx in range(top_n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()

        return fig

    def plot_consequence_over_time(
        self,
        records: List[SMAXConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Consequence Scores Over Time",
    ) -> plt.Axes:
        """
        Plot all consequence metrics over episode timesteps.

        Args:
            records: List of SMAXConsequenceRecord objects.
            ax: Matplotlib axes. If None, creates new figure.
            title: Plot title.

        Returns:
            Matplotlib axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        timesteps = np.array([r.timestep for r in records])
        kl_scores = np.array([r.kl_score for r in records])
        jsd_scores = np.array(
            [r.jsd_score if r.jsd_score is not None else 0.0 for r in records]
        )
        tv_scores = np.array(
            [r.tv_score if r.tv_score is not None else 0.0 for r in records]
        )
        w_scores = np.array(
            [
                r.wasserstein_score if r.wasserstein_score is not None else 0.0
                for r in records
            ]
        )

        ax.plot(
            timesteps,
            kl_scores,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="KL Divergence",
            color="tab:blue",
        )
        ax.plot(
            timesteps,
            jsd_scores,
            marker="s",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="Jensen-Shannon",
            color="tab:orange",
        )
        ax.plot(
            timesteps,
            tv_scores,
            marker="^",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="Total Variation",
            color="tab:green",
        )
        ax.plot(
            timesteps,
            w_scores,
            marker="d",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="Wasserstein",
            color="tab:red",
        )

        max_idx = np.argmax(kl_scores)
        ax.axvline(
            x=timesteps[max_idx],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Peak KL (t={timesteps[max_idx]})",
        )

        ax.set_xlabel("Episode Timestep")
        ax.set_ylabel("Consequence Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        return ax

    def plot_comprehensive(
        self,
        records: List[SMAXConsequenceRecord],
        save_path: Optional[str] = None,
        title: str = "SMAX Counterfactual Analysis",
    ) -> plt.Figure:
        """
        Create a comprehensive multi-panel figure with all visualizations.

        Panels:
            - Top-left: KDE histogram of consequence scores
            - Top-right: Consequence metrics over time
            - Bottom: Return distributions for top-3 most consequential moments

        Args:
            records: List of SMAXConsequenceRecord objects.
            save_path: If provided, save the figure to this path.
            title: Main figure title.

        Returns:
            Matplotlib figure.
        """
        fig = plt.figure(figsize=(18, 14))

        # Top row: histogram + time-series
        ax_hist = fig.add_subplot(2, 2, 1)
        ax_time = fig.add_subplot(2, 2, 2)

        self.plot_histogram(records, ax=ax_hist, title="Score Distribution")
        self.plot_consequence_over_time(
            records, ax=ax_time, title="Metrics Over Time"
        )

        # Bottom row: top-3 return distributions
        top_n = min(3, len(records))
        if top_n > 0:
            scores = np.array([r.kl_score for r in records])
            scores_for_ranking = np.array(
                [-1 if np.isinf(s) else s for s in scores]
            )
            top_indices = np.argsort(scores_for_ranking)[-top_n:][::-1]
            top_records = [records[i] for i in top_indices]

            colors = plt.cm.tab10(np.linspace(0, 1, 10))

            for panel_idx, record in enumerate(top_records):
                ax = fig.add_subplot(2, top_n, top_n + panel_idx + 1)
                actions = list(record.return_distributions.keys())

                for action_idx, action in enumerate(actions):
                    returns = record.return_distributions[action]
                    action_label = format_joint_action(action, self.n_enemies)
                    if action == record.action:
                        action_label += " (chosen)"

                    ax.hist(
                        returns,
                        bins=20,
                        alpha=0.5,
                        color=colors[action_idx % len(colors)],
                        label=action_label,
                        edgecolor="black",
                        linewidth=0.5,
                        density=False,
                    )

                    mean_val = returns.mean()
                    ax.axvline(
                        mean_val,
                        color=colors[action_idx % len(colors)],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                    )

                ax.set_xlabel("Return Value")
                ax.set_ylabel("Count")
                ax.legend(loc="best", fontsize=6)
                ax.grid(True, alpha=0.3)

                score_display = (
                    "-1 (inf)"
                    if np.isinf(record.kl_score)
                    else f"{record.kl_score:.2f}"
                )
                ax.set_title(
                    f"Rank #{panel_idx + 1}: t={record.timestep}, "
                    f"KL={score_display}",
                    fontsize=9,
                    fontweight="bold",
                )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Comprehensive plot saved to: {save_path}")

        return fig

    def print_statistics(
        self, records: List[SMAXConsequenceRecord], top_n: int = 5
    ):
        """
        Print detailed statistics about consequence scores.

        Args:
            records: List of SMAXConsequenceRecord objects.
            top_n: Number of top states to display.
        """
        print("\n" + "=" * 60)
        print("SMAX COUNTERFACTUAL ANALYSIS STATISTICS")
        print("=" * 60)

        scores = np.array([r.kl_score for r in records])
        finite_scores = scores[np.isfinite(scores)]
        n_infinite = np.sum(np.isinf(scores))

        print(f"\nTotal timesteps analyzed: {len(records)}")

        if len(finite_scores) > 0:
            print("\nConsequence Scores (KL Divergence):")
            print(f"  Mean:   {finite_scores.mean():.4f}")
            print(f"  Median: {np.median(finite_scores):.4f}")
            print(f"  Std:    {finite_scores.std():.4f}")
            print(f"  Min:    {finite_scores.min():.4f}")
            print(f"  Max:    {finite_scores.max():.4f}")

        if n_infinite > 0:
            print(
                f"  Infinite: {n_infinite}/{len(scores)} "
                f"({n_infinite / len(scores) * 100:.1f}%)"
            )

        # Additional metric summaries
        jsd_scores = [r.jsd_score for r in records if r.jsd_score is not None]
        if jsd_scores:
            jsd_arr = np.array(jsd_scores)
            print(f"\nJensen-Shannon Divergence:")
            print(f"  Mean: {jsd_arr.mean():.4f}, Max: {jsd_arr.max():.4f}")

        tv_scores = [r.tv_score for r in records if r.tv_score is not None]
        if tv_scores:
            tv_arr = np.array(tv_scores)
            finite_tv = tv_arr[np.isfinite(tv_arr)]
            if len(finite_tv) > 0:
                print(f"\nTotal Variation:")
                print(
                    f"  Mean: {finite_tv.mean():.4f}, Max: {finite_tv.max():.4f}"
                )

        w_scores = [
            r.wasserstein_score for r in records if r.wasserstein_score is not None
        ]
        if w_scores:
            w_arr = np.array(w_scores)
            print(f"\nWasserstein Distance:")
            print(f"  Mean: {w_arr.mean():.4f}, Max: {w_arr.max():.4f}")

        # Top-N most consequential moments
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        print(f"\nTop-{top_n} Most Consequential Moments:")

        for idx, record in enumerate(top_records):
            action_names = format_joint_action(record.action, self.n_enemies)
            print(f"  #{idx + 1}: Timestep {record.timestep}")
            print(f"       Action: {action_names}")
            print(f"       KL Score: {record.kl_score:.4f}")

            if record.kl_divergences:
                max_alt = max(
                    record.kl_divergences.items(), key=lambda x: x[1]
                )
                max_alt_names = format_joint_action(max_alt[0], self.n_enemies)
                print(
                    f"       Max KL vs {max_alt_names}: {max_alt[1]:.4f}"
                )

        print("=" * 60)
