"""
Visualization tools for Gardner chess counterfactual analysis.

Consumes List[ChessConsequenceRecord] and produces:
  - consequence_over_time.png   — 4 metrics across episode steps
  - consequence_histogram.png   — KDE density of scores
  - return_distributions.png    — return histograms for top-N consequential moves
  - analysis_comprehensive.png  — 2x2 grid combining all of the above
"""

from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from counterfactual_rl.utils.chess_data_structures import ChessConsequenceRecord


class ChessConsequencePlotter:
    """Visualization tools for Gardner chess counterfactual analysis."""

    def plot_histogram(
        self,
        records: List[ChessConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Distribution of Consequence Scores",
    ) -> plt.Axes:
        """KDE density of Wasserstein consequence scores across all steps."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        scores = np.array([r.wasserstein_score for r in records
                           if r.wasserstein_score is not None])
        finite = scores[np.isfinite(scores)]

        if len(finite) > 1:
            kde = gaussian_kde(finite, bw_method='scott')
            x_min, x_max = finite.min(), finite.max()
            margin = 0.1 * (x_max - x_min) if x_max > x_min else 0.5
            x_plot = np.linspace(x_min - margin, x_max + margin, 200)
            ax.fill_between(x_plot, kde(x_plot), alpha=0.3, color='steelblue')
            ax.plot(x_plot, kde(x_plot), color='steelblue', linewidth=2)
            ax.axvline(finite.mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {finite.mean():.3f}")
            ax.axvline(np.median(finite), color='green', linestyle='--', linewidth=2,
                       label=f"Median: {np.median(finite):.3f}")

        ax.set_xlabel("Wasserstein Consequence Score")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_consequence_over_time(
        self,
        records: List[ChessConsequenceRecord],
        ax: Optional[plt.Axes] = None,
        title: str = "Consequence Scores Over Episode",
    ) -> plt.Axes:
        """Line plot of all 4 metrics over white's move number."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        timesteps = np.array([r.timestep for r in records])
        kl  = np.array([r.kl_score for r in records])
        jsd = np.array([r.jsd_score if r.jsd_score is not None else 0.0 for r in records])
        tv  = np.array([r.tv_score  if r.tv_score  is not None else 0.0 for r in records])
        ws  = np.array([r.wasserstein_score if r.wasserstein_score is not None else 0.0
                        for r in records])

        ax.plot(timesteps, kl,  marker='o', linewidth=2, markersize=5,
                label='KL Divergence',    color='tab:blue')
        ax.plot(timesteps, jsd, marker='s', linewidth=2, markersize=5,
                label='Jensen-Shannon',   color='tab:orange')
        ax.plot(timesteps, tv,  marker='^', linewidth=2, markersize=5,
                label='Total Variation',  color='tab:green')
        ax.plot(timesteps, ws,  marker='d', linewidth=2, markersize=5,
                label='Wasserstein',      color='tab:red')

        if len(timesteps) > 0:
            peak = timesteps[np.argmax(ws)]
            ax.axvline(peak, color='gray', linestyle='--', alpha=0.5,
                       label=f"Peak WS (t={peak})")

        ax.set_xlabel("White's Move Number")
        ax.set_ylabel("Consequence Score")
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        return ax

    def plot_return_distributions(
        self,
        records: List[ChessConsequenceRecord],
        top_n: int = 6,
        figsize: Optional[tuple] = None,
        title: str = "Return Distributions — Most Consequential Moves",
    ) -> plt.Figure:
        """
        Return distribution histograms for the top-N most consequential moves.

        Each subplot shows overlaid histograms for all K candidate moves,
        with the chosen move highlighted.
        """
        ws_scores = np.array([
            r.wasserstein_score if r.wasserstein_score is not None else 0.0
            for r in records
        ])
        top_indices = np.argsort(ws_scores)[-top_n:][::-1]
        top_records = [records[i] for i in top_indices]

        n_plots = len(top_records)
        ncols = min(3, n_plots)
        nrows = max(1, int(np.ceil(n_plots / ncols)))
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten() if n_plots > 1 else [axes]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, record in enumerate(top_records):
            ax = axes[idx]
            for color_i, (move, returns) in enumerate(record.return_distributions.items()):
                label = f"move {move}"
                if move == record.action:
                    label += " (chosen)"
                ax.hist(returns, bins=20, alpha=0.5,
                        color=colors[color_i % len(colors)],
                        label=label, edgecolor='black', linewidth=0.5, density=False)
                ax.axvline(returns.mean(), color=colors[color_i % len(colors)],
                           linestyle='--', linewidth=2, alpha=0.8)

            ax.set_xlabel("Return")
            ax.set_ylabel("Count")
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
            ws = record.wasserstein_score or 0.0
            ax.set_title(
                f"Rank #{idx+1}  t={record.timestep}  WS={ws:.3f}",
                fontsize=9, fontweight='bold',
            )

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_comprehensive(
        self,
        records: List[ChessConsequenceRecord],
        save_path: Optional[str] = None,
        title: str = "Gardner Chess — Counterfactual Consequence Analysis",
    ) -> plt.Figure:
        """
        2×2 comprehensive figure:
          top-left:  score histogram
          top-right: scores over time
          bottom:    return distributions for top-3 most consequential moves
        """
        fig = plt.figure(figsize=(18, 13))

        ax_hist = fig.add_subplot(2, 2, 1)
        ax_time = fig.add_subplot(2, 2, 2)
        self.plot_histogram(records, ax=ax_hist, title="Score Distribution")
        self.plot_consequence_over_time(records, ax=ax_time, title="Scores Over Episode")

        top_n = min(3, len(records))
        if top_n > 0:
            ws_scores = np.array([r.wasserstein_score or 0.0 for r in records])
            top_records = [records[i] for i in np.argsort(ws_scores)[-top_n:][::-1]]
            colors = plt.cm.tab10(np.linspace(0, 1, 10))

            for panel_idx, record in enumerate(top_records):
                ax = fig.add_subplot(2, top_n, top_n + panel_idx + 1)
                for ci, (move, returns) in enumerate(record.return_distributions.items()):
                    label = f"{move}" + (" ✓" if move == record.action else "")
                    ax.hist(returns, bins=20, alpha=0.5,
                            color=colors[ci % len(colors)],
                            label=label, edgecolor='black', linewidth=0.5)
                    ax.axvline(returns.mean(), color=colors[ci % len(colors)],
                               linestyle='--', linewidth=2, alpha=0.8)
                ws = record.wasserstein_score or 0.0
                ax.set_title(f"#{panel_idx+1}  t={record.timestep}  WS={ws:.3f}",
                             fontsize=9, fontweight='bold')
                ax.set_xlabel("Return")
                ax.legend(loc='best', fontsize=6)
                ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def print_statistics(
        self,
        records: List[ChessConsequenceRecord],
        top_n: int = 5,
    ):
        """Print summary statistics and top-N most consequential moves."""
        print("\n" + "=" * 60)
        print("CHESS COUNTERFACTUAL ANALYSIS — STATISTICS")
        print("=" * 60)
        print(f"Total moves analyzed: {len(records)}")

        for metric_name, attr in [
            ("Wasserstein",     "wasserstein_score"),
            ("KL Divergence",   "kl_score"),
            ("Jensen-Shannon",  "jsd_score"),
            ("Total Variation", "tv_score"),
        ]:
            vals = np.array([getattr(r, attr) for r in records
                             if getattr(r, attr) is not None])
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                print(f"\n{metric_name}:")
                print(f"  Mean={finite.mean():.4f}  Median={np.median(finite):.4f}  "
                      f"Std={finite.std():.4f}  Max={finite.max():.4f}")

        ws_scores = np.array([r.wasserstein_score or 0.0 for r in records])
        top_records = [records[i] for i in np.argsort(ws_scores)[-top_n:][::-1]]
        print(f"\nTop-{top_n} most consequential moves (by Wasserstein):")
        for rank, record in enumerate(top_records, 1):
            print(f"  #{rank}: move={record.action}  t={record.timestep}  "
                  f"WS={record.wasserstein_score:.4f}  KL={record.kl_score:.4f}")
        print("=" * 60)
