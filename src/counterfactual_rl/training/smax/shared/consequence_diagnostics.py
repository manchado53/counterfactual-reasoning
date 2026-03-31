"""
Consequence scoring diagnostics for observability into the scoring process.

Logs per-scoring-pass statistics to a JSONL file and periodically generates
a 4-panel diagnostic plot showing metric evolution, return spreads, score
distributions, and buffer health.
"""

import json
import os
import numpy as np
from typing import List, Optional

from counterfactual_rl.analysis.metrics import compute_all_consequence_metrics


class ConsequenceDiagnostics:
    """Lightweight diagnostics for consequence scoring passes."""

    def __init__(self, run_dir: str, metric_name: str, plot_interval: int = 50,
                 n_step_slices: int = 10):
        self.run_dir = run_dir
        self.metric_name = metric_name
        self.plot_interval = plot_interval
        self.n_step_slices = n_step_slices

        self.jsonl_path = os.path.join(run_dir, 'consequence_diagnostics.jsonl')
        self.plot_path = os.path.join(run_dir, 'consequence_diagnostics.png')
        self._file = open(self.jsonl_path, 'a')

        # In-memory history for plotting
        self._history: List[dict] = []
        self._pass_count = 0

        # Accumulated (episode_step, score, q_update) for step-vs-score plot
        self._step_score_data: List[tuple] = []

    def log_scoring_pass(
        self,
        q_update_count: int,
        scores: np.ndarray,
        returns_np: np.ndarray,
        all_actual_actions: list,
        all_actions: list,
        all_action_probs: list,
        buffer,
        episode_steps: np.ndarray = None,
    ):
        """
        Log statistics from one consequence scoring pass.

        Args:
            q_update_count: Current Q-update count
            scores: (B,) configured metric scores per transition
            returns_np: (B, K, N) rollout returns
            all_actual_actions: list of B actual action tuples
            all_actions: list of B lists of K action tuples
            all_action_probs: list of B dicts {action_tuple: prob}
            buffer: ConsequenceReplayBuffer instance
        """
        B = len(scores)

        # --- All 4 metrics per transition (observability only) ---
        metric_names = ['kl_divergence', 'jensen_shannon', 'total_variation', 'wasserstein']
        metric_means = {k: 0.0 for k in metric_names}
        metric_counts = {k: B for k in metric_names}
        per_transition_metrics = {k: np.zeros(B) for k in metric_names}
        for i in range(B):
            return_distributions = {}
            for j, action_tuple in enumerate(all_actions[i]):
                if action_tuple not in return_distributions:
                    return_distributions[action_tuple] = returns_np[i, j]

            all_metrics = compute_all_consequence_metrics(
                all_actual_actions[i],
                return_distributions,
                action_probs=all_action_probs[i],
            )
            for key in metric_names:
                val = all_metrics[key][0]
                per_transition_metrics[key][i] = val
                if np.isfinite(val):
                    metric_means[key] += val
                else:
                    metric_counts[key] -= 1

        for key in metric_names:
            denom = max(metric_counts[key], 1)
            metric_means[key] /= denom

        # --- Score stats (configured metric) ---
        score_stats = {
            'score_mean': float(np.mean(scores)),
            'score_median': float(np.median(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
            'score_p75': float(np.percentile(scores, 75)),
            'score_p90': float(np.percentile(scores, 90)),
            'score_p95': float(np.percentile(scores, 95)),
        }

        # --- Return distribution stats ---
        # returns_np shape: (B, K, N) -> flatten K*N per transition
        per_transition_std = np.std(returns_np.reshape(B, -1), axis=1)
        per_transition_range = (
            np.max(returns_np.reshape(B, -1), axis=1)
            - np.min(returns_np.reshape(B, -1), axis=1)
        )
        return_stats = {
            'return_spread_mean': float(np.mean(per_transition_std)),
            'return_range_mean': float(np.mean(per_transition_range)),
        }

        # --- Buffer-level stats ---
        buf_cscores = np.array(buffer.consequence_scores[:len(buffer)])
        buf_td = np.array(buffer.td_magnitudes[:len(buffer)])
        default_cscore = buffer.max_priority  # initial value assigned on add
        scored_mask = buf_cscores != default_cscore
        buffer_stats = {
            'buffer_scored_frac': float(np.mean(scored_mask)),
            'buffer_cscore_mean': float(np.mean(buf_cscores)),
            'buffer_cscore_std': float(np.std(buf_cscores)),
            'buffer_size': len(buffer),
        }

        # --- Accumulate step-vs-score data (all 4 metrics) ---
        if episode_steps is not None:
            for i, step in enumerate(episode_steps):
                self._step_score_data.append((
                    int(step), q_update_count,
                    float(per_transition_metrics['kl_divergence'][i]),
                    float(per_transition_metrics['jensen_shannon'][i]),
                    float(per_transition_metrics['total_variation'][i]),
                    float(per_transition_metrics['wasserstein'][i]),
                ))

        # --- Assemble record ---
        record = {
            'q_update': q_update_count,
            **{f'metric_{k}': v for k, v in metric_means.items()},
            **score_stats,
            **return_stats,
            **buffer_stats,
        }

        # Write JSONL
        self._file.write(json.dumps(record) + '\n')
        self._file.flush()

        # Store for plotting
        self._history.append(record)
        self._pass_count += 1

        if self._pass_count % self.plot_interval == 0:
            self.plot()

    def plot(self):
        """Generate diagnostic plots."""
        if not self._history:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self._plot_step_scores(plt)
        self._plot_diagnostics(plt)

    def _plot_step_scores(self, plt):
        """Score vs step-in-episode: n_step_slices rows (training phases) x 4 cols (metrics).

        Each cell shows mean +/- std aggregated by integer episode step.
        Steps with fewer than 3 samples are omitted.
        """
        if not self._step_score_data:
            return

        n_slices = self.n_step_slices
        metric_labels = ['KL Divergence', 'Jensen-Shannon', 'Total Variation', 'Wasserstein']
        metric_colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        min_samples = 3

        # Columns: step, q_update, kl, jsd, tv, wasserstein
        data = np.array(self._step_score_data)
        steps = data[:, 0].astype(int)
        q_updates = data[:, 1]
        metric_scores = [data[:, 2], data[:, 3], data[:, 4], data[:, 5]]
        q_min, q_max = q_updates.min(), q_updates.max()
        max_step = int(steps.max())

        fig, axes = plt.subplots(n_slices, 4, figsize=(20, 2.5 * n_slices),
                                 sharex=True, sharey='col')
        if n_slices == 1:
            axes = axes[np.newaxis, :]

        def _plot_aggregated(ax, step_vals, score_vals, color):
            """Plot mean line + std band for score_vals grouped by integer step."""
            finite = np.isfinite(score_vals)
            step_vals = step_vals[finite]
            score_vals = score_vals[finite]
            if len(step_vals) == 0:
                return
            unique_steps = np.arange(0, max_step + 1)
            means = np.full(len(unique_steps), np.nan)
            stds = np.full(len(unique_steps), np.nan)
            for idx, s in enumerate(unique_steps):
                mask_s = step_vals == s
                if mask_s.sum() >= min_samples:
                    means[idx] = np.mean(score_vals[mask_s])
                    stds[idx] = np.std(score_vals[mask_s])
            valid = np.isfinite(means)
            if not valid.any():
                return
            x = unique_steps[valid]
            m = means[valid]
            sd = stds[valid]
            ax.plot(x, m, color=color, linewidth=1.2)
            ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.2)

        if q_max > q_min:
            boundaries = np.linspace(q_min, q_max, n_slices + 1)
            for s in range(n_slices):
                if s == n_slices - 1:
                    mask = (q_updates >= boundaries[s]) & (q_updates <= boundaries[s + 1])
                else:
                    mask = (q_updates >= boundaries[s]) & (q_updates < boundaries[s + 1])

                n_samples = int(mask.sum())
                for m in range(4):
                    ax = axes[s, m]
                    if mask.any():
                        _plot_aggregated(ax, steps[mask], metric_scores[m][mask],
                                         metric_colors[m])
                    ax.set_xlim(-0.5, max_step + 0.5)
                    ax.grid(True, alpha=0.3)
                    if m == 0:
                        ax.set_ylabel(
                            f'Q {int(boundaries[s])}-{int(boundaries[s+1])}\n(n={n_samples})',
                            fontsize=8)
                    if s == 0:
                        ax.set_title(metric_labels[m], fontsize=10)
        else:
            n_samples = len(steps)
            for m in range(4):
                _plot_aggregated(axes[0, m], steps, metric_scores[m], metric_colors[m])
                axes[0, m].set_title(metric_labels[m], fontsize=10)
                axes[0, m].set_xlim(-0.5, max_step + 0.5)
                axes[0, m].grid(True, alpha=0.3)
                if m == 0:
                    axes[0, m].set_ylabel(f'All\n(n={n_samples})', fontsize=8)

        for m in range(4):
            axes[-1, m].set_xlabel('Step in episode')

        fig.suptitle('Consequential Moments by Episode Step', fontsize=14, y=1.0)
        fig.tight_layout()
        step_plot_path = os.path.join(self.run_dir, 'consequence_step_scores.png')
        fig.savefig(step_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_diagnostics(self, plt):
        """3-panel diagnostics: return spread, score distribution, buffer health."""
        x = [r['q_update'] for r in self._history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Return distribution spread
        ax = axes[0]
        ax.plot(x, [r['return_spread_mean'] for r in self._history],
                label='Spread (std)', color='tab:blue')
        ax.plot(x, [r['return_range_mean'] for r in self._history],
                label='Range (max-min)', color='tab:orange')
        ax.set_xlabel('Q-update')
        ax.set_ylabel('Value')
        ax.set_title('Return distribution spread')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Score distribution (configured metric)
        ax = axes[1]
        means = np.array([r['score_mean'] for r in self._history])
        stds = np.array([r['score_std'] for r in self._history])
        p90 = [r['score_p90'] for r in self._history]
        ax.plot(x, means, label='Mean', color='tab:blue')
        ax.fill_between(x, means - stds, means + stds, alpha=0.2, color='tab:blue')
        ax.plot(x, p90, label='P90', color='tab:red', linestyle='--')
        ax.set_xlabel('Q-update')
        ax.set_ylabel('Score')
        ax.set_title(f'Score distribution ({self.metric_name})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Buffer health
        ax = axes[2]
        ax.plot(x, [r['buffer_scored_frac'] for r in self._history],
                label='Scored fraction', color='tab:green')
        ax2 = ax.twinx()
        ax2.plot(x, [r['buffer_cscore_mean'] for r in self._history],
                 label='Mean cscore', color='tab:purple', linestyle='--')
        ax.set_xlabel('Q-update')
        ax.set_ylabel('Fraction', color='tab:green')
        ax2.set_ylabel('Mean score', color='tab:purple')
        ax.set_title('Buffer health')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Consequence Scoring Diagnostics', fontsize=14)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)

    def close(self):
        """Final plot and close file handle."""
        self.plot()
        self._file.close()
