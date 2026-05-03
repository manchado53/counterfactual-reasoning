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
from scipy.stats import spearmanr

from counterfactual_rl.analysis.metrics import compute_all_consequence_metrics


class ConsequenceDiagnostics:
    """Lightweight diagnostics for consequence scoring passes."""

    def __init__(self, run_dir: str, metric_name: str, plot_interval: int = 50,
                 n_step_slices: int = 10, n_scatter_snapshots: int = 10):
        self.run_dir = run_dir
        self.metric_name = metric_name
        self.plot_interval = plot_interval
        self.n_step_slices = n_step_slices
        self.n_scatter_snapshots = n_scatter_snapshots

        self.jsonl_path = os.path.join(run_dir, 'consequence_diagnostics.jsonl')
        self.plot_path = os.path.join(run_dir, 'consequence_diagnostics.png')
        self._file = open(self.jsonl_path, 'a')

        # In-memory history for plotting
        self._history: List[dict] = []
        self._pass_count = 0

        # Accumulated (episode_step, score, q_update) for step-vs-score plot
        self._step_score_data: List[tuple] = []

        # Priority snapshots for p_per vs p_consequence scatter plot
        self._priority_snapshots: List[dict] = []

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

        # --- Priority snapshot for scatter plot ---
        n = len(buffer)
        cs = np.nan_to_num(buffer.consequence_scores[:n], nan=0.0, posinf=0.0, neginf=0.0)
        td = np.nan_to_num(buffer.td_magnitudes[:n], nan=0.0, posinf=0.0, neginf=0.0)

        # p_per: TD-only (PER baseline)
        p_td_raw = (td + buffer.eps) ** buffer.beta
        p_per = p_td_raw / p_td_raw.sum() if p_td_raw.sum() > 0 else np.ones(n) / n

        # p_c: consequence-only component (normalized)
        p_c_raw = (cs + buffer.eps) ** buffer.beta
        p_c = p_c_raw / p_c_raw.sum() if p_c_raw.sum() > 0 else np.ones(n) / n

        # Additive (Eq 4): mu * p_C + (1-mu) * p_TD
        p_additive = buffer.mu * p_c + (1.0 - buffer.mu) * p_per
        p_additive /= p_additive.sum()

        # Multiplicative (Eq 5): p_C^mu_c * p_TD^mu_delta / Z
        p_mult = (p_c ** buffer.mu_c) * (p_per ** buffer.mu_delta)
        p_mult_sum = p_mult.sum()
        p_mult = p_mult / p_mult_sum if p_mult_sum > 0 else np.ones(n) / n

        # Spearman R between raw signals (consequence_scores vs td_magnitudes)
        r_raw, _ = spearmanr(cs, td) if n > 1 else (0.0, 1.0)

        # Spearman R between probability vectors (p_PER vs p_additive / p_multiplicative)
        r_add,  _ = spearmanr(p_per, p_additive) if n > 1 else (1.0, 1.0)
        r_mult, _ = spearmanr(p_per, p_mult)     if n > 1 else (1.0, 1.0)

        self._priority_snapshots.append({
            'q_update': q_update_count,
            'p_per': p_per,
            'p_additive': p_additive,
            'p_multiplicative': p_mult,
            'consequence_scores': cs.copy(),
            'spearman_r': float(r_raw)  if np.isfinite(r_raw)  else 0.0,
            'spearman_r_add':  float(r_add)  if np.isfinite(r_add)  else 1.0,
            'spearman_r_mult': float(r_mult) if np.isfinite(r_mult) else 1.0,
        })

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
        self._plot_priority_scatter(plt)

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

    def _plot_priority_scatter(self, plt):
        """Hexbin priority scatter: two sections (additive / multiplicative) + Spearman R.

        Layout (n_cols=5, n_sr = n_scatter_rows):
          Rows 0      .. n_sr-1        : additive  — % mass  (Blues)
          Rows n_sr   .. 2*n_sr-1      : additive  — count   (GnBu)
          Rows 2*n_sr .. 3*n_sr-1      : mult      — % mass  (Oranges)
          Rows 3*n_sr .. 4*n_sr-1      : mult      — count   (YlOrRd)
          Row  4*n_sr                  : Spearman R (raw)
          Row  4*n_sr + 1              : Spearman R (probability)

        Mass hex color  = Σ p_PER in bin (% of total sampling budget).
        Count hex color = number of transitions in bin.
        Compare same snapshot: dark mass + light count → few high-priority transitions.
        """
        if len(self._priority_snapshots) < 1:
            return

        import matplotlib.gridspec as gridspec

        n_snaps = len(self._priority_snapshots)
        n_scatter = min(self.n_scatter_snapshots, n_snaps)
        scatter_indices = np.linspace(0, n_snaps - 1, n_scatter, dtype=int)
        scatter_snaps = [self._priority_snapshots[i] for i in scatter_indices]

        n_cols = 5
        n_sr = (n_scatter + n_cols - 1) // n_cols  # rows per sub-section
        n_rows = 4 * n_sr + 2
        fig_height = 3.5 * 4 * n_sr + 7.0
        fig = plt.figure(figsize=(26, fig_height))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.40,
                               top=0.93, bottom=0.05, left=0.05, right=0.97)

        def _draw_mass_section(row_offset, p_key, ylabel, cmap, section_title):
            first_ax = None
            for i, snap in enumerate(scatter_snaps):
                row, col = divmod(i, n_cols)
                ax = fig.add_subplot(gs[row_offset + row, col])
                if i == 0:
                    first_ax = ax

                p_per = snap['p_per']
                p_y   = snap[p_key]
                q_upd = snap['q_update']
                N     = len(p_per)

                hb = ax.hexbin(p_per * 100, p_y * 100, C=p_per * 100,
                               reduce_C_function=np.sum,
                               gridsize=35, cmap=cmap, mincnt=1, linewidths=0.2)
                cb = plt.colorbar(hb, ax=ax)
                cb.ax.tick_params(labelsize=5)
                cb.set_label('% mass', fontsize=5)

                lim_max = max((p_per * 100).max(), (p_y * 100).max()) * 1.05
                ax.plot([0, lim_max], [0, lim_max], '--', color='lime',
                        alpha=0.9, linewidth=1.2)

                ax.set_title(f'Q-upd {q_upd:,}  N={N:,}', fontsize=8)
                ax.set_xlabel('p_PER (% mass)', fontsize=7)
                ax.set_ylabel(ylabel, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            if first_ax is not None:
                first_ax.set_title(
                    f'[{section_title} — % mass]  ' + first_ax.get_title(),
                    fontsize=8, fontweight='bold'
                )

        def _draw_count_section(row_offset, p_key, ylabel, cmap, section_title):
            first_ax = None
            for i, snap in enumerate(scatter_snaps):
                row, col = divmod(i, n_cols)
                ax = fig.add_subplot(gs[row_offset + row, col])
                if i == 0:
                    first_ax = ax

                p_per = snap['p_per']
                p_y   = snap[p_key]
                q_upd = snap['q_update']
                N     = len(p_per)

                pct_per_transition = np.full(N, 100.0 / N)
                hb = ax.hexbin(p_per * 100, p_y * 100, C=pct_per_transition,
                               reduce_C_function=np.sum,
                               gridsize=35, cmap=cmap, mincnt=1, linewidths=0.2)
                cb = plt.colorbar(hb, ax=ax)
                cb.ax.tick_params(labelsize=5)
                cb.set_label('% transitions', fontsize=5)

                lim_max = max((p_per * 100).max(), (p_y * 100).max()) * 1.05
                ax.plot([0, lim_max], [0, lim_max], '--', color='lime',
                        alpha=0.9, linewidth=1.2)

                ax.set_title(f'Q-upd {q_upd:,}  N={N:,}', fontsize=8)
                ax.set_xlabel('p_PER (% mass)', fontsize=7)
                ax.set_ylabel(ylabel, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

            if first_ax is not None:
                first_ax.set_title(
                    f'[{section_title} — % transitions]  ' + first_ax.get_title(),
                    fontsize=8, fontweight='bold'
                )

        _draw_mass_section(0,        'p_additive',       'p_add (%)',  'Blues',   'ADDITIVE')
        _draw_count_section(n_sr,    'p_additive',       'p_add (%)',  'GnBu',    'ADDITIVE')
        _draw_mass_section(2 * n_sr, 'p_multiplicative', 'p_mult (%)', 'Oranges', 'MULTIPLICATIVE')
        _draw_count_section(3 * n_sr,'p_multiplicative', 'p_mult (%)', 'YlOrRd',  'MULTIPLICATIVE')

        # ── Spearman R: raw consequence scores vs TD magnitudes ──────────────
        ax_r = fig.add_subplot(gs[4 * n_sr, :])
        q_updates   = [s['q_update']   for s in self._priority_snapshots]
        spearman_rs = [s['spearman_r'] for s in self._priority_snapshots]

        ax_r.plot(q_updates, spearman_rs, color='tab:purple', linewidth=1.5)
        ax_r.axhline(y=0.0, color='white', linestyle='--', alpha=0.4, linewidth=1,
                     label='r=0  (fully independent)')
        ax_r.axhline(y=1.0, color='red', linestyle=':', alpha=0.4, linewidth=1,
                     label='r=1  (identical to PER)')
        ax_r.set_xlabel('Q-update')
        ax_r.set_ylabel('Spearman r')
        ax_r.set_title('Rank correlation: consequence scores vs TD magnitudes\n'
                        'Near 0 = consequence signal is independent from TD error', fontsize=9)
        ax_r.set_ylim(-0.1, 1.05)
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3)

        # ── Spearman R: p_PER vs p_additive / p_multiplicative ───────────────
        ax_r2 = fig.add_subplot(gs[4 * n_sr + 1, :])
        r_adds  = [s['spearman_r_add']  for s in self._priority_snapshots]
        r_mults = [s['spearman_r_mult'] for s in self._priority_snapshots]

        ax_r2.plot(q_updates, r_adds,  color='tab:blue',   linewidth=1.5, label='Additive vs PER')
        ax_r2.plot(q_updates, r_mults, color='tab:orange', linewidth=1.5, label='Multiplicative vs PER')
        ax_r2.axhline(y=1.0, color='red',   linestyle=':', alpha=0.4, linewidth=1,
                      label='r=1  (identical to PER)')
        ax_r2.axhline(y=0.0, color='white', linestyle='--', alpha=0.4, linewidth=1,
                      label='r=0  (fully different from PER)')
        ax_r2.set_xlabel('Q-update')
        ax_r2.set_ylabel('Spearman r')
        ax_r2.set_title('Rank correlation: p_PER vs p_consequence (sampling distribution similarity)\n'
                         'Near 1 = consequence-DQN samples same transitions as PER; '
                         'Near 0 = sampling very different transitions', fontsize=9)
        ax_r2.set_ylim(-0.1, 1.05)
        ax_r2.legend(fontsize=8)
        ax_r2.grid(True, alpha=0.3)

        fig.suptitle(
            'Priority distribution: consequence-DQN vs PER  —  hex color = Σ p_PER mass per bin\n'
            'Above diagonal → consequence upweights vs PER;  '
            'dark cluster above diagonal = meaningful mass shift',
            fontsize=11, fontweight='bold',
        )

        scatter_path = os.path.join(self.run_dir, 'consequence_priority_scatter.png')
        fig.savefig(scatter_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _plot_additive_vs_mult_comparison(self, plt):
        """Side-by-side additive vs multiplicative, % mass and % transitions.

        Layout: 4 rows × 5 cols
          Row 0: additive       — % mass        (Blues)
          Row 1: additive       — % transitions  (GnBu)
          Row 2: multiplicative — % mass        (Oranges)
          Row 3: multiplicative — % transitions  (YlOrRd)

        Same axis limits per column → direct visual comparison across modes and metrics.
        """
        if len(self._priority_snapshots) < 1:
            return

        n_snaps = len(self._priority_snapshots)
        n_cols = min(5, n_snaps)
        snap_indices = np.linspace(0, n_snaps - 1, n_cols, dtype=int)
        snaps = [self._priority_snapshots[i] for i in snap_indices]

        fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 17),
                                 gridspec_kw={'hspace': 0.50, 'wspace': 0.38})
        if n_cols == 1:
            axes = axes.reshape(4, 1)

        # (p_key, cmap, colorbar_label, row_label)
        rows_def = [
            ('p_additive',       'Blues',   '% mass',        'Additive — % mass'),
            ('p_additive',       'GnBu',    '% transitions', 'Additive — % transitions'),
            ('p_multiplicative', 'Oranges', '% mass',        'Multiplicative — % mass'),
            ('p_multiplicative', 'YlOrRd',  '% transitions', 'Multiplicative — % transitions'),
        ]

        for col, snap in enumerate(snaps):
            p_per = snap['p_per'] * 100
            q_upd = snap['q_update']
            N     = len(snap['p_per'])
            lim_max = max(p_per.max(),
                          snap['p_additive'].max() * 100,
                          snap['p_multiplicative'].max() * 100) * 1.05
            pct_per_transition = np.full(N, 100.0 / N)

            for row, (key, cmap, cb_label, row_label) in enumerate(rows_def):
                ax = axes[row, col]
                p_y = snap[key] * 100

                if cb_label == '% mass':
                    hb = ax.hexbin(p_per, p_y, C=p_per, reduce_C_function=np.sum,
                                   gridsize=30, cmap=cmap, mincnt=1, linewidths=0.2)
                else:
                    hb = ax.hexbin(p_per, p_y, C=pct_per_transition,
                                   reduce_C_function=np.sum,
                                   gridsize=30, cmap=cmap, mincnt=1, linewidths=0.2)

                cb = plt.colorbar(hb, ax=ax)
                cb.ax.tick_params(labelsize=6)
                cb.set_label(cb_label, fontsize=6)

                ax.plot([0, lim_max], [0, lim_max], '--', color='lime',
                        alpha=0.9, linewidth=1.2)
                ax.set_xlim(0, lim_max)
                ax.set_ylim(0, lim_max)

                title = f'Q-upd {q_upd:,}  N={N:,}'
                if col == 0:
                    title = f'[{row_label}]\n' + title
                ax.set_title(title, fontsize=8,
                             fontweight='bold' if col == 0 else 'normal')
                ax.set_xlabel('p_PER (% mass)', fontsize=7)
                ax.set_ylabel(f'p_{key[2:5]} ({cb_label})', fontsize=7)
                ax.tick_params(labelsize=6)

        fig.suptitle(
            'Additive vs Multiplicative — same snapshots, same axis limits per column\n'
            'Rows 0–1: Additive  |  Rows 2–3: Multiplicative  |  '
            'Odd rows: % mass, Even rows: % transitions',
            fontsize=11, fontweight='bold',
        )

        path = os.path.join(self.run_dir, 'consequence_additive_vs_mult_comparison.png')
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _save_snapshots(self):
        """Persist priority snapshots to disk so plots can be regenerated later."""
        if not self._priority_snapshots:
            return
        arrays = {}
        for i, snap in enumerate(self._priority_snapshots):
            arrays[f'{i}_q_update']          = np.array([snap['q_update']])
            arrays[f'{i}_p_per']             = snap['p_per']
            arrays[f'{i}_p_additive']        = snap['p_additive']
            arrays[f'{i}_p_multiplicative']  = snap['p_multiplicative']
            arrays[f'{i}_consequence_scores'] = snap['consequence_scores']
            arrays[f'{i}_spearman_r']        = np.array([snap['spearman_r']])
            arrays[f'{i}_spearman_r_add']    = np.array([snap['spearman_r_add']])
            arrays[f'{i}_spearman_r_mult']   = np.array([snap['spearman_r_mult']])
        np.savez_compressed(
            os.path.join(self.run_dir, 'priority_snapshots.npz'), **arrays
        )

    def close(self):
        """Final plot and close file handle."""
        self.plot()
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self._plot_additive_vs_mult_comparison(plt)
        except ImportError:
            pass
        self._save_snapshots()
        self._file.close()
