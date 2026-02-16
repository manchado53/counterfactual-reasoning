"""Metrics logging and plotting for SMAX DQN training."""

import os
from datetime import datetime
from typing import List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MetricsLogger:
    """Handles metrics log file creation, writing, plotting, and closing."""

    HEADER_COLUMNS = f"{'episode':>8} {'epsilon':>8} {'win_rate':>10} {'avg_allies':>12} {'avg_return':>12} {'avg_length':>12}\n"

    def __init__(self, backend: str, config: dict, env_info: dict,
                 n_episodes: int, eval_interval, eval_episodes: int):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        self.dir = f'metrics/{job_id}'
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, 'metrics.log')

        self._file = open(self.path, 'w')
        self._file.write(f"# SMAX DQN Training Metrics ({backend}) - {datetime.now()}\n")
        self._file.write(f"# Scenario: {env_info['scenario']}, Obs: {env_info['obs_type']}\n")
        self._file.write(f"# Episodes: {n_episodes}, Eval interval: {eval_interval}, Eval episodes: {eval_episodes}\n")
        self._file.write(f"#\n# === Hyperparameters ===\n")
        for k, v in config.items():
            self._file.write(f"# {k}: {v}\n")
        self._file.write(f"# =======================\n#\n")
        self._file.write(self.HEADER_COLUMNS)
        self._file.flush()

        # In-memory accumulation for eval curve plotting
        self._episodes = []
        self._epsilons = []
        self._win_rates = []
        self._avg_allies = []
        self._avg_returns = []
        self._avg_lengths = []

    def log_eval(self, episode: int, epsilon: float, metrics: dict):
        self._file.write(
            f"{episode:>8d} {epsilon:>8.3f} {metrics['win_rate']:>10.1%} "
            f"{metrics['avg_allies_alive']:>12.2f} {metrics['avg_return']:>12.2f} "
            f"{metrics['avg_length']:>12.1f}\n"
        )
        self._file.flush()

        self._episodes.append(episode)
        self._epsilons.append(epsilon)
        self._win_rates.append(metrics['win_rate'])
        self._avg_allies.append(metrics['avg_allies_alive'])
        self._avg_returns.append(metrics['avg_return'])
        self._avg_lengths.append(metrics['avg_length'])

    def plot_eval_curves(self):
        """Generate 2x2 eval curve plots from accumulated eval data."""
        if not self._episodes:
            return

        save_path = os.path.join(self.dir, 'eval_curves.png')

        episodes = np.array(self._episodes)
        win_rates = np.array(self._win_rates) * 100
        avg_returns = np.array(self._avg_returns)
        avg_allies = np.array(self._avg_allies)
        avg_lengths = np.array(self._avg_lengths)
        epsilons = np.array(self._epsilons)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Win Rate
        ax = axes[0, 0]
        ax.plot(episodes, win_rates, 'o-', color='#2196F3', markersize=2, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

        # Average Return
        ax = axes[0, 1]
        ax.plot(episodes, avg_returns, 'o-', color='#4CAF50', markersize=2, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Return')
        ax.set_title('Average Return')
        ax.grid(True, alpha=0.3)

        # Avg Allies Alive
        ax = axes[1, 0]
        ax.plot(episodes, avg_allies, 'o-', color='#FF9800', markersize=2, linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Allies Alive')
        ax.set_title('Average Allies Alive')
        ax.grid(True, alpha=0.3)

        # Epsilon + Avg Episode Length (dual axis)
        ax = axes[1, 1]
        color_eps = '#9C27B0'
        color_len = '#F44336'
        ax.plot(episodes, epsilons, '-', color=color_eps, linewidth=1.5, label='Epsilon')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon', color=color_eps)
        ax.tick_params(axis='y', labelcolor=color_eps)
        ax.set_ylim(-0.05, 1.05)

        ax2 = ax.twinx()
        ax2.plot(episodes, avg_lengths, 'o-', color=color_len, markersize=2, linewidth=1, label='Avg Length')
        ax2.set_ylabel('Avg Episode Length', color=color_len)
        ax2.tick_params(axis='y', labelcolor=color_len)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
        ax.set_title('Epsilon Decay & Episode Length')

        fig.suptitle(f'SMAX DQN Eval Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved eval curves to {save_path}")

    def plot_training_curves(self, episode_returns: List[float], episode_lengths: List[float]):
        """Plot per-episode training returns and lengths with moving average."""
        save_path = os.path.join(self.dir, 'training_curves.png')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Episode returns
        axes[0].plot(episode_returns, alpha=0.3, label='Episode Return')
        if len(episode_returns) >= 100:
            smoothed = np.convolve(episode_returns, np.ones(100) / 100, mode='valid')
            axes[0].plot(range(99, len(episode_returns)), smoothed, label='100-ep Average')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Return')
        axes[0].set_title('Training Returns')
        axes[0].legend()

        # Episode lengths
        axes[1].plot(episode_lengths, alpha=0.3, label='Episode Length')
        if len(episode_lengths) >= 100:
            smoothed = np.convolve(episode_lengths, np.ones(100) / 100, mode='valid')
            axes[1].plot(range(99, len(episode_lengths)), smoothed, label='100-ep Average')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Episode Lengths')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved training curves to {save_path}")

    def close(self):
        """Close the log file and auto-generate eval curve plots."""
        self._file.close()
        self.plot_eval_curves()
