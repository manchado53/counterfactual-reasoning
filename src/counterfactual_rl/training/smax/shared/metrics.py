"""Metrics logging for SMAX DQN training."""

import os
from datetime import datetime


class MetricsLogger:
    """Handles metrics log file creation, writing, and closing."""

    HEADER_COLUMNS = f"{'episode':>8} {'epsilon':>8} {'win_rate':>10} {'avg_allies':>12} {'avg_return':>12} {'avg_length':>12}\n"

    def __init__(self, backend: str, config: dict, env_info: dict,
                 n_episodes: int, eval_interval, eval_episodes: int):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        self.path = f'logs/metrics_{job_id}.log'
        os.makedirs('logs', exist_ok=True)

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

    def log_eval(self, episode: int, epsilon: float, metrics: dict):
        self._file.write(
            f"{episode:>8d} {epsilon:>8.3f} {metrics['win_rate']:>10.1%} "
            f"{metrics['avg_allies_alive']:>12.2f} {metrics['avg_return']:>12.2f} "
            f"{metrics['avg_length']:>12.1f}\n"
        )
        self._file.flush()

    def close(self):
        self._file.close()