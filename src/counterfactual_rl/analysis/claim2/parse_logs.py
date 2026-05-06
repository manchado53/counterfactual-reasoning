"""Parse metrics.log files from a manifest into numpy arrays for rliable.

Output shapes:
    raw[alg]         (n_seeds, 1, n_checkpoints)  win_rate (or chess_score for chess)
    raw_length[alg]  (n_seeds, 1, n_checkpoints)  avg_length
    raw_allies[alg]  (n_seeds, 1, n_checkpoints)  avg_allies (SMAX only; 0 elsewhere)
    raw_wdl[alg]     (n_seeds, n_checkpoints, 3)  [win_rate, draw_rate, loss_rate] (chess)
    eval_steps[alg]  (n_checkpoints,)             cumulative env steps at each checkpoint

Steps-to-threshold unit convention:
    smax_3m / smax_8m : eval_steps[t] = episode number (one episode = one collection chunk)
    frozen_lake       : eval_steps[t] = episode number
    chess             : eval_steps[t] = chunk_idx * n_envs * collect_steps
                                       = chunk_idx * 256 * 256  (from config defaults)
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Column indices in metrics.log data rows
# SMAX / Chess (shared MetricsLogger):
#   episode  updates  epsilon  win_rate  avg_allies  avg_return  avg_length  chess_score  draw_rate  loss_rate
_SHARED_COLS = {
    'episode': 0, 'updates': 1, 'epsilon': 2,
    'win_rate': 3, 'avg_allies': 4, 'avg_return': 5, 'avg_length': 6,
    'chess_score': 7, 'draw_rate': 8, 'loss_rate': 9,
}

# FrozenLake (_MetricsLogger in dqn.py):
#   episode  updates  epsilon  win_rate  avg_length  avg_return
_FL_COLS = {
    'episode': 0, 'updates': 1, 'epsilon': 2,
    'win_rate': 3, 'avg_length': 4, 'avg_return': 5,
}

# Chess-specific: n_envs * collect_steps per chunk (from config defaults)
_CHESS_TRANSITIONS_PER_CHUNK = 256 * 256  # n_envs=256, collect_steps=256


def _detect_env(log_path: str) -> str:
    """Return 'chess', 'frozen_lake', or 'smax' based on header comments."""
    with open(log_path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'FrozenLake' in line or 'frozen_lake' in line.lower():
                return 'frozen_lake'
            if 'gardner_chess' in line.lower() or 'Chess' in line:
                return 'chess'
    return 'smax'


def _parse_single_log(log_path: str) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one metrics.log file.

    Returns (env_type, win_rate, avg_length, avg_allies, wdl, eval_steps).
    All arrays are 1-D with length = number of eval checkpoints.
    wdl has shape (n_checkpoints, 3).
    """
    env_type = _detect_env(log_path)
    cols = _FL_COLS if env_type == 'frozen_lake' else _SHARED_COLS

    win_rates, avg_lengths, avg_allies_list = [], [], []
    draw_rates, loss_rates, chess_scores = [], [], []
    episodes = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Skip header row (starts with 'episode')
            if line.split()[0] == 'episode':
                continue
            parts = line.split()
            try:
                ep = int(parts[cols['episode']])
                wr_str = parts[cols['win_rate']].rstrip('%')
                wr = float(wr_str) / 100.0 if '%' in parts[cols['win_rate']] else float(wr_str)
                al = float(parts[cols['avg_length']])
                episodes.append(ep)
                win_rates.append(wr)
                avg_lengths.append(al)

                if env_type != 'frozen_lake':
                    avg_allies_list.append(float(parts[cols['avg_allies']]))
                    if len(parts) > cols.get('chess_score', 99):
                        draw_rates.append(float(parts[cols['draw_rate']]))
                        loss_rates.append(float(parts[cols['loss_rate']]))
                        chess_scores.append(float(parts[cols['chess_score']]))
                    else:
                        draw_rates.append(0.0)
                        loss_rates.append(0.0)
                        chess_scores.append(wr)
                else:
                    avg_allies_list.append(0.0)
                    draw_rates.append(0.0)
                    loss_rates.append(0.0)
                    chess_scores.append(wr)
            except (IndexError, ValueError):
                continue

    n = len(episodes)
    if n == 0:
        raise ValueError(f"No eval rows found in {log_path}")

    ep_arr = np.array(episodes, dtype=np.float64)
    if env_type == 'chess':
        eval_steps = ep_arr * _CHESS_TRANSITIONS_PER_CHUNK
    else:
        eval_steps = ep_arr  # episodes directly

    primary = np.array(chess_scores if env_type == 'chess' else win_rates)
    wdl = np.stack([win_rates, draw_rates, loss_rates], axis=1)  # (n, 3)

    return (
        env_type,
        primary,
        np.array(avg_lengths),
        np.array(avg_allies_list),
        wdl,
        eval_steps,
    )


def _find_run_dir(job_id: str, env: str) -> Optional[str]:
    """Locate the run directory for a given SLURM job ID."""
    base_dirs = {
        'smax': os.path.join(os.path.dirname(__file__), '..', '..', 'agents', 'shared', 'runs'),
        'chess': os.path.join(os.path.dirname(__file__), '..', '..', 'agents', 'chess', 'runs'),
        'frozen_lake': os.path.join(os.path.dirname(__file__), '..', '..', 'agents', 'frozen_lake', 'runs'),
    }
    for env_name, base in base_dirs.items():
        candidate = os.path.normpath(os.path.join(base, job_id))
        if os.path.isdir(candidate):
            return candidate
    return None


def load_manifest(
    manifest_path: str,
    alg_key: str = 'algorithm',
    mixing_key: str = 'priority_mixing',
    mu_key: str = 'mu',
) -> Dict:
    """Load a manifest JSON and return parsed arrays keyed by algorithm label.

    Algorithm labelling:
        dqn-uniform                                 → 'DQN-Uniform'
        dqn                                         → 'DQN+PER'
        consequence-dqn, additive, mu=1.0           → 'DQN+CCE-only'
        consequence-dqn, additive, mu<1.0           → 'CCE+TD (add)'
        consequence-dqn, multiplicative             → 'CCE+TD (mul)'

    Returns dict with keys:
        raw          {alg: (n_seeds, 1, n_checkpoints)}
        raw_length   {alg: (n_seeds, 1, n_checkpoints)}
        raw_allies   {alg: (n_seeds, 1, n_checkpoints)}
        raw_wdl      {alg: (n_seeds, n_checkpoints, 3)}
        eval_steps   {alg: (n_checkpoints,)}  — from first seed (assumed aligned)
        env_type     str
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    def _label(cfg: dict) -> str:
        alg = cfg.get(alg_key, 'dqn-uniform')
        if alg == 'dqn-uniform':
            return 'DQN-Uniform'
        if alg == 'dqn':
            return 'DQN+PER'
        mixing = cfg.get(mixing_key, 'additive')
        mu = float(cfg.get(mu_key, 0.25))
        if mixing == 'multiplicative':
            return 'CCE+TD (mul)'
        if mu >= 1.0:
            return 'DQN+CCE-only'
        return 'CCE+TD (add)'

    per_alg: Dict[str, List] = defaultdict(list)
    run_dirs_per_alg: Dict[str, List] = defaultdict(list)
    env_types: Dict[str, str] = {}

    for job_id, cfg in manifest.items():
        label = _label(cfg)
        run_dir = _find_run_dir(job_id, '')
        if run_dir is None:
            print(f"Warning: run dir not found for job {job_id}, skipping")
            continue
        log_path = os.path.join(run_dir, 'metrics.log')
        if not os.path.isfile(log_path):
            print(f"Warning: no metrics.log in {run_dir}, skipping")
            continue
        try:
            env_type, primary, length, allies, wdl, steps = _parse_single_log(log_path)
            per_alg[label].append((primary, length, allies, wdl, steps))
            run_dirs_per_alg[label].append(run_dir)
            env_types[label] = env_type
        except ValueError as e:
            print(f"Warning: {e}")

    # Align checkpoint lengths across seeds (truncate to shortest)
    result_raw, result_len, result_all, result_wdl, result_steps = {}, {}, {}, {}, {}
    for alg, seed_data in per_alg.items():
        n_ckpts = min(d[0].shape[0] for d in seed_data)
        result_raw[alg] = np.stack([d[0][:n_ckpts] for d in seed_data])[:, np.newaxis, :]
        result_len[alg] = np.stack([d[1][:n_ckpts] for d in seed_data])[:, np.newaxis, :]
        result_all[alg] = np.stack([d[2][:n_ckpts] for d in seed_data])[:, np.newaxis, :]
        result_wdl[alg] = np.stack([d[3][:n_ckpts] for d in seed_data])
        result_steps[alg] = seed_data[0][4][:n_ckpts]

    env_type = next(iter(env_types.values())) if env_types else 'smax'
    return {
        'raw': result_raw,
        'raw_length': result_len,
        'raw_allies': result_all,
        'raw_wdl': result_wdl,
        'eval_steps': result_steps,
        'env_type': env_type,
        'run_dirs': dict(run_dirs_per_alg),
    }
