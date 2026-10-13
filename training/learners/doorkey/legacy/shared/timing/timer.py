"""TrainingTimer — structured JSONL timing for SMAX training runs.

Per-episode accumulation: high-frequency components (env, action, buffer.add,
update) are accumulated in memory during each episode, then flushed as one
JSONL record per component at episode end.  Sparse components (eval,
update.scoring.*, update.q_update) write immediately since they fire rarely.
"""

import json
import os
import time


# Components whose per-step durations are accumulated per-episode
_ACCUMULATED = frozenset({'env', 'action', 'buffer.add', 'update'})


class _TimerContext:
    """Context manager that records duration and routes to accumulate or write."""

    def __init__(self, timer, name, episode, extra):
        self._timer = timer
        self._name = name
        self._episode = episode
        self._extra = extra

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *args):
        duration = time.monotonic() - self._t0
        if self._name in _ACCUMULATED:
            self._timer._accumulate(self._name, duration)
        else:
            self._timer._write_record(self._name, duration, self._episode, self._extra)


class TrainingTimer:
    """Writes structured JSONL timing records to a run directory.

    Usage:
        timer = TrainingTimer(run_dir)
        timer.start('total')

        for episode in range(n):
            timer.begin_episode(episode)
            with timer('env', episode=episode):
                ...
            with timer('update', episode=episode):
                ...
            timer.flush_episode()

        timer.stop('total')
        timer.close()
    """

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._path = os.path.join(run_dir, 'timing.jsonl')
        self._file = open(self._path, 'w')
        self._open_timers = {}  # name -> start time (for start/stop API)
        self._episode_accum = {}  # name -> accumulated duration
        self._current_episode = None

    def start(self, name):
        """Start a named timer (for long-running spans like 'total')."""
        self._open_timers[name] = time.monotonic()

    def stop(self, name, episode=None, **extra):
        """Stop a named timer and write its record."""
        t0 = self._open_timers.pop(name, None)
        if t0 is None:
            return
        duration = time.monotonic() - t0
        self._write_record(name, duration, episode, extra)

    def __call__(self, name, episode=None, **extra):
        """Return a context manager that times the enclosed block.

        Args:
            name: Component name (e.g. 'env', 'update.scoring.rollouts')
            episode: Current episode number
            **extra: Additional fields written to the JSONL record
        """
        return _TimerContext(self, name, episode, extra)

    def begin_episode(self, episode):
        """Start a new episode: flush any previous episode's accumulator."""
        if self._current_episode is not None:
            self._flush_accum()
        self._current_episode = episode
        self._episode_accum = {}

    def flush_episode(self):
        """Write one JSONL record per accumulated component for the current episode."""
        self._flush_accum()

    def _accumulate(self, name, duration):
        """Add duration to the per-episode accumulator for *name*."""
        self._episode_accum[name] = self._episode_accum.get(name, 0.0) + duration

    def _flush_accum(self):
        """Write accumulated durations for the current episode to JSONL."""
        ep = self._current_episode
        for name, duration in self._episode_accum.items():
            self._write_record(name, duration, ep, {})
        self._episode_accum = {}

    def _write_record(self, component, duration, episode, extra):
        record = {
            'component': component,
            'duration_s': round(duration, 6),
            'episode': episode,
        }
        if extra:
            record.update(extra)
        self._file.write(json.dumps(record) + '\n')
        self._file.flush()

    def close(self):
        """Flush, close file, generate per-run timing plots, delete JSONL."""
        # Flush any remaining episode data
        if self._episode_accum:
            self._flush_accum()
        self._file.close()
        # Auto-generate plots if there's data
        try:
            from .plot import plot_run_breakdown, plot_timeseries
            plot_run_breakdown(self._path, save_dir=self.run_dir)
            plot_timeseries(self._path, save_dir=self.run_dir)
        except Exception as e:
            print(f"Warning: failed to generate timing plots: {e}")
