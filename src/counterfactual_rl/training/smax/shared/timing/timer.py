"""TrainingTimer — structured JSONL timing for SMAX training runs."""

import json
import os
import time


class _NullContext:
    """No-op context manager when timing is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

_NULL = _NullContext()


class _TimerContext:
    """Context manager that records duration and writes a JSONL record on exit."""

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
        self._timer._write_record(self._name, duration, self._episode, self._extra)


class TrainingTimer:
    """Writes structured JSONL timing records to a run directory.

    Usage:
        timer = TrainingTimer(run_dir)
        timer.start('total')

        for episode in range(n):
            sampled = timer.is_sampled(episode)
            with timer('env', episode=episode, enabled=sampled):
                ...
            with timer('update', episode=episode, enabled=sampled):
                ...

        timer.stop('total')
        timer.close()
    """

    def __init__(self, run_dir, sample_interval=100):
        self.run_dir = run_dir
        self.sample_interval = sample_interval
        self._path = os.path.join(run_dir, 'timing.jsonl')
        self._file = open(self._path, 'w')
        self._open_timers = {}  # name -> start time (for start/stop API)

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

    def __call__(self, name, episode=None, enabled=True, **extra):
        """Return a context manager that times the enclosed block.

        Args:
            name: Component name (e.g. 'env', 'update.scoring.rollouts')
            episode: Current episode number
            enabled: If False, returns a no-op context manager
            **extra: Additional fields written to the JSONL record
        """
        if not enabled:
            return _NULL
        return _TimerContext(self, name, episode, extra)

    def is_sampled(self, episode):
        """Return True if this episode should be timed (for high-frequency ops)."""
        return episode % self.sample_interval == 0

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
        """Flush, close file, and auto-generate per-run timing plots."""
        self._file.close()
        # Auto-generate plots if there's data
        try:
            from .plot import plot_run_breakdown, plot_timeseries
            plot_run_breakdown(self._path, save_dir=self.run_dir)
            plot_timeseries(self._path, save_dir=self.run_dir)
        except Exception as e:
            print(f"Warning: failed to generate timing plots: {e}")
