"""Plateau-based early stopping.

Unlike a target-score early stop (stop once a metric hits X), this stops once
the metric has stopped IMPROVING, whatever level it plateaued at. Useful when
different algorithms converge to different final performance and there's no
single shared "success" threshold (e.g. CCE vs PER on JaxNav's holes map).
"""

from typing import List


class PlateauEarlyStopper:
    """
    Call ``update(episode, value)`` once per eval checkpoint. Returns True once
    training should stop: the smoothed value hasn't improved by ``min_delta``
    over the best smoothed value seen, for ``patience`` consecutive checkpoints.

    A single eval is noisy (e.g. 100 test episodes can swing +-10-15 points
    checkpoint to checkpoint), so the comparison runs on a rolling mean
    (``smooth_window``) rather than the raw value.

    ``min_episodes`` is a warmup floor — no stop decision is made before it
    (e.g. set to ``epsilon_decay_episodes``, so a still-mostly-random policy
    never reads as "plateaued").
    """

    def __init__(self, patience: int, min_delta: float, smooth_window: int, min_episodes: int):
        self.patience = patience
        self.min_delta = min_delta
        self.smooth_window = smooth_window
        self.min_episodes = min_episodes

        self._history: List[float] = []
        self.best: float = -1.0
        self.last_smoothed: float = 0.0
        self._since_improve: int = 0

    def update(self, episode: int, value: float) -> bool:
        self._history.append(value)
        window = self._history[-self.smooth_window:]
        self.last_smoothed = sum(window) / len(window)

        if episode < self.min_episodes:
            return False

        if self.last_smoothed > self.best + self.min_delta:
            self.best = self.last_smoothed
            self._since_improve = 0
            return False

        self._since_improve += 1
        return self._since_improve >= self.patience

    def status(self) -> str:
        return (f"smoothed={self.last_smoothed:.1%}  best={self.best:.1%}  "
                f"no-improve={self._since_improve}/{self.patience}")
