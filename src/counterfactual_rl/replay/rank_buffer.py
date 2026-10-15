"""Standard inverse-rank PER atop the shared CCE buffer storage and sampler.

This is distinct from CCE's ascending percentile normalization. Rank one has
largest |TD| and p(i) is proportional to rank(i)**(-alpha). Average tied ranks
avoid inventing a preference among equal errors. Shared importance weights
remain 1/(N*p), for every non-uniform arm.
"""
import numpy as np
from scipy.stats import rankdata
from .consequence_buffers import ConsequenceReplayBuffer


class SelectionReplayBuffer(ConsequenceReplayBuffer):
    def __init__(self, *args, selection, rank_alpha=.25, **kwargs):
        super().__init__(*args, **kwargs)
        if selection not in ('uniform', 'rank_per', 'cce'):
            raise ValueError(selection)
        self.selection = selection
        self.rank_alpha = rank_alpha

    def _compute_priorities(self):
        if self.selection == 'cce':
            return super()._compute_priorities()
        if self._cached_probs is None:
            if self.selection == 'uniform':
                p = np.ones(len(self), dtype=float)
            else:
                p = rankdata(-self.td_magnitudes[:len(self)], method='average') ** -self.rank_alpha
            self._cached_probs = p / p.sum()
        return self._cached_probs
