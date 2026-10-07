"""Probability interpolation; original ranked sampler is unchanged at weight zero."""
import numpy as np
from counterfactual_rl.agents.shared.consequence_buffers import ConsequenceReplayBuffer, _normalise


def blend_weight(episode, enabled=True, start=100000, end=150000):
    if end <= start:
        raise ValueError('end must exceed start')
    return float(np.clip((episode-start)/(end-start), 0., 1.)) if enabled else 0.


class DecayBuffer(ConsequenceReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.normalise_priorities != 'rank' or self.target_ess_frac is not None:
            raise ValueError('This experiment requires fixed rank normalization')
        self.blend = 0.
        self._blend_key = None
        self._blend_probs = None

    def ranked_td(self):
        td = np.nan_to_num(self.td_magnitudes[:self._size].copy(), nan=0., posinf=0., neginf=0.)
        weights = (_normalise(td, 'rank') + self.eps)**self.beta
        return weights/weights.sum()

    def _compute_priorities(self):
        # Parent cache always holds ORIGINAL probabilities, never the blend.
        original = super()._compute_priorities()
        if self.blend == 0.:
            return original
        if self._blend_key is not original or self._blend_value != self.blend:
            td = self.ranked_td()
            self._blend_probs = td if self.blend == 1. else (1-self.blend)*original+self.blend*td
            self._blend_key = original
            self._blend_value = self.blend
        return self._blend_probs
