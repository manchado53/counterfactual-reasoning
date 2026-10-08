"""Shared realized-replay-sampling instrumentation (Option B: precision@k / ESS).

Mixed into BOTH replay buffers (ConsequenceReplayBuffer, PrioritizedReplayBuffer) so logging is
identical and DRY — letting us compare what CCE drilled vs what plain PER / uniform drilled.
Gated by `enable_draw_log` (default off → zero cost).

Logs two streams, both keyed by the transition's (state, action) from the stored dict ('s','a'):
  - DRAWS  : how often each (s,a) was pulled OUT to learn from (demand).
  - ADDS   : how often each (s,a) was dropped IN to the buffer (supply ≈ visitation).
The supply stream is what de-confounds precision@k: raw draws ≈ visitation, but draws normalized
by supply isolates whether the priority over-weights a state beyond its fair share.

Snapshots of BOTH streams are taken on a fixed Q-update cadence (labeled by q_update_count, since
the trigger fires on gradient steps — episode labels would collide) so oversampling can be
computed per training window (within a window, cumulative adds ≈ buffer occupancy even though the
ring buffer wraps over the full run).
"""

import numpy as np


class DrawLogMixin:
    def _init_draw_log(self):
        """Call from the buffer __init__. Sets up the (off-by-default) draw log."""
        self.enable_draw_log = False
        self._draw_counts = {}            # {(state, action): count}  — pulled out
        self._add_counts = {}             # {(state, action): count}  — dropped in (supply)
        self._draw_snapshots = []         # list of dict copies (drift)
        self._add_snapshots = []
        self._snapshot_updates = []       # q_update_count at each snapshot (NOT episode — fixes label collision)
        self._snapshot_eps = []           # episode too (for reference)

    def _record_draws(self, transitions):
        """Tally one sampled batch by (state, action). No-op when logging is off."""
        if not getattr(self, 'enable_draw_log', False):
            return
        for t in transitions:
            if not isinstance(t, dict) or 's' not in t or 'a' not in t:
                continue
            key = (int(t['s']), int(t['a']))
            self._draw_counts[key] = self._draw_counts.get(key, 0) + 1

    def _record_add(self, transition):
        """Tally one added transition by (state, action) — the supply stream. No-op when off."""
        if not getattr(self, 'enable_draw_log', False):
            return
        if not isinstance(transition, dict) or 's' not in transition or 'a' not in transition:
            return
        key = (int(transition['s']), int(transition['a']))
        self._add_counts[key] = self._add_counts.get(key, 0) + 1

    def snapshot_draws(self, update_count, episode=-1):
        """Append a copy of the cumulative draw + add counts (for windowed / drift analysis).

        Labeled by `update_count` (gradient steps) since the trigger fires on Q-updates."""
        if not getattr(self, 'enable_draw_log', False):
            return
        self._draw_snapshots.append(dict(self._draw_counts))
        self._add_snapshots.append(dict(self._add_counts))
        self._snapshot_updates.append(int(update_count))
        self._snapshot_eps.append(int(episode))

    def dump_sampling(self, path, n_states, n_actions):
        """Write realized replay draw + supply counts to an npz. Safe even if logging was off."""
        def densify(counts):
            arr = np.zeros((n_states, n_actions), dtype=np.int64)
            for (s, a), c in counts.items():
                if 0 <= s < n_states and 0 <= a < n_actions:
                    arr[s, a] = c
            return arr

        def densify_stack(snaps):
            if snaps:
                return np.stack([densify(c) for c in snaps])
            return np.zeros((0, n_states, n_actions), dtype=np.int64)

        cumulative = densify(self._draw_counts)
        adds = densify(self._add_counts)
        np.savez_compressed(
            path,
            cumulative=cumulative,
            adds=adds,
            snapshots=densify_stack(self._draw_snapshots),
            adds_snapshots=densify_stack(self._add_snapshots),
            updates=np.array(self._snapshot_updates, dtype=np.int64),
            episodes=np.array(self._snapshot_eps, dtype=np.int64),
            total_draws=np.int64(cumulative.sum()),
            total_adds=np.int64(adds.sum()),
            n_states=np.int64(n_states), n_actions=np.int64(n_actions),
        )
