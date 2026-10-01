"""Target draw logs must survive the imported JaxNav buffer features."""
import numpy as np
import pytest

from counterfactual_rl.agents.shared.buffers import PrioritizedReplayBuffer
from counterfactual_rl.agents.shared.consequence_buffers import ConsequenceReplayBuffer


@pytest.mark.parametrize('kind', [PrioritizedReplayBuffer, ConsequenceReplayBuffer])
def test_draw_log_survives_wrap_and_dump(kind, tmp_path):
    buf = kind(capacity=3)
    buf.enable_draw_log = True
    for i in range(5):
        buf.add({'s': i, 'a': 0, 'r': float(i), "s'": i + 1, 'done': False})
    assert buf.reward_diagnostics()['buffer_size'] == 3
    draws = [t for _ in range(4) for t in buf.sample(3)[0]]
    assert sum(buf._add_counts.values()) == 5
    assert sum(buf._draw_counts.values()) == 12
    assert {t['s'] for t in draws} <= {2, 3, 4}
    buf.snapshot_draws(7, episode=2)
    path = tmp_path / 'sampling.npz'
    buf.dump_sampling(path, 6, 1)
    with np.load(path) as data:
        assert data['total_adds'] == 5
        assert data['total_draws'] == 12
        assert data['updates'].tolist() == [7]


def test_batch_pytree_scores_and_draw_logs():
    buf = ConsequenceReplayBuffer(capacity=3)
    buf.enable_draw_log = True
    transitions = {'s': np.arange(5), 'a': np.zeros(5, dtype=int),
                   'r': np.ones(5), "s'": np.arange(1, 6), 'done': np.zeros(5, dtype=bool)}
    slots = buf.add_batch(transitions, jax_states={'position': np.arange(10).reshape(5, 2)})
    np.testing.assert_array_equal(slots, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(buf.get_jax_state(0)['position'], [6, 7])
    assert sum(buf._add_counts.values()) == 5
    before = buf._compute_priorities().copy()
    buf.update_consequence_scores(np.arange(3), np.array([.1, .5, .9]), q_update_count=7)
    assert np.all(buf.scored_at == 7)
    assert not np.allclose(before, buf._compute_priorities())
    for _ in range(3):
        buf.sample(3)
    assert sum(buf._draw_counts.values()) == 9
