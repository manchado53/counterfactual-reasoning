import json
import numpy as np
import pytest

from experiments.parity.run import first_difference, safe_file, snapshot, digest
from experiments.parity.replay_worker import pack, exercise
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight


def test_exact_comparison_detects_signed_zero_and_dtype():
    assert first_difference(pack(np.array([0.])), pack(np.array([-0.]))) == "$.bytes"
    assert first_difference(pack(np.array([1], dtype="i4")), pack(np.array([1], dtype="i8"))) == "$.dtype"
    assert first_difference({"x": [1, 2]}, {"x": [1, 3]}) == "$.x[1]"
    assert first_difference({"x": []}, {"x": [1]}) == "$.x: length"
    assert first_difference({"x": 1}, {"y": 1}) == "$: keys"
    assert first_difference({"x": 1}, {"x": 1}) is None


def test_safe_paths(tmp_path):
    for name in ("../escape", "/tmp/escape"):
        with pytest.raises(ValueError):
            safe_file(tmp_path, name)
    (tmp_path / "link").symlink_to("/etc/hosts")
    with pytest.raises(ValueError):
        safe_file(tmp_path, "link")


def test_snapshot_rejects_incomplete_manifest(tmp_path):
    (tmp_path / "hashes.json").write_text("{}")
    with pytest.raises(ValueError, match="required"):
        snapshot(tmp_path, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_snapshot_pins_source_and_rejects_drift(tmp_path):
    names = ["manifest.json", "versions.json", "scripts/decay_buffer.py"]
    names += [f"{c}/src/counterfactual_rl/agents/shared/consequence_buffers.py"
              for c in ("old", "fixed")]
    source = tmp_path / "source"
    for name in names:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    hashes = {name: digest(source/name) for name in names}
    (source / "hashes.json").write_text(json.dumps(hashes))
    copy = tmp_path / "copy"
    result = snapshot(source, copy)
    assert result["files"] == len(names)
    assert all(digest(copy/name) == h for name, h in hashes.items())
    (source / "scripts/decay_buffer.py").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        snapshot(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_trace_catches_wrong_importance_weights():
    cfg = dict(PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.),
               mu=.5, priority_mixing="multiplicative", mu_c=1., mu_delta=1.,
               normalise_priorities="rank", decay_start=100000, decay_end=150000)
    row = dict(config=cfg, seed=0, decay=True)
    original = exercise(DecayBuffer, blend_weight, row)
    assert first_difference(original, exercise(DecayBuffer, blend_weight, row)) is None

    class BrokenBuffer(DecayBuffer):
        def sample(self, n):
            batch, ids, weights = super().sample(n)
            return batch, ids, weights * 1.001

    mismatch = first_difference(original, exercise(BrokenBuffer, blend_weight, row))
    assert mismatch == "$[0].weights.bytes"


@pytest.mark.parametrize("mix,mu", [("multiplicative", .5), ("additive", .5), ("additive", 1.)])
def test_ranked_td_endpoint_independent_of_cce(mix, mu):
    b = DecayBuffer(capacity=5, normalise_priorities="rank", priority_mixing=mix, mu=mu)
    for i in range(5):
        b.add({"r": i})
    b.update_priorities(np.arange(5), np.array([3., -1., 1., 0., 7.]))
    b.update_consequence_scores(np.arange(5), np.array([90., 1., 7., 6., 0.]))
    # Independent average ranks for abs(TD)=[3,1,1,0,7]: [4,2.5,2.5,1,5].
    ranks = (np.array([4., 2.5, 2.5, 1., 5.]) - 1) / 4
    expected = (ranks + .01)**.25
    expected /= expected.sum()
    original = b._compute_priorities().copy()
    b.blend = 1.
    np.testing.assert_array_equal(b._compute_priorities(), expected)
    b.blend = .5
    np.testing.assert_array_equal(b._compute_priorities(), .5 * original + .5 * expected)
    b.blend = 0.
    np.testing.assert_array_equal(b._compute_priorities(), original)


@pytest.mark.parametrize("episode,expected", [(99999,0), (100000,0), (125000,.5), (150000,1), (150001,1)])
def test_production_clock(episode, expected):
    assert blend_weight(episode) == expected
    assert blend_weight(episode, False) == 0


def test_invalid_schedule():
    with pytest.raises(ValueError):
        blend_weight(1, start=3, end=3)
