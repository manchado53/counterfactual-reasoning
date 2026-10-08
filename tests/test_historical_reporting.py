"""Public manifest reader and frozen-output gate; no training imports."""
import hashlib
import json
from pathlib import Path

import pytest

from analysis.historical_reporting import read_manifest, equality_gate


def example(tmp_path):
    runs = []
    for seed, rows in enumerate([[(1, .2), (2, .4), (5000, .8), (10000, .9)],
                                 [(1, .8), (5000, .6), (9000, .5)]]):
        blob = b"".join((json.dumps(dict(episode=e, win_rate=w)) + "\n").encode()
                        for e, w in rows)
        path = tmp_path / f"{seed}.jsonl"
        path.write_bytes(blob)
        runs.append(dict(arm="mul", group="Old / decay", seed=seed,
                         path=path.name, sha256=hashlib.sha256(blob).hexdigest(),
                         bytes=len(blob), count=len(rows), format="eval"))
    manifest = dict(schema=1, protocol="decay-20261007-v1", runs=runs,
                    absent=[], banks=[], snapshot="test")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_complete_bins_equal_seed_mean_and_thin_tails(tmp_path):
    data = read_manifest(example(tmp_path))
    lines = data["panels"][0]["lines"]
    curve = next(line for line in lines if line["label"] == "Old / decay: mean of 2/5 seeds")
    assert curve["x"][:3] == [2.5, 7.5, 12.5]
    assert curve["y"][:3] == [.55, None, None]
    thin = [line for line in lines if line["label"] is None]
    assert thin[0]["y"][:3] == [pytest.approx(.3), .8, None]
    assert thin[1]["y"][:3] == [.8, None, None]
    assert data["summary"][0]["mean_through"] == 5000


def test_gate_rejects_changed_tail_and_labels(tmp_path):
    # Independent tiny old-output stand-in tests the public comparison, not bins.
    frozen = tmp_path / "old.json"
    frozen.write_text('{"label": "Old / decay", "y": [0.5, null]}')
    expected = dict(label="Old / decay", y=[.5, None])
    assert equality_gate(expected, frozen)["status"] == "pass"
    for wrong in (dict(label="Fixed / decay", y=[.5, None]),
                  dict(label="Old / decay", y=[.5, .5])):
        report = equality_gate(wrong, frozen)
        assert report["status"] == "fail"
        assert report["first_difference"]


def test_pinned_prefix_ignores_append_but_rejects_drift(tmp_path):
    path = example(tmp_path)
    before = read_manifest(path)
    log = tmp_path / "0.jsonl"
    log.write_bytes(log.read_bytes() + b'{"episode": 15000, "win_rate": 1}\n')
    assert read_manifest(path) == before
    log.write_bytes(log.read_bytes().replace(b'0.2', b'0.7', 1))
    with pytest.raises(ValueError, match="hash"):
        read_manifest(path)


def test_duplicate_and_missing_inputs_fail(tmp_path):
    path = example(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["runs"].append(manifest["runs"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)
    manifest["runs"].pop()
    path.write_text(json.dumps(manifest))
    (tmp_path / "0.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        read_manifest(path)


def test_gate_checks_seed_identity_even_when_mean_is_unchanged(tmp_path):
    path = example(tmp_path)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(read_manifest(path)))
    manifest = json.loads(path.read_text())
    manifest["runs"][0]["seed"], manifest["runs"][1]["seed"] = 1, 0
    # Source identity is compared even though the plotted mean is unchanged.
    expected = read_manifest(path)
    path.write_text(json.dumps(manifest))
    actual = read_manifest(path)
    assert actual["panels"] == expected["panels"]
    assert actual["selection"] != expected["selection"]
    assert equality_gate(actual, frozen)["status"] == "fail"


def test_gzip_bundle_has_same_pinned_bytes(tmp_path):
    import gzip
    path = example(tmp_path)
    manifest = json.loads(path.read_text())
    for entry in manifest["runs"]:
        log = tmp_path / entry["path"]
        log.write_bytes(gzip.compress(log.read_bytes(), mtime=0))
        entry["compression"] = "gzip"
    path.write_text(json.dumps(manifest))
    assert read_manifest(path)["summary"][0]["mean_through"] == 5000


def test_five_arm_closed_right_bins_mean_median_and_sem(tmp_path):
    from analysis.historical_five_arm import read_manifest as read_five
    runs = []
    for arm in ("Uniform", "PER", "CCE Mul", "CCE Only", "CCE Add 50/50"):
        for seed, score in enumerate([0., 0., 0., 0., 1.]):
            rows = [dict(episode=0, win_rate=1.)] + [
                dict(episode=e, win_rate=score) for e in range(5000, 250001, 5000)]
            blob = "\n".join(json.dumps(r) for r in rows).encode()
            name = f"{len(runs)}.log"
            (tmp_path / name).write_bytes(blob)
            runs.append(dict(path=name, origin=name, arm=arm, seed=seed, config={},
                             bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    path = tmp_path / "five.json"
    path.write_text(json.dumps(dict(schema=1, protocol="five-arm-20261007-v1", runs=runs)))
    result = read_five(path)
    assert result["bin_centers"][:2] == [2500., 7500.]
    assert result["summaries"]["CCE Mul"]["mean"] == [.2] * 50
    assert result["summaries"]["CCE Mul"]["median"] == [0.] * 50
    assert result["summaries"]["CCE Mul"]["sem"] == pytest.approx([.2] * 50)


@pytest.mark.parametrize("seed,arm,group", [(5, "mul", "Old / decay"), (0, "Uniform", "Old / decay")])
def test_decay_rejects_invalid_manifest_identity(tmp_path, seed, arm, group):
    path = example(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["runs"][0].update(seed=seed, arm=arm, group=group)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        read_manifest(path)


@pytest.mark.parametrize("bundle,reader", [
    ("decay-20261007-2224-v2", "analysis.historical_reporting"),
    ("five-arm-20261007", "analysis.historical_five_arm"),
])
def test_real_saved_logs_match_frozen_old_outputs(bundle, reader):
    import importlib
    root = Path(__file__).resolve().parents[1] / "analysis/historical_reproduction" / bundle
    manifest = json.loads((root / "manifest.json").read_text())
    frozen = root / manifest["reference"]["path"]
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest["reference"]["sha256"]
    actual = importlib.import_module(reader).read_manifest(root / "manifest.json")
    assert equality_gate(actual, frozen)["status"] == "pass"
