"""History is lossless evidence; only complete compatible rows are runnable."""
import base64
import copy
import json
from pathlib import Path
import zipfile

import pytest

from experiments.history import (
    export_manifest, import_manifest, main, read_metrics_config, sha256,
    validate_manifest, export_legacy_bundle,
)
from experiments.recipes import load_recipe

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "configs/historical"


@pytest.mark.parametrize("path", sorted(HISTORY.glob("*.json")), ids=lambda p: p.stem)
def test_archived_manifest_round_trip_without_worktrees(path, tmp_path):
    record = json.loads(path.read_text())
    raw = validate_manifest(record)
    target = tmp_path / "original"
    export_manifest(record, target)
    assert target.read_bytes() == raw
    reimported = import_manifest(target, encoding=record["source"]["encoding"])
    assert reimported["rows"] == record["rows"]
    assert reimported["source"]["sha256"] == record["source"]["sha256"]
    assert record["resolution"]["defaults_applied"] is False


def test_import_preserves_unknowns_and_format(tmp_path):
    source = tmp_path / "old.json"
    raw = b'{"arm/3": {"seed":3, "future_field":null}}\r\n'
    source.write_bytes(raw)
    record = import_manifest(source)
    assert record["rows"] == [{"id": "arm/3", "raw": {"seed": 3, "future_field": None}}]
    assert validate_manifest(record) == raw
    assert "fixed_eval_seed" not in record["rows"][0]["raw"]


@pytest.mark.parametrize("raw", [
    b'{"x":{},"x":{}}', b'{"rows":[{"task":0},{"task":0}]}',
    b'{"x":{"seed":NaN}}', b'[]', b'{"rows":"bad"}', b'{"x":1}',
])
def test_reject_ambiguous_manifest(tmp_path, raw):
    source = tmp_path / "bad.json"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        import_manifest(source)


@pytest.mark.parametrize("field", ["hash", "rows", "schema", "bool-as-number"])
def test_tampering_is_rejected(field):
    record = json.loads((HISTORY / "coverage.json").read_text())
    if field == "hash":
        record["source"]["sha256"] = "0" * 64
    elif field == "rows":
        record["rows"][0]["raw"]["seed"] = 999
    elif field == "bool-as-number":
        record["rows"][0]["raw"]["seed"] = False
    else:
        record["schema_version"] = 900
    with pytest.raises(ValueError):
        validate_manifest(record)


def test_cli_round_trip_and_no_overwrite(tmp_path):
    source, archive, restored = [tmp_path / name for name in ("input", "archive", "restored")]
    raw = base64.b64encode(b'{"seed":4}') + b"\n"
    source.write_bytes(raw)
    main(["import", str(source), str(archive), "--encoding", "base64-lines"])
    main(["verify", str(archive)])
    main(["export", str(archive), str(restored)])
    assert restored.read_bytes() == raw
    with pytest.raises(FileExistsError):
        main(["export", str(archive), str(restored)])
    with pytest.raises(FileExistsError):
        main(["import", str(source), str(archive), "--encoding", "base64-lines"])


@pytest.mark.parametrize("name,sparse,reward", [
    ("sparse_fixed_baselines", True, 1), ("dense_fixed_baselines", False, 4),
])
def test_new_path_settings_equal_preserved_headers(name, sparse, reward, tmp_path):
    path = HISTORY / "runnable" / f"{name}.json"
    manifest = json.loads(path.read_text())
    assert len(manifest["rows"]) == 10
    arms = set()
    for raw_row in manifest["rows"]:
        row = load_recipe(path, raw_row["task"])
        source = row["historical_source"]
        header = base64.b64decode(source["config_header_base64"], validate=True)
        assert sha256(header) == source["config_header_sha256"]
        saved = tmp_path / "metrics.log"
        saved.write_bytes(header)
        original = read_metrics_config(saved)
        config = copy.deepcopy(row["config"])
        for field, value in source["adapter_fields"].items():
            assert field not in original
            assert config.pop(field) == value
        assert config == original  # No changed scientific fields or inferred defaults.
        assert config["sparse_reward"] is sparse
        assert config["goal_rew"] == reward
        assert config["fixed_eval_seed"] == 20260923
        assert config["eval_episodes"] == 1000
        assert row["collector"] == "old" and row["decay"] is False
        arms.add((config["algorithm"], config["seed"]))
    assert arms == {(arm, seed) for arm in ("dqn", "dqn-uniform") for seed in range(5)}


@pytest.mark.parametrize("text", [
    "# Old log without config\n", "# seed: 1\n# seed: 2\n",
    "# alpha: __import__('os').system('false')\n",
])
def test_metrics_parser_rejects_inference_and_executable_values(tmp_path, text):
    path = tmp_path / "metrics.log"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_metrics_config(path)


def test_ledger_keeps_limits_and_distinct_routes():
    ledger = json.loads((ROOT / "studies/jaxnav-migration/history-ledger.json").read_text())
    entries = ledger["entries"]
    assert len({e["id"] for e in entries}) == len(entries)
    assert {e["code_status"] for e in entries} == {"blocked", "pinned_legacy", "new_path"}
    for entry in entries:
        path = ROOT / entry["recipe"]
        assert path.is_file()
        assert entry["blocked_fields"]  # No full historical reproducibility claim.
        if entry["code_status"] == "new_path":
            for row in json.loads(path.read_text())["rows"]:
                load_recipe(path, row["task"])
        elif entry["code_status"] == "pinned_legacy":
            assert not entry["source_bundle"]["mismatches"]
            assert entry["source_bundle"]["verified_files"] > 0
            assert entry["saved_checks"]
        else:
            assert "launch_source" in entry["blocked_fields"]


def legacy_fixture(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.json").write_text('{"rows":[{"task":0,"config":{"seed":0}}]}')
    (root / "versions.json").write_text('{"python":"test-only"}')
    (root / "trainer.py").write_bytes(b"raise RuntimeError('must never execute')\n")
    expected = {p.name: sha256(p.read_bytes()) for p in root.iterdir()}
    (root / "hashes.json").write_text(json.dumps(expected))
    entry = {"id": "temporal_replay_v2", "code_status": "pinned_legacy",
             "source_bundle": {"root": str(root), "hash_manifest": {
                 "sha256": sha256((root / "hashes.json").read_bytes())}}}
    return root, entry, import_manifest(root / "manifest.json")


def test_bundle_export_keeps_sealed_bytes_and_excludes_run_data(tmp_path):
    root, entry, record = legacy_fixture(tmp_path)
    (root / "unrelated-run.log").write_text("not source")
    target = tmp_path / "recovery.zip"
    report = export_legacy_bundle(entry, record, target)
    assert report["sealed_files"] == 3
    with zipfile.ZipFile(target) as archive:
        assert "unrelated-run.log" not in archive.namelist()
        for name, digest in report["member_sha256"].items():
            assert archive.read(name) == (root / name).read_bytes()
            assert sha256(archive.read(name)) == digest
    with pytest.raises(FileExistsError):
        export_legacy_bundle(entry, record, target)


@pytest.mark.parametrize("fault", ["source", "hashes", "recipe", "cap", "escape", "symlink"])
def test_bundle_export_fails_before_creating_output(tmp_path, fault):
    root, entry, record = legacy_fixture(tmp_path)
    kwargs = {}
    if fault == "source":
        (root / "trainer.py").write_text("changed")
    elif fault == "hashes":
        (root / "hashes.json").write_text("{}")
    elif fault == "recipe":
        record["rows"][0]["raw"]["config"]["seed"] = 3
    elif fault == "cap":
        kwargs["max_bytes"] = 1
    elif fault == "symlink":
        (root / "trainer.py").unlink()
        (root / "trainer.py").symlink_to(root / "versions.json")
    else:
        hashes = root / "hashes.json"
        expected = json.loads(hashes.read_text())
        expected["../escape"] = "0" * 64
        hashes.write_text(json.dumps(expected))
        entry["source_bundle"]["hash_manifest"]["sha256"] = sha256(hashes.read_bytes())
    target = tmp_path / "bad.zip"
    with pytest.raises(ValueError):
        export_legacy_bundle(entry, record, target, **kwargs)
    assert not target.exists()
