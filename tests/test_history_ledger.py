import json
import subprocess
from pathlib import Path

import pytest

from analysis.history_ledger import build_records, load_committed_records, validate


ROOT = Path(__file__).resolve().parents[1]


def test_historical_ledger_is_valid_and_current():
    report = validate(ROOT, compare_generated=False)
    assert report["experiments"] >= 20
    assert report["runs"] >= 4000
    assert report["duplicate_scheduler_keys"] == {}


def test_historical_ledger_rebuild_is_deterministic():
    audit_ref = json.loads((ROOT / "experiments/history/sources.json").read_text())["audit_ref"]
    if subprocess.run(
        ["git", "cat-file", "-e", f"{audit_ref}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        pytest.skip("immutable audit ref is not present in this checkout")
    generated, _ = build_records(ROOT)
    committed = load_committed_records(ROOT)
    assert committed == generated
    assert [row["run_id"] for row in committed] == sorted(row["run_id"] for row in committed)


def test_no_recipe_only_rows_are_called_runs():
    rows = load_committed_records(ROOT)
    assert all(row["run_kind"] != "recipe" for row in rows)


def test_each_experiment_names_its_evidence_limit():
    doc = json.loads((ROOT / "experiments/history/experiments.json").read_text())
    assert all(item.get("evidence_status") for item in doc["experiments"])
    assert all(item.get("question") for item in doc["experiments"])


def test_scheduler_snapshot_is_explicitly_not_result_proof():
    path = ROOT / "experiments/history/scheduler-snapshot-20261012.json"
    snapshot = json.loads(path.read_text())
    assert snapshot["returned_records"] == len(snapshot["records"])
    assert snapshot["query_job_ids"] >= 2000
    assert sum(snapshot["state_counts"].values()) == len(snapshot["records"])
    assert "not scientific validity" in snapshot["warning"]


def test_chat_discovered_variants_resolve_to_known_questions():
    experiments = json.loads(
        (ROOT / "experiments/history/experiments.json").read_text()
    )
    variants = json.loads((ROOT / "experiments/history/variants.json").read_text())
    known = {row["id"] for row in experiments["experiments"]}
    assert len(variants["variants"]) >= 40
    assert all(row["experiment_id"] in known for row in variants["variants"])
    assert len({row["name"] for row in variants["variants"]}) == len(variants["variants"])
