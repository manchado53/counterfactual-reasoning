import json

import pytest

from analysis.plotting.per_style_study import analyze_study, render_study


def _write_run(root, arm, seed, *, bank="bank-a", status="completed", steps=(0, 5, 10),
               environment="frozenlake", axis="env_steps", metric="win_rate",
               log_file="eval.jsonl", step_field=None):
    path = root / f"{arm}-{seed}"
    path.mkdir()
    config = {"seed": seed, "algorithm": arm, "fixed_eval_seed": 17,
              "batch_size": 32, "reward_rule": "sparse"}
    attempt = {"resolved_config": config, "environment": environment,
               "eval_axis": axis, "eval_bank_hash": bank,
               "simulator_source_hashes": {"env.py": "source-a"},
               "status": status, "smoke": False, "seconds": 12 + seed}
    (path / "config.json").write_text(json.dumps(config))
    (path / "attempt.json").write_text(json.dumps(attempt))
    scores = {"uniform": 0.2, "proportional_per": 0.4}
    with (path / log_file).open("w") as stream:
        for step in steps:
            stream.write(json.dumps({step_field or axis: step,
                                     metric: scores[arm] + step / 100}) + "\n")
    return {"directory": str(path), "arm": arm, "seed": seed}


def _study(root, *, bank="bank-a", partial=False):
    runs = []
    for arm in ("uniform", "proportional_per"):
        for seed in (0, 1):
            runs.append(_write_run(root, arm, seed,
                                   bank="bank-b" if bank == "different" and arm == "uniform"
                                   and seed == 1 else "bank-a",
                                   status="running" if partial and arm == "proportional_per"
                                   and seed == 1 else "completed",
                                   steps=(0, 5) if partial and arm == "proportional_per"
                                   and seed == 1 else (0, 5, 10)))
    manifest = {"schema_version": 1, "study": "test-study", "cells": [{
        "id": "lake", "environment": "frozenlake", "axis": "env_steps",
        "metric": {"field": "win_rate", "label": "Goal reach rate",
                   "bounds": [0, 1], "higher_is_better": True},
        "budget": 10, "early_step": 5, "seeds": [0, 1],
        "arms": ["uniform", "proportional_per"],
        "method_config_keys": ["algorithm"], "runs": runs,
    }]}
    path = root / "study.json"
    path.write_text(json.dumps(manifest))
    return path


def test_matched_study_renders_observed_median_and_final_score(tmp_path):
    result = analyze_study(_study(tmp_path))
    cell = result["cells"][0]
    assert cell["comparison_status"] == "final_budget"
    assert cell["shared_observed_steps"] == [0, 5, 10]
    assert cell["curves"][0]["final_score"] == pytest.approx(0.3)
    assert cell["curves"][1]["early_difference_vs_uniform"] == pytest.approx(0.2)
    assert cell["curves"][0]["early_auc"] == pytest.approx(0.225)
    assert cell["curves"][1]["steps_to_uniform_final"] == 0
    assert cell["curves"][0]["median_wall_seconds"] == pytest.approx(12.5)
    assert result["cross_environment_summary"][0]["final_better_cells"] == 1
    assert result["cross_environment_summary"][0]["early_auc_better_cells"] == 1
    output = tmp_path / "report"
    render_study(result, output)
    assert {file.name for file in output.iterdir()} == {
        "study-results.json", "summary.csv", "cross-environment.csv",
        "learning-curves.png", "learning-curves.pdf",
        "learning-curves-early.png", "learning-curves-early.pdf"}


def test_mismatched_evaluation_bank_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Evaluation/simulator mismatch"):
        analyze_study(_study(tmp_path, bank="different"))


def test_running_attempt_keeps_observed_tail_and_no_final_claim(tmp_path):
    result = analyze_study(_study(tmp_path, partial=True))
    cell = result["cells"][0]
    assert cell["comparison_status"] == "interim"
    assert cell["shared_observed_steps"] == [0, 5]
    assert all(curve["final_score"] is None for curve in cell["curves"])
    assert all(curve["steps_to_uniform_final"] is None for curve in cell["curves"])
    assert result["cross_environment_summary"] == []
    assert all(curve["steps"] == [0, 5] for curve in cell["curves"])


def test_two_environments_keep_their_own_axis_and_score_name(tmp_path):
    path = _study(tmp_path)
    manifest = json.loads(path.read_text())
    second = tmp_path / "other"
    second.mkdir()
    runs = [_write_run(second, arm, seed, environment="connect_four",
                       axis="episode", metric="score", log_file="evaluations.jsonl",
                       step_field="step")
            for arm in ("uniform", "proportional_per") for seed in (0, 1)]
    cell = dict(manifest["cells"][0])
    cell.update(id="four", environment="connect_four", axis="episode",
                log_file="evaluations.jsonl", step_field="step",
                metric={"field": "score", "label": "Match score",
                        "higher_is_better": True}, runs=runs)
    manifest["cells"].append(cell)
    path.write_text(json.dumps(manifest))
    result = analyze_study(path)
    assert [cell["axis"] for cell in result["cells"]] == ["env_steps", "episode"]
    assert [cell["metric"]["field"] for cell in result["cells"]] == ["win_rate", "score"]
    render_study(result, tmp_path / "two-cell-report")
