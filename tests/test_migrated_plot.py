import json
from pathlib import Path
import pytest
from analysis.plot_migrated_runs import read_runs


def make_run(root, name, seed, rows, **changes):
    path = root/name
    path.mkdir()
    cfg = dict(seed=seed, algorithm="consequence-dqn", priority_mixing="additive",
        mu=.5, mu_c=1, mu_delta=1, normalise_priorities="rank", collector_variant="fixed",
        decay_enabled=True, fixed_eval_seed=23, map_size=[8,8])
    cfg.update(changes)
    (path/"config.json").write_text(json.dumps(cfg))
    (path/"attempt.json").write_text(json.dumps(dict(eval_bank_hash="test-bank",
        jaxmarl_source_hashes={"env.py":"test-source"})))
    (path/"eval.jsonl").write_text("\n".join(json.dumps(dict(episode=e,win_rate=w)) for e,w in rows))
    return path


def test_equal_seed_weight_and_no_filled_tail(tmp_path):
    a = make_run(tmp_path,"a",0,[(1,.2),(2,.4),(11,.9)])
    b = make_run(tmp_path,"b",1,[(1,.8)])
    curve = read_runs([a,b],10)["curves"][0]
    assert curve["bin_lower_edges"] == [0]
    assert curve["n_seeds"] == 2
    assert curve["mean"] == pytest.approx([.55])
    assert curve["median"] == pytest.approx([.55])


def test_duplicate_seed_rejected(tmp_path):
    a = make_run(tmp_path,"a",0,[(1,.2)])
    b = make_run(tmp_path,"b",0,[(1,.4)])
    with pytest.raises(ValueError,match="Duplicate"):
        read_runs([a,b])


def test_incompatible_maps_rejected(tmp_path):
    a = make_run(tmp_path,"a",0,[(1,.2)])
    b = make_run(tmp_path,"b",1,[(1,.4)],map_size=[11,11])
    with pytest.raises(ValueError,match="Incompatible"):
        read_runs([a,b])


def test_restarted_log_rejected(tmp_path):
    a = make_run(tmp_path,"a",0,[(2,.2),(1,.4)])
    with pytest.raises(ValueError,match="Invalid"):
        read_runs([a])


def test_same_seed_different_bank_rejected(tmp_path):
    a = make_run(tmp_path,"a",0,[(1,.2)])
    b = make_run(tmp_path,"b",1,[(1,.4)])
    metadata = json.loads((b/"attempt.json").read_text())
    metadata["eval_bank_hash"] = "different-bank"
    (b/"attempt.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="bank or simulator differs"):
        read_runs([a,b])


def test_frozenlake_uses_step_clock_and_generic_simulator_identity(tmp_path):
    a = make_run(tmp_path, "fl", 0, [(0, .25), (0, .75)])
    (a/"attempt.json").write_text(json.dumps(dict(environment="frozenlake",
        eval_axis="env_steps", eval_bank_hash="fl-bank",
        simulator_source_hashes={"frozenlake.py": "fl-source"})))
    (a/"eval.jsonl").write_text('\n'.join(json.dumps(dict(episode=0, env_steps=s, win_rate=w))
        for s, w in [(8, .25), (16, .75)]))
    data = read_runs([a], 8)
    assert data["x_field"] == "env_steps"
    assert data["curves"][0]["mean"] == [.25, .75]
    assert data["curves"][0]["bin_lower_edges"] == [8, 16]
