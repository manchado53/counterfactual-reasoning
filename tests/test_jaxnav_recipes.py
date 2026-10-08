import json
from pathlib import Path
import pytest
from experiments.recipes import load_recipe
from experiments.smoke import JAXNAV_OVERRIDES

RECIPE = Path(__file__).resolve().parents[1] / "configs/experiment/jaxnav_decay_20261007.json"


def test_all_45_resolved_recipes():
    manifest = json.loads(RECIPE.read_text())
    assert len(manifest["rows"]) == 45
    for row in manifest["rows"]:
        assert load_recipe(RECIPE, row["task"]) == row
    assert {r["arm"] for r in manifest["rows"]} == {"mul", "add", "only"}
    assert {r["seed"] for r in manifest["rows"]} == set(range(5))


@pytest.mark.parametrize("kind", ["unknown", "missing", "collector", "seed", "duplicate", "schedule"])
def test_reject_recipe_drift(tmp_path, kind):
    manifest = json.loads(RECIPE.read_text())
    row = manifest["rows"][0]
    if kind == "unknown":
        row["config"]["typo"] = 1
    elif kind == "missing":
        del row["config"]["gamma"]
    elif kind == "collector":
        row["config"]["collector_variant"] = "fixed"
    elif kind == "seed":
        row["config"]["seed"] = 999
    elif kind == "duplicate":
        manifest["rows"].append(row)
    else:
        row["config"]["decay_end"] = row["config"]["decay_start"]
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_recipe(path,row["task"])


def test_smoke_is_explicit_and_preserves_scoring_horizon():
    row = load_recipe(RECIPE,1)
    cfg = dict(row["config"], **JAXNAV_OVERRIDES)
    assert row["config"]["n_episodes"] == 200000
    assert cfg["n_episodes"] == 6
    assert cfg["cf_horizon"] == row["config"]["cf_horizon"] == 60


def test_frozenlake_recipe_uses_its_own_defaults(tmp_path):
    from experiments.learners.frozenlake.build import smoke_config
    cfg = smoke_config()
    row = dict(task=0, seed=cfg["seed"], collector="fixed", config=cfg)
    path = tmp_path / "frozenlake.json"
    path.write_text(json.dumps(dict(schema_version=1, environment="frozenlake", rows=[row])))
    assert load_recipe(path, 0) == row
