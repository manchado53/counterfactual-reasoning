import copy
import json
from pathlib import Path

import pytest

from training.learners.doorkey.build import validate_config
from training.learners.doorkey.config import JAXNAV_PARITY_FIELDS
from training.recipes import load_recipe


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/doorkey/jaxnav-decay-transfer/training.json"
JAXNAV = ROOT / "experiments/jaxnav/ranked-td-decay/training.json"


def _jaxnav_winner():
    rows = json.loads(JAXNAV.read_text())["rows"]
    return next(row for row in rows if row["task"] == 33)["config"]


def test_recipe_is_two_maps_three_arms_five_matched_seeds():
    rows = json.loads(RECIPE.read_text())["rows"]
    assert len(rows) == 30
    assert {row["task"] for row in rows} == set(range(30))
    assert {(row["layout"], row["arm"], row["seed"]) for row in rows} == {
        (layout, arm, seed)
        for layout in ("6x6", "6x6_lava")
        for arm in ("uniform", "per", "cce_decay")
        for seed in range(5)
    }
    for row in rows:
        assert load_recipe(RECIPE, row["task"]) == row


def test_every_shared_learning_field_matches_jaxnav_winner():
    reference = _jaxnav_winner()
    rows = json.loads(RECIPE.read_text())["rows"]
    for row in rows:
        cfg = row["config"]
        for field in JAXNAV_PARITY_FIELDS:
            assert cfg[field] == reference[field], (row["task"], field)


def test_only_environment_and_arm_fields_change_between_matched_rows():
    rows = json.loads(RECIPE.read_text())["rows"]
    clean = next(row["config"] for row in rows
                 if row["layout"] == "6x6" and row["arm"] == "cce_decay" and row["seed"] == 0)
    lava = next(row["config"] for row in rows
                if row["layout"] == "6x6_lava" and row["arm"] == "cce_decay" and row["seed"] == 0)
    changed = {key for key in clean if clean[key] != lava[key]}
    assert changed == {"layout_name"}


@pytest.mark.parametrize("field", sorted(JAXNAV_PARITY_FIELDS))
def test_validation_rejects_jaxnav_field_drift(field):
    row = load_recipe(RECIPE, 10)
    bad = copy.deepcopy(row["config"])
    value = bad[field]
    if isinstance(value, bool):
        bad[field] = not value
    elif isinstance(value, (int, float)):
        bad[field] = value + 1
    elif isinstance(value, dict):
        bad[field] = dict(value, eps=value["eps"] + 1)
    else:
        bad[field] = f"changed-{value}"
    with pytest.raises(ValueError):
        validate_config(bad)


def test_decay_is_active_only_for_cce_arm():
    rows = json.loads(RECIPE.read_text())["rows"]
    for row in rows:
        expected = row["arm"] == "cce_decay"
        assert row["decay"] is expected
        assert row["config"]["decay_enabled"] is expected
        assert row["config"]["cce_enabled"] is expected
