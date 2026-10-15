import json

import jax
import numpy as np


def test_16x16_full_observation_and_default_cap():
    from training.envs.doorkey_procedural import ProceduralDoorKey

    env = ProceduralDoorKey(size=16)
    observation, state = env.reset(jax.random.PRNGKey(7))
    full = np.asarray(env.observe_full(state)["image"])

    assert full.shape == (16, 16, 3)
    assert env.max_steps == 2560
    assert np.count_nonzero(full[..., 0] == 10) == 1
    assert np.count_nonzero(full[..., 0] == 5) == 1
    assert np.count_nonzero(full[..., 0] == 4) == 1
    assert np.count_nonzero(full[..., 0] == 8) == 1


def test_16x16_changes_only_task_size_and_identity():
    from training.learners.doorkey.config import (
        fullstate_16x16_config,
        paper_parity_config,
    )

    allowed = {
        "protocol",
        "size",
        "max_episode_steps",
        "eval_max_steps",
        "experiment",
    }
    for arm in ("uniform", "per", "cce_decay"):
        for seed in range(5):
            old = paper_parity_config(arm, seed)
            new = fullstate_16x16_config(arm, seed)
            changed = {key for key in old if old[key] != new[key]}
            assert changed == allowed
            assert new["size"] == 16
            assert new["max_episode_steps"] == 2560
            assert new["eval_max_steps"] == 2560


def test_16x16_manifest_has_five_paired_seeds_per_arm(tmp_path, monkeypatch):
    from training import generate_doorkey_fullstate_16x16 as generator

    output = tmp_path / "training.json"
    monkeypatch.setattr(generator, "OUTPUT", output)
    generator.main()
    manifest = json.loads(output.read_text())

    assert len(manifest["rows"]) == 15
    assert [(row["task"], row["arm"], row["seed"]) for row in manifest["rows"]] == [
        (task, arm, seed)
        for task, (arm, seed) in enumerate(
            (arm, seed)
            for arm in ("uniform", "per", "cce_decay")
            for seed in range(5)
        )
    ]
    hashes = {
        (row["config"]["fixed_eval_seed"], row["config"]["eval_episodes"])
        for row in manifest["rows"]
    }
    assert hashes == {(20260923, 1000)}
