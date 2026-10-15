import json

import numpy as np


def test_full_observation_matches_minigrid_fully_obs_encoding():
    from training.envs.doorkey_procedural import DoorKeyState, ProceduralDoorKey

    env = ProceduralDoorKey(size=12)
    state = DoorKeyState(
        x=5,
        y=6,
        direction=1,
        carrying=False,
        key_x=6,
        key_y=7,
        door_state=2,
        split_x=8,
        door_y=3,
        step_count=0,
    )
    image = np.asarray(env.observe_full(state)["image"])

    assert image.shape == (12, 12, 3)
    np.testing.assert_array_equal(image[5, 6], [10, 0, 1])
    np.testing.assert_array_equal(image[6, 7], [5, 4, 0])
    np.testing.assert_array_equal(image[8, 3], [4, 4, 2])
    np.testing.assert_array_equal(image[10, 10], [8, 1, 0])
    np.testing.assert_array_equal(image[0, 0], [2, 5, 0])


def test_paper_recipe_pins_reported_settings_and_shared_eval_bank():
    from training.learners.doorkey.config import paper_parity_config

    rows = [
        paper_parity_config(arm, seed)
        for arm in ("uniform", "per", "cce_decay")
        for seed in range(5)
    ]
    assert len(rows) == 15
    for row in rows:
        assert row["protocol"] == "doorkey-vlmrb-paper-parity"
        assert row["observation"] == "symbolic-full-grid"
        assert row["n_actions"] == 5
        assert row["gamma"] == .95
        assert row["alpha"] == 4e-5
        assert row["batch_size"] == 128
        assert row["target_update_freq"] == 1000
        assert row["n_steps_per_update"] == 4
        assert row["learning_starts_steps"] == 500
        assert row["epsilon_decay_steps"] == 500_000
        assert row["total_env_steps"] == 1_000_000
        assert row["fixed_eval_seed"] == 20260923
        assert row["eval_episodes"] == 1000
    per = paper_parity_config("per", 0)
    assert per["PER_parameters"] == dict(eps=1e-6, beta=.7, maximum_priority=1.)
    assert per["use_importance_weights"] is True
    cce = paper_parity_config("cce_decay", 0)
    assert cce["score_on_insert"] is True
    assert cce["normalise_priorities"] == "rank"


def test_paper_network_is_spatial_and_outputs_five_actions():
    import jax
    import jax.numpy as jnp

    from training.learners.doorkey.procedural_runner import PaperGridQNetwork

    network = PaperGridQNetwork()
    image = jnp.zeros((2, 12, 12, 3), dtype=jnp.uint8)
    params = network.init(jax.random.PRNGKey(0), image)
    values = network.apply(params, image)
    assert values.shape == (2, 5)
    assert np.isfinite(np.asarray(values)).all()


def test_cce_weighted_score_accepts_the_five_action_paper_space():
    from counterfactual_rl.scoring import weighted_wasserstein_scores

    returns = np.array([[0., 1., 2., 3., 4.], [4., 3., 2., 1., 0.]])
    probabilities = np.full((2, 5), .2)
    scores = weighted_wasserstein_scores(returns, probabilities, np.array([0, 4]))
    np.testing.assert_allclose(scores, [2.5, 2.5])


def test_paper_agent_maps_fifth_policy_action_to_toggle_and_smokes_all_arms():
    from training.learners.doorkey.build import smoke_overrides, validate_config
    from training.learners.doorkey.config import paper_parity_config
    from training.learners.doorkey.procedural_runner import ProceduralAgent

    hashes = set()
    for arm in ("uniform", "per", "cce_decay"):
        config = paper_parity_config(arm, 0)
        config.update(smoke_overrides(config))
        validate_config(config)
        agent = ProceduralAgent(config)
        assert np.asarray(agent.policy_to_env_action).tolist() == [0, 1, 2, 3, 5]
        row = agent.train_block()
        assert row["env_steps"] == 16
        assert np.isfinite(row["loss"])
        evaluation = agent.evaluate()
        hashes.add(evaluation["eval_bank_hash"])
        if arm == "cce_decay":
            assert row["scored_frac"] == 1.0
    assert len(hashes) == 1


def test_generated_manifest_is_exactly_five_paired_seeds_per_arm(tmp_path, monkeypatch):
    from training import generate_doorkey_paper_parity as generator

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
