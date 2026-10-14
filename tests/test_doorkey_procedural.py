import numpy as np


def test_symbolic_view_matches_minigrid_3_seed_zero_fixture():
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
    expected = np.array(
        [
            [[0, 0, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0], [2, 5, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [5, 4, 0], [1, 0, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [2, 5, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        ],
        dtype=np.uint8,
    )

    observation = env.observe(state)
    np.testing.assert_array_equal(np.asarray(observation["image"]), expected)
    assert int(observation["direction"]) == 1


def test_reset_and_step_follow_official_doorkey_rules():
    import jax

    from training.envs.doorkey_procedural import (
        FORWARD,
        PICKUP,
        TOGGLE,
        DOOR_OPEN,
        DoorKeyState,
        ProceduralDoorKey,
    )

    env = ProceduralDoorKey(size=12)
    observation, reset = env.reset(jax.random.PRNGKey(9))
    assert observation["image"].shape == (7, 7, 3)
    assert 2 <= int(reset.split_x) <= 9
    assert 1 <= int(reset.door_y) <= 9
    assert 1 <= int(reset.x) < int(reset.split_x)
    assert 1 <= int(reset.key_x) < int(reset.split_x)
    assert (int(reset.x), int(reset.y)) != (int(reset.key_x), int(reset.key_y))

    by_key = DoorKeyState(5, 7, 0, False, 6, 7, 2, 8, 3, 0)
    _, picked, reward, terminated, truncated, _ = env.step(None, by_key, PICKUP)
    assert bool(picked.carrying)
    assert (int(picked.key_x), int(picked.key_y)) == (-1, -1)
    assert float(reward) == 0 and not bool(terminated) and not bool(truncated)

    by_door = picked._replace(x=7, y=3, direction=0)
    _, opened, _, _, _, _ = env.step(None, by_door, TOGGLE)
    assert int(opened.door_state) == DOOR_OPEN

    by_goal = opened._replace(x=9, y=10, direction=0, step_count=0)
    _, reached, reward, terminated, truncated, _ = env.step(None, by_goal, FORWARD)
    assert (int(reached.x), int(reached.y)) == (10, 10)
    assert np.isclose(float(reward), 1 - 0.9 / 1440)
    assert bool(terminated) and not bool(truncated)


def test_research_recipe_uses_paired_12x12_protocol():
    from training.learners.doorkey.config import procedural_benchmark_config

    rows = [procedural_benchmark_config(arm, seed)
            for arm in ("uniform", "per", "cce_decay") for seed in range(5)]
    assert len(rows) == 15
    assert {(row["arm"], row["seed"]) for row in rows} == {
        (arm, seed) for arm in ("uniform", "per", "cce_decay") for seed in range(5)
    }
    for row in rows:
        assert row["protocol"] == "procedural-doorkey-12x12"
        assert row["size"] == 12
        assert row["max_episode_steps"] == 1440
        assert row["observation"] == "symbolic-7x7-partial"
        assert row["total_env_steps"] == 1_000_000
        assert row["eval_episodes"] == 1000
        assert row["fixed_eval_seed"] == 20260923
    cce = next(row for row in rows if row["arm"] == "cce_decay")
    assert cce["score_on_insert"] is True
    assert cce["normalise_priorities"] == "rank"
    assert (cce["decay_start"], cce["decay_end"]) == (500_000, 750_000)


def test_procedural_learner_smoke_has_exact_cce_coverage_and_fixed_eval_bank():
    import pickle

    from training.learners.doorkey.config import procedural_benchmark_config
    from training.learners.doorkey.procedural_runner import ProceduralAgent

    hashes = set()
    for arm in ("uniform", "per", "cce_decay"):
        config = procedural_benchmark_config(arm, 0)
        config.update(
            n_envs=4,
            collect_steps=4,
            buffer_capacity=32,
            batch_size=4,
            total_env_steps=16,
            eval_interval_steps=16,
            eval_episodes=2,
            max_episode_steps=24,
            eval_max_steps=24,
            epsilon_decay_steps=4,
            cf_horizon=2,
            cf_score_chunk=8,
            decay_start=2,
            decay_end=6,
            _smoke_profile=True,
        )
        agent = ProceduralAgent(config)
        before = pickle.dumps(agent.training_rng_state())
        evaluation = agent.evaluate()
        assert pickle.dumps(agent.training_rng_state()) == before
        hashes.add(evaluation["eval_bank_hash"])
        row = agent.train_block()
        assert row["env_steps"] == 16
        assert np.isfinite(row["loss"])
        if arm == "cce_decay":
            assert row["scored_frac"] == 1.0
    assert len(hashes) == 1
