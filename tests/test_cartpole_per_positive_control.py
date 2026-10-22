import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from training.learners.cartpole.config import DEFAULT_CONFIG
from training.learners.cartpole.dqn import CartPoleDDQN, weighted_huber_td_loss
from training.learners.cartpole.replay import CartPoleReplayBuffer
from training.recipes import load_recipe


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/cartpole/per-positive-control/training.json"


def _buffer(mode):
    replay = CartPoleReplayBuffer(8, 2, mode, np.random.default_rng(7))
    for value in range(4):
        observation = np.full(2, value, dtype=np.float32)
        replay.add(observation, value % 2, float(value), observation + 1, False)
    replay.update_priorities(np.arange(4), np.array([1.0, 2.0, 4.0, 8.0]))
    return replay


def test_manifest_has_twenty_matched_seeds_and_three_named_arms():
    manifest = json.loads(RECIPE.read_text())
    assert len(manifest["rows"]) == 60
    assert {row["seed"] for row in manifest["rows"]} == set(range(20))
    for seed in range(20):
        rows = [row for row in manifest["rows"] if row["seed"] == seed]
        assert [row["arm"] for row in rows] == ["uniform", "paper_per", "repo_per"]
        for row in rows:
            assert load_recipe(RECIPE, row["task"]) == row
            expected = {**DEFAULT_CONFIG, "seed": seed, "replay": row["arm"]}
            assert row["config"] == expected


def test_uniform_probabilities_and_weights_are_flat():
    replay = _buffer("uniform")
    np.testing.assert_allclose(replay.probabilities(), np.full(4, 0.25))
    np.testing.assert_array_equal(replay.sample(32, beta=0.7).weights, np.ones(32))


def test_paper_per_uses_alpha_and_max_normalized_beta_weights():
    replay = _buffer("paper_per")
    raw = np.power(np.array([1.0, 2.0, 4.0, 8.0]) + 1e-6, 0.6)
    probabilities = raw / raw.sum()
    np.testing.assert_allclose(replay.probabilities(), probabilities)
    sample = replay.sample(128, beta=0.7)
    expected = np.power(4 * sample.probabilities, -0.7)
    expected /= np.power(4 * probabilities.min(), -0.7)
    np.testing.assert_allclose(sample.weights, expected, rtol=1e-6)
    assert sample.weights.max() <= 1.0


def test_repo_per_matches_current_repo_equations():
    replay = _buffer("repo_per")
    raw = np.power(np.array([1.0, 2.0, 4.0, 8.0]) + 0.01, 0.25)
    probabilities = raw / raw.sum()
    np.testing.assert_allclose(replay.probabilities(), probabilities)
    sample = replay.sample(128, beta=0.7)
    np.testing.assert_allclose(sample.weights, 1.0 / (4 * sample.probabilities), rtol=1e-6)


def test_new_entries_use_largest_observed_raw_td_magnitude():
    replay = _buffer("paper_per")
    replay.update_priorities(np.array([0]), np.array([19.0]))
    replay.add(np.zeros(2), 0, 0.0, np.ones(2), False)
    assert replay.max_priority == 19.0
    assert replay.td_magnitudes[4] == 19.0


def test_exploration_and_beta_schedules_reach_declared_endpoints():
    agent = object.__new__(CartPoleDDQN)
    agent.config = DEFAULT_CONFIG
    assert agent._epsilon(0) == 1.0
    assert agent._epsilon(8_000) == pytest.approx(0.04)
    assert agent._epsilon(50_000) == pytest.approx(0.04)
    assert agent._beta(0) == 0.4
    assert agent._beta(50_000) == 1.0


def test_weighted_huber_loss_uses_signed_shared_td_path():
    loss, td = weighted_huber_td_loss(
        jnp.array([[1.0, 3.0], [5.0, 2.0]]),
        jnp.array([1, 0]),
        jnp.array([5.0, 4.5]),
        jnp.array([0.5, 2.0]),
    )
    np.testing.assert_allclose(td, np.array([2.0, -0.5]))
    np.testing.assert_allclose(loss, 0.5 * (0.5 * 1.5 + 2.0 * 0.125))
