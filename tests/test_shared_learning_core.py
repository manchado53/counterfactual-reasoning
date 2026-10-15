import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from counterfactual_rl.learning.td import q_learning_targets, weighted_td_loss
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight
from counterfactual_rl.scoring.metrics import compute_consequence_metric
from counterfactual_rl.scoring.weighted_wasserstein import weighted_wasserstein_scores
from training.registry import CURRENT_ENVIRONMENTS


ROOT = Path(__file__).resolve().parents[1]


def test_double_dqn_targets_use_online_choice_and_target_value():
    online_next = jnp.array([[1.0, 4.0, 3.0], [9.0, 8.0, 7.0]])
    target_next = jnp.array([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])

    actual = q_learning_targets(
        rewards=jnp.array([1.0, -1.0]),
        terminated=jnp.array([False, True]),
        online_next_q=online_next,
        target_next_q=target_next,
        gamma=0.5,
        double_dqn=True,
    )

    np.testing.assert_array_equal(actual, np.array([11.0, -1.0]))


def test_q_targets_respect_legal_action_mask():
    actual = q_learning_targets(
        rewards=jnp.array([1.0]),
        terminated=jnp.array([False]),
        online_next_q=jnp.array([[1.0, 100.0, 3.0]]),
        target_next_q=jnp.array([[10.0, 200.0, 30.0]]),
        gamma=0.5,
        double_dqn=True,
        legal_actions=jnp.array([[True, False, True]]),
    )

    np.testing.assert_array_equal(actual, np.array([16.0]))


def test_q_targets_make_invalid_live_action_masks_loud():
    actual = q_learning_targets(
        rewards=jnp.array([1.0]),
        terminated=jnp.array([False]),
        online_next_q=jnp.array([[1.0, 2.0]]),
        target_next_q=jnp.array([[10.0, 20.0]]),
        gamma=0.5,
        double_dqn=True,
        legal_actions=jnp.array([[False, False]]),
    )

    assert np.isnan(np.asarray(actual)[0])


def test_standard_dqn_targets_choose_with_target_network():
    actual = q_learning_targets(
        rewards=jnp.array([1.0]),
        terminated=jnp.array([False]),
        online_next_q=jnp.array([[100.0, 1.0, 2.0]]),
        target_next_q=jnp.array([[10.0, 20.0, 30.0]]),
        gamma=0.5,
        double_dqn=False,
    )

    np.testing.assert_array_equal(actual, np.array([16.0]))


def test_weighted_td_loss_returns_one_shared_td_definition():
    loss, td_errors, chosen_q = weighted_td_loss(
        q_values=jnp.array([[2.0, 3.0], [5.0, 7.0]]),
        actions=jnp.array([1, 0]),
        targets=jnp.array([5.0, 1.0]),
        weights=jnp.array([0.5, 2.0]),
    )

    np.testing.assert_array_equal(chosen_q, np.array([3.0, 5.0]))
    np.testing.assert_array_equal(td_errors, np.array([2.0, -4.0]))
    np.testing.assert_array_equal(loss, np.array(17.0))


def _current_source_files(environment):
    root = ROOT / "training/learners" / environment
    return [
        path
        for path in root.rglob("*.py")
        if not {"legacy", "_historical", "_source_shared"} & set(path.parts)
        and not path.name.endswith("_legacy.py")
    ]


def test_every_current_environment_crosses_the_shared_learning_seams():
    required_imports = {
        "counterfactual_rl.learning.td": {"q_learning_targets", "weighted_td_loss"},
        "counterfactual_rl.replay.buffers": {"PrioritizedReplayBuffer"},
        "counterfactual_rl.replay.consequence_buffers": {"ConsequenceReplayBuffer"},
        "counterfactual_rl.scoring.weighted_wasserstein": {"weighted_wasserstein_scores"},
    }
    for environment in CURRENT_ENVIRONMENTS:
        imports = {module: set() for module in required_imports}
        calls = set()
        for path in _current_source_files(environment):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in imports:
                    imports[node.module].update(alias.name for alias in node.names)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
        for module, names in required_imports.items():
            assert names <= imports[module], (environment, module)
            if module in {
                "counterfactual_rl.learning.td",
                "counterfactual_rl.scoring.weighted_wasserstein",
            }:
                assert names <= calls, (environment, names - calls)


def test_current_environments_do_not_copy_learning_math_or_import_frozen_code():
    banned_definitions = {
        "targets", "ddqn_targets", "td_targets", "q_learning_targets",
        "weighted_scores", "consequence", "weighted_wasserstein_scores",
    }
    banned_classes = {"PrioritizedReplayBuffer", "ConsequenceReplayBuffer", "DecayBuffer"}
    for environment in CURRENT_ENVIRONMENTS:
        for path in _current_source_files(environment):
            tree = ast.parse(path.read_text())
            definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
            frozen_imports = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and any(part in node.module for part in ("legacy", "_historical", "_source_shared"))
            }
            direct_wasserstein = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "wasserstein_distance"
            }
            assert not definitions & banned_definitions, path
            assert not classes & banned_classes, path
            historical_adapters = {
                "build.py", "temporal.py", "temporal_fixed.py", "carry_pilot.py"
            }
            if path.name not in historical_adapters:
                assert not frozen_imports, (path, frozen_imports)
            assert not direct_wasserstein, path


def test_weighted_wasserstein_uses_action_weights_and_excludes_taken_action():
    scores = weighted_wasserstein_scores(
        returns=np.array([[0.0, 2.0, 5.0]]),
        action_probabilities=np.array([[0.2, 0.3, 0.5]]),
        taken_actions=np.array([0]),
    )

    np.testing.assert_array_equal(scores, np.array([3.875]))


def test_weighted_wasserstein_excludes_illegal_actions():
    scores = weighted_wasserstein_scores(
        returns=np.array([[0.0, 2.0, 100.0]]),
        action_probabilities=np.array([[0.2, 0.3, 0.5]]),
        taken_actions=np.array([0]),
        legal_actions=np.array([[True, True, False]]),
    )

    np.testing.assert_array_equal(scores, np.array([2.0]))


@pytest.mark.parametrize("taken", [np.array([-1]), np.array([3])])
def test_weighted_wasserstein_rejects_out_of_range_taken_actions(taken):
    with pytest.raises(ValueError, match="taken action"):
        weighted_wasserstein_scores(
            returns=np.array([[0.0, 1.0, 2.0]]),
            action_probabilities=np.array([[0.2, 0.3, 0.5]]),
            taken_actions=taken,
        )


def test_weighted_wasserstein_rejects_illegal_taken_action():
    with pytest.raises(ValueError, match="taken action"):
        weighted_wasserstein_scores(
            returns=np.array([[0.0, 1.0, 2.0]]),
            action_probabilities=np.array([[0.2, 0.3, 0.5]]),
            taken_actions=np.array([1]),
            legal_actions=np.array([[True, False, True]]),
        )


def test_weighted_wasserstein_supports_many_rollouts_per_action():
    scores = weighted_wasserstein_scores(
        returns=np.array([[[0.0, 2.0], [1.0, 3.0], [0.0, 4.0]]]),
        action_probabilities=np.array([[0.2, 0.3, 0.5]]),
        taken_actions=np.array([0]),
    )

    np.testing.assert_array_equal(scores, np.array([1.0]))


def test_unweighted_frozenlake_score_keeps_its_historical_max_fallback():
    score = compute_consequence_metric(
        action=(0,),
        return_distributions={
            (0,): np.array([0.0]),
            (1,): np.array([2.0]),
            (2,): np.array([5.0]),
        },
        metric="wasserstein",
        aggregation="weighted_mean",
    )

    assert score == 5.0
    source = ast.parse(
        (ROOT / "training/learners/frozenlake/consequence_dqn.py").read_text()
    )
    calls = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_consequence_metric"
    ]
    assert len(calls) == 1
    assert "action_probs" not in {keyword.arg for keyword in calls[0].keywords}




def test_shared_td_matches_previous_dqn_and_double_dqn_formulas():
    rng = np.random.default_rng(1907)
    batch, actions = 257, 15
    online = rng.normal(size=(batch, actions)).astype(np.float32)
    target = rng.normal(size=(batch, actions)).astype(np.float32)
    rewards = rng.normal(size=batch).astype(np.float32)
    done = rng.integers(0, 2, size=batch).astype(np.float32)
    for double_dqn in (False, True):
        selection = online if double_dqn else target
        selected = selection.argmax(axis=-1)
        expected = rewards + np.float32(.99) * target[np.arange(batch), selected] * (
            np.float32(1) - done
        )
        actual = q_learning_targets(
            jnp.asarray(rewards), jnp.asarray(done), jnp.asarray(online),
            jnp.asarray(target), .99, double_dqn=double_dqn,
        )
        np.testing.assert_array_equal(actual, expected)


def test_shared_masked_td_matches_previous_connect_four_formula():
    online = jnp.array([[1.0, 100.0, 3.0], [8.0, 7.0, 6.0]])
    target = jnp.array([[10.0, 200.0, 30.0], [4.0, 5.0, 6.0]])
    mask = jnp.array([[True, False, True], [False, True, True]])
    rewards = jnp.array([1.0, -1.0])
    done = jnp.array([False, True])
    old_action = jnp.argmax(jnp.where(mask, online, -jnp.inf), axis=-1)
    old_value = jnp.take_along_axis(target, old_action[:, None], -1)[:, 0]
    expected = rewards + .99 * jnp.where(done, 0.0, old_value)
    actual = q_learning_targets(
        rewards, done, online, target, .99,
        double_dqn=True, legal_actions=mask,
    )
    np.testing.assert_array_equal(actual, expected)


def test_shared_one_rollout_score_matches_previous_ordered_formula():
    rng = np.random.default_rng(2081)
    batch, actions = 4000, 15
    returns = rng.normal(size=(batch, actions, 1)).astype(np.float32)
    probabilities = rng.random((batch, actions)).astype(np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    taken = rng.integers(0, actions, size=batch)
    values = returns[:, :, 0].astype(np.float64)
    reference = values[np.arange(batch), taken]
    numerator = np.zeros(batch)
    denominator = np.zeros(batch)
    for action in range(actions):
        alternative = taken != action
        weight = probabilities[:, action].astype(np.float64)
        numerator = np.where(
            alternative,
            numerator + weight * np.abs(reference - values[:, action]),
            numerator,
        )
        denominator = np.where(alternative, denominator + weight, denominator)
    expected = numerator / denominator
    actual = weighted_wasserstein_scores(returns, probabilities, taken)
    np.testing.assert_array_equal(actual, expected)


def test_shared_many_rollout_score_matches_previous_frozenlake_formula():
    rng = np.random.default_rng(8112)
    states, actions, rollouts = 17, 4, 20
    returns = rng.normal(size=(states, actions, rollouts)).astype(np.float32)
    probabilities = rng.random((states, actions))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = np.zeros((states, actions))
    for state in range(states):
        distances = np.zeros((actions, actions))
        for action in range(actions):
            for other in range(action):
                distances[action, other] = distances[other, action] = wasserstein_distance(
                    returns[state, action], returns[state, other]
                )
        for action in range(actions):
            weights = probabilities[state].copy()
            weights[action] = 0.0
            expected[state, action] = np.dot(weights, distances[action]) / weights.sum()
    actual = np.stack(
        [
            weighted_wasserstein_scores(
                returns, probabilities, np.full(states, action, dtype=np.int64)
            )
            for action in range(actions)
        ],
        axis=1,
    )
    np.testing.assert_array_equal(actual, expected)


def test_shared_decay_boundaries_and_probability_blend():
    assert blend_weight(99, start=100, end=150) == 0.0
    assert blend_weight(125, start=100, end=150) == 0.5
    assert blend_weight(151, start=100, end=150) == 1.0
    buffer = DecayBuffer(
        capacity=4, eps=.01, beta=.25, max_priority=1.0,
        mu=.5, priority_mixing="additive", mu_c=1.0, mu_delta=1.0,
        normalise_priorities="rank",
    )
    slots = buffer.add_batch(
        {"s": np.arange(4), "a": np.zeros(4), "r": np.zeros(4),
         "s'": np.arange(1, 5), "done": np.zeros(4)},
        jax_states=np.arange(4),
    )
    buffer.update_consequence_scores(slots, np.array([.1, .4, .2, .3]))
    buffer.update_priorities(slots, np.array([4.0, 1.0, 3.0, 2.0]))
    buffer.blend = 0.0
    original = buffer._compute_priorities().copy()
    ranked_td = buffer.ranked_td().copy()
    buffer.blend = 0.5
    np.testing.assert_array_equal(
        buffer._compute_priorities(), .5 * original + .5 * ranked_td
    )
    buffer.blend = 1.0
    np.testing.assert_array_equal(buffer._compute_priorities(), ranked_td)
