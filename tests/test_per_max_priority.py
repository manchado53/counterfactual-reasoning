"""PER insertion-priority and live-learner wiring checks."""

import ast
from pathlib import Path

import numpy as np

from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer


ROOT = Path(__file__).resolve().parents[1]


def _transition(reward=0.0):
    return {"s": 0, "a": 0, "r": reward, "ns": 1, "done": False}


def _stored_priority(buffer, raw_priority):
    return (raw_priority + buffer.eps) ** buffer.beta


def test_new_transitions_start_at_one_then_use_largest_observed_td_error():
    buffer = PrioritizedReplayBuffer(capacity=4, eps=0.01, beta=0.25)

    buffer.add(_transition())
    assert buffer.max_priority == 1.0
    assert buffer.priorities[0] == _stored_priority(buffer, 1.0)

    buffer.update_priority(0, -3.0)
    assert buffer.max_priority == 3.0

    buffer.update_priority(0, 0.2)
    assert buffer.max_priority == 3.0

    buffer.add(_transition())
    assert buffer.priorities[1] == _stored_priority(buffer, 3.0)


def test_batch_priority_updates_track_largest_raw_td_error():
    buffer = PrioritizedReplayBuffer(capacity=4)
    buffer.add_batch({
        "s": np.array([0, 1]),
        "a": np.array([0, 0]),
        "r": np.array([0.0, 0.0]),
        "ns": np.array([1, 2]),
        "done": np.array([False, False]),
    })

    buffer.update_priorities(np.array([0, 1]), np.array([0.4, -2.5]))

    assert buffer.max_priority == 2.5
    np.testing.assert_allclose(
        buffer.priorities[:2],
        [_stored_priority(buffer, 0.4), _stored_priority(buffer, 2.5)],
    )


def test_uniform_replay_behavior_is_unchanged():
    buffer = PrioritizedReplayBuffer(capacity=2, uniform=True)
    buffer.add(_transition())
    buffer.update_priority(0, 9.0)

    transitions, _, weights = buffer.sample(1)

    assert buffer.max_priority == 1.0
    assert transitions[0] == _transition()
    np.testing.assert_array_equal(weights, np.ones(1))


def test_current_learners_import_the_canonical_per_buffer():
    paths = [
        ROOT / "training/learners/jaxnav/dqn.py",
        ROOT / "training/learners/frozenlake/dqn.py",
        ROOT / "training/learners/doorkey/train.py",
        ROOT / "training/learners/connect_four/transfer.py",
    ]

    for path in paths:
        tree = ast.parse(path.read_text())
        imports = {
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert (
            "counterfactual_rl.replay.buffers",
            "PrioritizedReplayBuffer",
        ) in imports, path
