"""PER insertion-priority and live-learner wiring checks."""

import numpy as np

from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer
from counterfactual_rl.replay.consequence_buffers import ConsequenceReplayBuffer
from training.learners.connect_four._source_shared.consequence_buffers import (
    ConsequenceReplayBuffer as ConnectFourFrozenConsequenceReplayBuffer,
)
from training.learners.doorkey._source_shared.consequence_buffers import (
    ConsequenceReplayBuffer as DoorKeyFrozenConsequenceReplayBuffer,
)

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


def _exercise_cce_buffer(buffer_class, priority_mixing, mu):
    buffer = buffer_class(
        capacity=8,
        eps=0.01,
        beta=0.25,
        max_priority=1.0,
        mu=mu,
        priority_mixing=priority_mixing,
        mu_c=1.0,
        mu_delta=1.0,
        normalise_priorities="rank",
    )
    transitions = {
        "s": np.arange(18, dtype=np.float32).reshape(6, 3),
        "a": np.arange(6) % 3,
        "r": np.array([0.0, 1.0, 0.0, -1.0, 0.5, 0.0]),
        "s_next": np.arange(18, 36, dtype=np.float32).reshape(6, 3),
        "done": np.array([False, True, False, True, False, False]),
    }
    slots = buffer.add_batch(transitions, jax_states=np.arange(12).reshape(6, 2))
    buffer.update_consequence_scores(
        slots,
        np.array([0.7, 0.0, 1.2, 0.4, 0.9, 0.2]),
        q_update_count=17,
    )
    buffer.update_priorities(slots, np.array([-0.1, 2.0, -0.5, 0.25, 1.5, -0.75]))
    priorities = buffer._compute_priorities().copy()
    diagnostics = buffer.priority_diagnostics(now=23)
    np.random.seed(9127)
    _, indices, weights = buffer.sample(4)
    return buffer, slots, priorities, diagnostics, indices, weights


def test_shared_cce_buffer_matches_frozen_transfer_copies():
    arm_rules = (("multiplicative", 0.5), ("additive", 0.5), ("additive", 1.0))
    for priority_mixing, mu in arm_rules:
        shared = _exercise_cce_buffer(ConsequenceReplayBuffer, priority_mixing, mu)

        for frozen_class in (
            DoorKeyFrozenConsequenceReplayBuffer,
            ConnectFourFrozenConsequenceReplayBuffer,
        ):
            frozen = _exercise_cce_buffer(frozen_class, priority_mixing, mu)
            shared_buffer, shared_slots, shared_p, shared_diag, shared_i, shared_w = shared
            frozen_buffer, frozen_slots, frozen_p, frozen_diag, frozen_i, frozen_w = frozen

            np.testing.assert_array_equal(frozen_slots, shared_slots)
            np.testing.assert_array_equal(
                frozen_buffer.consequence_scores, shared_buffer.consequence_scores
            )
            np.testing.assert_array_equal(frozen_buffer.td_magnitudes, shared_buffer.td_magnitudes)
            np.testing.assert_array_equal(frozen_buffer.scored_at, shared_buffer.scored_at)
            np.testing.assert_array_equal(frozen_p, shared_p)
            assert frozen_diag == shared_diag
            np.testing.assert_array_equal(frozen_i, shared_i)
            np.testing.assert_array_equal(frozen_w, shared_w)
