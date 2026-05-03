"""Correctness tests for ConsequenceReplayBuffer circular buffer implementation.

Run with:
    conda run -n counterfactual python tests/test_consequence_buffer.py
"""

import sys
import numpy as np
from counterfactual_rl.agents.shared.consequence_buffers import ConsequenceReplayBuffer


def make_transition(i):
    return {
        's': np.array([float(i)]),
        'a': np.array([0]),
        'r': float(i),
        "s'": np.array([float(i + 1)]),
        'done': False,
    }


passed = 0
failed = 0


def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ''))
        failed += 1


# ---------------------------------------------------------------------------
print("\n[Size tracking]")

buf = ConsequenceReplayBuffer(capacity=10)
check("empty buffer has len 0", len(buf) == 0)
check("can_sample(1) is False when empty", not buf.can_sample(1))

buf = ConsequenceReplayBuffer(capacity=5)
for i in range(5):
    buf.add(make_transition(i))
check("size grows to capacity", len(buf) == 5)

for i in range(15):
    buf.add(make_transition(i))
check("size does not exceed capacity", len(buf) == 5)

# ---------------------------------------------------------------------------
print("\n[FIFO eviction]")

buf = ConsequenceReplayBuffer(capacity=3)
for i in range(3):
    buf.add(make_transition(i))  # T0(r=0), T1(r=1), T2(r=2)
buf.add(make_transition(3))       # T3(r=3) overwrites T0(r=0)
check("size stays at capacity after eviction", len(buf) == 3)
rewards = {buf.buffer[i]['r'] for i in range(3)}
check("oldest entry evicted (r=0 gone, r=3 present)", rewards == {1.0, 2.0, 3.0}, f"got {rewards}")

buf = ConsequenceReplayBuffer(capacity=4)
for i in range(4 * 3):  # wrap 3 times
    buf.add(make_transition(i))
check("size correct after multiple wraparounds", len(buf) == 4)

# ---------------------------------------------------------------------------
print("\n[Sampling]")

buf = ConsequenceReplayBuffer(capacity=20)
buf.add(make_transition(0))
check("can_sample(1) True after 1 add", buf.can_sample(1))
check("can_sample(2) False after 1 add", not buf.can_sample(2))

for i in range(1, 10):
    buf.add(make_transition(i))
transitions, indices, weights = buf.sample(5)
check("sample returns correct count", len(transitions) == 5 and len(indices) == 5 and len(weights) == 5)
check("sample indices in valid range", np.all(indices >= 0) and np.all(indices < len(buf)))
check("IS weights are positive", np.all(weights > 0))

_, indices = buf.sample_uniform(10)
check("sample_uniform no duplicate indices", len(indices) == len(np.unique(indices)))

_, indices = buf.sample_uniform(100)  # request more than available
check("sample_uniform clamps to buffer size", len(indices) == len(buf))

# ---------------------------------------------------------------------------
print("\n[Priority computation]")

buf = ConsequenceReplayBuffer(capacity=20)
for i in range(10):
    buf.add(make_transition(i))

probs = buf._compute_priorities()
check("priorities length matches buffer size", len(probs) == len(buf))
check("priorities sum to 1.0", abs(probs.sum() - 1.0) < 1e-6, f"sum={probs.sum()}")

probs1 = buf._compute_priorities()
probs2 = buf._compute_priorities()
check("priorities are cached (same object)", probs1 is probs2)

buf.update_priorities(np.array([0]), np.array([99.0]))
probs3 = buf._compute_priorities()
check("cache invalidated after update_priorities", probs3 is not probs2)

buf2 = ConsequenceReplayBuffer(capacity=10, mu=0.0)  # pure TD priority
for i in range(5):
    buf2.add(make_transition(i))
buf2.update_priorities(np.array([2]), np.array([100.0]))
probs = buf2._compute_priorities()
check("high TD error slot gets highest priority", probs[2] == probs.max(), f"probs={probs}")

buf3 = ConsequenceReplayBuffer(capacity=10, priority_mixing='multiplicative')
for i in range(8):
    buf3.add(make_transition(i))
probs = buf3._compute_priorities()
check("multiplicative mixing sums to 1.0", abs(probs.sum() - 1.0) < 1e-6)

# ---------------------------------------------------------------------------
print("\n[Update methods]")

buf = ConsequenceReplayBuffer(capacity=10)
for i in range(5):
    buf.add(make_transition(i))

buf.update_consequence_scores(np.array([1, 3]), np.array([0.8, 0.4]))
check("update_consequence_scores writes correct values",
      abs(buf.consequence_scores[1] - 0.8) < 1e-9 and abs(buf.consequence_scores[3] - 0.4) < 1e-9)

buf.update_priorities(np.array([0, 4]), np.array([-2.5, 1.0]))
check("update_priorities stores abs(td_error)",
      abs(buf.td_magnitudes[0] - 2.5) < 1e-9 and abs(buf.td_magnitudes[4] - 1.0) < 1e-9)

# ---------------------------------------------------------------------------
print("\n[JAX state storage]")

buf = ConsequenceReplayBuffer(capacity=5)
fake_state = {'x': np.array([1.0, 2.0])}
buf.add(make_transition(0), jax_state=fake_state)
check("jax_state stored and retrieved", buf.get_jax_state(0) is fake_state)

buf2 = ConsequenceReplayBuffer(capacity=5)
buf2.add(make_transition(0))
check("jax_state is None when not provided", buf2.get_jax_state(0) is None)

buf3 = ConsequenceReplayBuffer(capacity=3)
states = [{'id': i} for i in range(3)]
for i in range(3):
    buf3.add(make_transition(i), jax_state=states[i])
new_state = {'id': 99}
buf3.add(make_transition(99), jax_state=new_state)
check("jax_state correct after wraparound", buf3.get_jax_state(0) is new_state)

# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
