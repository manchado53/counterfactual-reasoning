# FrozenLake — Gymnasium Implementation Reference

Research into the Gymnasium implementation of FrozenLake-v1, to be used as the
spec for a JAX reimplementation. All details verified against the Gymnasium
source at `gymnasium/envs/toy_text/frozen_lake.py`.

---

## State Space

States are single integers using **row-major indexing**:

```
state = row * ncols + col
```

- 4×4 map: 16 states (0–15)
- 8×8 map: 64 states (0–63)
- Any N×M map: N*M states

Position [row, col] maps to state `row * ncols + col`. Row 0 is the top,
col 0 is the left.

---

## Action Space

4 discrete actions:

| Integer | Name  | Effect                          |
|---------|-------|---------------------------------|
| 0       | LEFT  | col -= 1                        |
| 1       | DOWN  | row += 1                        |
| 2       | RIGHT | col += 1                        |
| 3       | UP    | row -= 1                        |

**Boundary behaviour:** clamped, not wrapped.

```
LEFT:  new_col = max(0, col - 1)
DOWN:  new_row = min(nrows - 1, row + 1)
RIGHT: new_col = min(ncols - 1, col + 1)
UP:    new_row = max(0, row - 1)
```

---

## Transition Dynamics

### Slippery = True (required for our use case)

Each action produces **three equally probable outcomes**:

```
outcomes = [(a - 1) % 4,  a,  (a + 1) % 4]    # each with prob 1/3
```

The intended direction plus its two perpendicular neighbours. Examples:

| Chosen action | Outcomes (each p=1/3)        |
|---------------|------------------------------|
| LEFT (0)      | UP (3), LEFT (0), DOWN (1)   |
| DOWN (1)      | LEFT (0), DOWN (1), RIGHT (2)|
| RIGHT (2)     | DOWN (1), RIGHT (2), UP (3)  |
| UP (3)        | RIGHT (2), UP (3), LEFT (0)  |

Each of the three directions applies the same boundary clamping as above.

### Slippery = False

Single deterministic outcome, probability = 1.0 in the intended direction.

### Terminal states

Hole (H) and Goal (G) states loop to themselves regardless of action:

```
P[terminal_state][any_action] = [(1.0, terminal_state, reward, terminated=True)]
```

---

## Reward Structure

Default `reward_schedule = (goal_reward=1, hole_reward=0, frozen_reward=0)`:

| Transition lands on | Reward | Terminated |
|---------------------|--------|------------|
| G (goal)            | 1.0    | True       |
| H (hole)            | 0.0    | True       |
| F (frozen)          | 0.0    | False      |
| S (start, if revisited) | 0.0 | False   |

Reward is given when **transitioning TO** a tile, not when leaving it.

---

## Map Format

Maps are lists of strings. Each character is one cell:

| Char | Meaning       |
|------|---------------|
| S    | Start (safe)  |
| F    | Frozen (safe) |
| H    | Hole (terminal, reward=0) |
| G    | Goal (terminal, reward=1) |

### Default 4×4 map

```python
["SFFF",
 "FHFH",
 "FFFH",
 "HFFG"]
```

```
State  0=S   1=F   2=F   3=F
State  4=F   5=H   6=F   7=H
State  8=F   9=F  10=F  11=H
State 12=H  13=F  14=F  15=G
```

Holes at states: 5, 7, 11, 12. Goal at state 15.

### Default 8×8 map

```python
["SFFFFFFF",
 "FFFFFFFF",
 "FFFHFFFF",
 "FFFFFHFF",
 "FFFHFFFF",
 "FHHFFFHF",
 "FHFFHFHF",
 "FFFHFFFG"]
```

Start at state 0 (position [0,0]). Goal at state 63 (position [7,7]).
Holes at positions: [2,3], [3,5], [4,3], [5,1], [5,2], [5,6], [6,1], [6,4],
[6,6], [7,3].

---

## env.P Structure

`env.P[state][action]` is a list of `(probability, next_state, reward, terminated)` tuples.

Example — state 1, action LEFT (0), slippery=True:

```python
[
  (0.333, 0,  0.0, False),   # slides UP   → state 0 (S, safe)
  (0.333, 1,  0.0, False),   # slides LEFT → state 1 (stays, F)
  (0.333, 5,  0.0, True ),   # slides DOWN → state 5 (H, hole)
]
```

### Construction algorithm

```python
for s in range(nstates):
    row, col = s // ncols, s % ncols
    tile = desc[row][col]          # b'S', b'F', b'H', or b'G'

    for a in range(4):
        outcomes = []

        if tile in b'GH':          # terminal state: loop in place
            outcomes = [(1.0, s, reward_for(tile), True)]
        elif is_slippery:
            for b in [(a-1)%4, a, (a+1)%4]:
                new_row, new_col = move(row, col, b)   # with clamping
                new_s = new_row * ncols + new_col
                new_tile = desc[new_row][new_col]
                outcomes.append((1/3, new_s, reward_for(new_tile),
                                 new_tile in b'GH'))
        else:
            new_row, new_col = move(row, col, a)
            new_s = new_row * ncols + new_col
            new_tile = desc[new_row][new_col]
            outcomes = [(1.0, new_s, reward_for(new_tile),
                         new_tile in b'GH')]

        P[s][a] = outcomes
```

---

## Observation

The observation returned to the agent is a **single integer** — the state index
directly. No one-hot encoding.

```python
obs = int(state)    # integer in [0, nstates)
```

---

## Reset

- Start position determined by all 'S' tiles in the map (default maps have one S at [0,0])
- Sampled categorically from the initial state distribution (uniform over S tiles)
- Default maps: deterministic reset to state 0
- Returns `(state_int, {"prob": 1.0})`

---

## Episode Termination

| Condition              | Flag          |
|------------------------|---------------|
| Agent reaches H (hole) | terminated=True |
| Agent reaches G (goal) | terminated=True |
| Steps exceed max_steps | truncated=True  |

Default max_steps:
- 4×4: 100 steps
- 8×8: 200 steps

---

## Step Function (Gymnasium pseudocode)

```python
def step(self, action):
    transitions = self.P[self.s][action]
    i = categorical_sample([t[0] for t in transitions], self.rng)
    prob, next_state, reward, terminated = transitions[i]
    self.s = next_state
    return int(next_state), reward, terminated, False, {"prob": prob}
```

---

## Key Facts for JAX Reimplementation

1. **No hidden state beyond current position** — the full env state is a single integer.

2. **Transition table is fully precomputable** — build P as a static array at
   construction time: shape `(nstates, 4, 3, 4)` for slippery:
   - dim 0: state
   - dim 1: action
   - dim 2: outcome index (always 3 for slippery)
   - dim 3: (next_state, reward, terminated, probability) or split into 4 separate arrays

3. **Slippery step needs one random sample** — `jax.random.choice` over the 3
   outcomes weighted by [1/3, 1/3, 1/3] (uniform, so `jax.random.randint(0, 3)`
   is sufficient). Non-slippery step is deterministic.

4. **Boundary clamping is the main logic in transition precomputation** — once P
   is built, `step()` is just a table lookup + random sample.

5. **Terminal states loop to themselves** — simplifies `lax.scan` usage since
   done states don't need special branching; they just accumulate zero reward.

6. **Observation = state integer** — trivial to produce, no transformation needed.

7. **Map is arbitrary** — parameterize by map string, not by grid size, to support
   4×4, 8×8, and custom maps without code changes.
