# Next simple environment for CCE replay

## Decision

**Run MinAtar Breakout next. Run bsuite DeepSea second.**

Breakout is the best bridge between our current tasks and a broader result:

```text
FrozenLake / JaxNav / DoorKey       MinAtar Breakout
navigation and board states   ->   moving objects and delayed control
small discrete actions        ->   3 unique discrete actions
saved-state rollouts          ->   explicit JAX EnvState rollouts
```

MinAtar was made to keep Atari-like delayed behaviour while cutting pixel and compute cost. Its paper tests a small DQN with replay. Breakout has a `10 x 10 x 4` binary state, three unique actions, `+1` per brick, and delayed effects between paddle movement and the next bounce. The Pgx port exposes the full game state as a JAX pytree and accepts explicit random keys, which is a strong fit for batched CCE branches. Pgx is already used by this repo for Chess and Connect Four; the installed `pgx==2.6.0` successfully creates `pgx.make("minatar-breakout")`. Sources: [MinAtar paper](https://arxiv.org/abs/1903.03176), [official MinAtar DQN](https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py), [official Breakout source](https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py), [Pgx repository](https://github.com/sotetsuk/pgx), and [Pgx MinAtar Breakout source](https://github.com/sotetsuk/pgx-minatar/blob/main/breakout.py).

## Scorecard

Scale: `++` strong, `+` useful, `0` mixed, `-` weak.

| Environment | Sparse or delayed reward | Branch from saved state | DQN / PER fit | Actions | Compute | Difficulty scale | New paper value | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **MinAtar Breakout** | `+` | `++` with Pgx | `++` | 3 unique; 6 full | `+` | `0` | `++` | **Next** |
| **bsuite DeepSea** | `++` | `++` | `+` | 2 | `++` | `++` (`N=10..50`) | `0` | **Follow-up** |
| MountainCar-v0 | `+` | `++` | `+` | 3 | `++` | `0` | `+` | Good fallback |
| bsuite Catch | `0` | `++` | `++` | 3 | `++` | `-` | `0` | Smoke test only |
| Taxi-v4 | `0` | `++` | `+` | 6 | `++` | `-` | `-` | Too close to old grid tasks |
| CliffWalking-v1 | `-` | `++` | `+` | 4 | `++` | `-` | `-` | Too dense and too close to old grid tasks |

### Why each score looks this way

- **Breakout:** reward arrives when the moving ball hits a brick, while a paddle action can matter many steps later. The official code supplies a DQN replay baseline. The original game has six actions; only no-op, left, and right are unique in Breakout. Pgx uses the unique three by default and stores the full hidden game state needed for exact branches. Breakout itself has no built-in difficulty ramp, but the same interface extends to four other MinAtar games. [Paper details](https://arxiv.org/html/1903.03176), [MinAtar repository](https://github.com/kenjyoung/MinAtar), [Pgx code](https://github.com/sotetsuk/pgx-minatar/blob/main/breakout.py).
- **DeepSea:** the agent gets the large `+1` only by reaching the bottom-right end of an `N x N` chain. It has two actions, fixed `N`-step episodes, an explicit state, and an official sweep from `N=10` through `N=50`. This is the cleanest sparse-reward test. But it is still a grid task, and replay cannot help before any useful transition is discovered. We must report first-discovery time separately from learning after discovery. [DeepSea source](https://github.com/google-deepmind/bsuite/blob/master/bsuite/environments/deep_sea.py), [official sweep](https://github.com/google-deepmind/bsuite/blob/master/bsuite/experiments/deep_sea/sweep.py), [bsuite paper](https://arxiv.org/abs/1908.03568).
- **MountainCar:** deterministic dynamics, two-number state `(position, velocity)`, three actions, `-1` each step, and a 200-step limit make exact branches cheap. It adds a control task, but it has no standard difficulty sweep and all non-terminal rewards match, so short CCE horizons may give weak separation before a branch reaches the goal. [Gymnasium docs](https://gymnasium.farama.org/environments/classic_control/mountain_car/), [official source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py).
- **Catch:** each episode ends after the ball crosses the board and always returns `+1` or `-1`. It is cheap and DQN-friendly, but reward is not rare and the task is mostly a learner smoke test. The official sweep is 10,000 episodes across seeds, with no size sweep. [Catch source](https://github.com/google-deepmind/bsuite/blob/master/bsuite/environments/catch.py), [official sweep](https://github.com/google-deepmind/bsuite/blob/master/bsuite/experiments/catch/sweep.py).
- **Taxi:** exact tabular branches are easy, but the task gives `-1` each step, `-10` for illegal pickup/drop-off, and `+20` on delivery. Its fixed 5x5 layout and 500 encoded states add little beyond our navigation results. [Gymnasium docs](https://gymnasium.farama.org/environments/toy_text/taxi/), [official source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/taxi.py).
- **CliffWalking:** exact branches are easy, but every step gives `-1` and the cliff gives `-100`. The fixed 4x12 map is a dense-risk grid benchmark, not a strong new domain for the paper. [Gymnasium docs](https://gymnasium.farama.org/environments/toy_text/cliff_walking/), [official source](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py).

## Minimal next experiment

Use **Pgx `minatar-breakout` with its default three unique actions**.

```text
same seeds, eval bank, and DQN settings
                 |
        +--------+--------+
        |        |        |
     Uniform    PER      CCE
```

1. Run Uniform, proportional PER, and the current best CCE replay rule.
2. Hold network, optimizer, exploration, buffer, update ratio, seeds, and evaluation states fixed across arms.
3. Use 5 training seeds for the pilot. Train for 1 million frames. Extend unchanged runs to the official 5-million-frame DQN budget only if learning is still moving or arms separate.
4. Start with CCE horizon `20`. This spans a meaningful part of a ball trip without the cost of a long JaxNav rollout. Use one common PRNG key across action branches from the same saved state.
5. Evaluate every 50,000 frames on 200 fixed reset keys. Report return AUC, final return, probability of improvement, wall time, and replay effective sample size.
6. Add one **six-action sensitivity check** after the pilot. The MinAtar paper used all six actions, but three are unique in Breakout. This check tells us whether duplicate no-op actions change CCE weighting.

### Pass rule

Move to a full seed count only if CCE beats both baselines on return AUC **and** the gain is not explained by more environment steps, more updates, or a different evaluation bank.

## Follow-up: bsuite DeepSea

Run `N = 10, 14, 20` with the official randomized action mapping and 10,000 episodes. Compare the same three replay arms. `N=20` is an intentional hard case where all replay methods may fail to discover reward. Log two curves:

```text
first useful reward found  ->  exploration
learning after first find  ->  replay credit assignment
```

That split matters. CCE can replay a rare useful event after it exists. It cannot prioritize an event that the agent has never found. DeepSea therefore gives a clean boundary test for what CCE does and does not solve.

## Source set

Only primary sources were used:

- [Behaviour Suite paper](https://arxiv.org/abs/1908.03568) and [official bsuite repository](https://github.com/google-deepmind/bsuite)
- [MinAtar paper](https://arxiv.org/abs/1903.03176) and [official MinAtar repository](https://github.com/kenjyoung/MinAtar)
- [Official Pgx repository](https://github.com/sotetsuk/pgx) and [Pgx paper](https://arxiv.org/abs/2303.17503)
- [Official Gymnasium documentation](https://gymnasium.farama.org/) and [official Gymnasium source](https://github.com/Farama-Foundation/Gymnasium)
