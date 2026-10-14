# MiniGrid DoorKey benchmarks

Research note, 2026-10-13. Sources are limited to official MiniGrid material and original papers or their first-party repositories.

## Short answer

The standard DoorKey benchmark is **not one fixed 6x6 board**. The official task rebuilds the map on every reset. It randomizes the dividing wall, door row, key location, agent location, and agent direction. The goal stays in the bottom-right corner. Transitions are deterministic unless a wrapper changes them.

For CCE, the best next benchmark is:

> **`MiniGrid-DoorKey-8x8-v0`, official procedural layouts, default 7x7 partial observation, default sparse reward and 640-step limit.**

Use 5 paired training seeds. Give every arm the same fixed bank of held-out evaluation seeds. Report success and official return against environment steps. Then use 16x16 as the stretch test.

Why 8x8 first: a replay-focused recurrent Q-learning paper compared uniform replay and PER on DoorKey 5x5, 6x6, and 8x8. Its plain R2D2 baseline solved 5x5 and 6x6 but failed on 8x8. Thus 8x8 has useful headroom without jumping straight to the much longer 16x16 task. [SeyedSalehi et al., paper and supplement](https://openreview.net/forum?id=co8W7pvSrS)

## What the official environment does

| Item | Official DoorKey rule |
|---|---|
| Sizes | 5x5, 6x6, 8x8, 16x16 |
| Layout | Procedurally regenerated at every reset |
| Randomized | Wall split, locked-door row, key position, agent position, agent direction |
| Fixed | Goal at bottom-right |
| Actions | 7 discrete actions: turn left/right, forward, pickup, drop, toggle, done |
| Default observation | Egocentric 7x7 symbolic view plus direction and mission; each visible tile is `(object, color, state)` |
| Dynamics | Deterministic by default |
| Episode limit | `10 * size^2`: 250, 360, 640, or 2,560 steps |
| Reward | Success only: `1 - 0.9 * step_count / max_steps`; failure gives 0 |

Sources: [official DoorKey docs](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/), [official DoorKey source](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/envs/doorkey.py), [base environment reset and reward source](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/minigrid_env.py), and [official observation wrappers](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/wrappers.py).

Important distinction:

```text
new episode
    |
    v
new wall + new door + new key + new start
    |
    v
same task: key -> unlock door -> goal
```

“Procedural” means the starting MDP changes between episodes. It does **not** mean actions slip at random. The official MiniGrid paper states that transition dynamics are deterministic by default and that MiniGrid is partially observed by default. [Chevalier-Boisvert et al.](https://arxiv.org/abs/2306.13831)

## What researchers use

| Source | DoorKey setup | Agent | Evaluation/reporting |
|---|---|---|---|
| MiniGrid author's starter code | 5x5; official partial observation | PPO/A2C; example trains PPO for 80k frames | Evaluation defaults to 100 episodes |
| Approximate-information-state recurrent Q-learning | 5x5, 6x6, 8x8; partial 7x7x3 observations | RQL-AIS and non-distributed R2D2, each with uniform replay and PER | 5 seeds; 4M environment steps; evaluate every 5k steps with 10 greedy episodes |
| GateL0RD | Train on procedural 8x8; test zero-shot on 16x16 | PPO with LSTM or GateL0RD memory | 10 seeds; learning curves over frames; 16x16 tests size transfer |
| DEIR exploration study | Standard procedural 16x16, plus harder observation variants | PPO, ICM, RND, NGU, NovelD, DEIR | 20 seeds; mean episodic return with standard error; standard view is 7x7 |
| ARM-FM | Fixed and procedural DoorKey at 5x5, 6x6, 8x8, 16x16 | DQN, DQN+ICM, ReAct, DQN+reward machine | 3 seeds; mean with one standard deviation; curves against environment steps |

Primary sources: [RL Starter Files](https://github.com/lcswillems/rl-starter-files), [RQL-AIS paper](https://openreview.net/forum?id=co8W7pvSrS), [RQL-AIS code](https://github.com/esalehi1996/POMDP_RL), [GateL0RD supplement](https://proceedings.neurips.cc/paper/2021/file/927b028cfa24b23a09ff20c1a7f9b398-Supplemental.pdf), [GateL0RD code](https://github.com/martius-lab/GateL0RD), [DEIR paper](https://www.ijcai.org/proceedings/2023/0477.pdf), [DEIR code](https://github.com/swan-utokyo/deir), and [ARM-FM paper](https://openreview.net/forum?id=OBpQdCWLfd).

## Common variants

### DoorKey-5x5 and 6x6

Good smoke tests. Often solved quickly. They have little room to separate strong methods.

### DoorKey-8x8

The common middle benchmark. It appears in recurrent-memory, exploration, temporal-abstraction, and replay studies. It is the strongest next choice for CCE because published uniform/PER recurrent Q-learning baselines begin to separate here. [RQL-AIS](https://openreview.net/forum?id=co8W7pvSrS) [GateL0RD](https://proceedings.neurips.cc/paper/2021/file/927b028cfa24b23a09ff20c1a7f9b398-Supplemental.pdf)

### DoorKey-16x16

The common hard or transfer setting. Exploration papers use it as a main sparse-reward task, while other work tests zero-shot transfer from 8x8. It is defensible, but a plain DQN may receive too few successes to compare replay rules at a small budget. [DEIR](https://www.ijcai.org/proceedings/2023/0477.pdf) [GateL0RD](https://proceedings.neurips.cc/paper/2021/file/927b028cfa24b23a09ff20c1a7f9b398-Supplemental.pdf)

### Fixed-layout DoorKey

Useful as a controlled diagnostic. It should be labeled **fixed-layout**, not presented as the standard DoorKey benchmark. ARM-FM reports fixed and procedural layouts separately, which is a good precedent. [ARM-FM](https://openreview.net/forum?id=OBpQdCWLfd)

### KeyCorridor and harder BabyAI key-door tasks

These add rooms, hidden keys, blocked doors, or longer subgoal chains. They are good follow-ups after CCE works on official DoorKey-8x8. They are not replacements for first validating the standard DoorKey task. [official KeyCorridor docs](https://minigrid.farama.org/environments/minigrid/KeyCorridorEnv/) [DEIR](https://www.ijcai.org/proceedings/2023/0477.pdf)

## Recommended CCE protocol

### Main run

| Choice | Recommendation |
|---|---|
| Environment | `MiniGrid-DoorKey-8x8-v0` |
| Training layouts | Fresh official procedural layout on each reset |
| Observation | Default 7x7 symbolic partial view; use a recurrent Q-network |
| Reward | Unchanged official sparse reward |
| Horizon | Unchanged 640-step cap |
| Arms | Uniform, standard PER, CCE under the same learner and compute budget |
| Seeds | At least 5 paired training seeds |
| Evaluation | Same held-out map-seed bank for every arm and checkpoint; greedy actions |
| Metrics | Success rate, official episodic return, steps on successful episodes, environment steps to threshold |
| Uncertainty | Show all seed traces and a seed-level interval; do not treat evaluation episodes as independent training seeds |

The fixed evaluation bank is stricter than many older protocols. It removes map-sampling noise from vertical arm comparisons while still testing many official procedural layouts.

### Follow-up

Run `MiniGrid-DoorKey-16x16-v0` only after 8x8 shows that all arms receive enough successful experience. If all plain replay arms stay at zero, that is mainly an exploration-floor test, not a clean replay-priority comparison.

### What not to use as the main claim

Our recent run used a custom tabular 6x6 layout, fixed start, one fixed key/door map, a 50-step evaluation cap, `+1/0` reward, and optional lava. That is a useful CCE mechanism test. It is not the standard MiniGrid DoorKey benchmark. The implementation itself documents these changes from Farama MiniGrid. [our custom DoorKey source at the tested commit](https://github.com/manchado53/counterfactual-reasoning/blob/39bbcb531141d1b3d199eab90c55ef8a5cf980a3/training/envs/doorkey.py)

```text
recent test                          next defensible benchmark
-----------                          -------------------------
one fixed 6x6 map              ->    official random 8x8 maps
full tabular state              ->    standard 7x7 partial view
50-step cap                     ->    official 640-step cap
+1 at goal                      ->    official time-scaled reward
mechanism check                 ->    benchmark claim
```
