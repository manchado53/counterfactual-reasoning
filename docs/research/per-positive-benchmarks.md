# PER-positive reproduction targets

## Bottom line

Start with **CartPole-v1**.

It is the cheapest published case I found where canonical **proportional PER** beat uniform replay under the same DDQN learner, network, optimizer, learning rate, budget, and evaluation rule. The paper used 20 matched-seed runs. Its first-party repository pins the software stack and exposes the DDQN and replay code.

Then run **Blind Cliffwalk** as a tiny mechanism test. It gives the cleanest test of both **rank-based PER** and **proportional PER**, but it is not an exact reproduction because the paper omits three stochastic-PER settings and did not release the experiment code.

Use **Asterix** only after both small gates pass. It has strong first-party Atari evidence, but a full matched study costs billions of frames.

```text
CartPole-v1       -> cheap positive control for proportional PER
Blind Cliffwalk   -> cheap mechanism check for rank and proportional PER
Asterix           -> full image-based reference
Space Invaders    -> second full image-based reference
```

## What “PER” means here

The original PER paper defines two different samplers:

- **Proportional PER:** `p_i = |delta_i| + epsilon`, then `P(i) = p_i^alpha / sum_k p_k^alpha`.
- **Rank-based PER:** `p_i = 1 / rank(i)`, then the same exponent and normalization are applied.

Importance weights are `(N * P(i))^(-beta)`, normalized so the largest weight is `1`. New items enter at the largest current priority. These details come from the [original paper and Algorithm 1](https://arxiv.org/pdf/1511.05952).

These are not interchangeable. The current repo should report each arm by its true rule.

## Ranked candidates

| Rank | Environment | Positive evidence | Exactness | Cost | Use |
|---:|---|---|---|---|---|
| 1 | CartPole-v1 | 20 matched runs; proportional PER needed about 13% fewer median steps than uniform | Good, with seed-list gap | Very low; CPU | First reproduction |
| 2 | Blind Cliffwalk | Both rank and proportional PER beat uniform by orders of magnitude across 10 starts | Learner and environment are exact; stochastic PER settings are incomplete | Tiny; CPU | Mechanism gate |
| 3 | Asterix | Original paper: both PER forms beat uniform; later DeepMind code/data: proportional PER beat uniform in all 5 paired seeds | Strong first-party code and data; old ALE/ROM drift remains | Very high; GPU | First Atari target |
| 4 | Space Invaders | Original paper: both PER forms beat uniform; later DeepMind code/data: proportional PER beat uniform in all 5 paired seeds | Same as Asterix | Very high; GPU | Atari confirmation |

The CartPole percentage is derived from the paper’s reported ReaPER improvements. If `T_R = 0.586 T_uniform` and `T_R = 0.674 T_PER`, then `T_PER = 0.869 T_uniform`: about **13.1% fewer** median steps. This is an inference from primary results, not a value printed directly in the paper.

## 1. CartPole-v1 — recommended first

### Evidence

The ICLR 2026 ReaPER paper compares uniform replay, canonical proportional PER, and ReaPER with the same DDQN model. It uses 20 matched-seed runs and stops when the score threshold is reached. The paper reports that ReaPER used 41.4% fewer steps than uniform and 32.6% fewer than PER. Those two values imply the proportional-PER arm used about 13.1% fewer median steps than uniform. See [Section 5, Figure 2, and Appendix G](https://openreview.net/pdf?id=hmQk2Iwdh0).

The authors’ [example names `CartPole-v1`](https://github.com/leonardpleiss/ReaPER/blob/ecbd0b83882e617a02224e9922301495324e7116/example.py). Their [DDQN source](https://github.com/leonardpleiss/ReaPER/blob/ecbd0b83882e617a02224e9922301495324e7116/custom_policies/custom_ddqn.py) shows Double-DQN targets, smooth-L1 loss, importance weighting, `beta: 0.4 -> 1.0`, and a `1e-6` TD floor. Their [replay source](https://github.com/leonardpleiss/ReaPER/blob/ecbd0b83882e617a02224e9922301495324e7116/custom_buffers/reaper.py) reduces to proportional PER when `omega = 0`. The [requirements file](https://github.com/leonardpleiss/ReaPER/blob/ecbd0b83882e617a02224e9922301495324e7116/requirements.txt) pins Gymnasium 0.29.1, Stable-Baselines3 dependencies, PyTorch 2.4.1, and NumPy 1.26.4.

### Paper protocol to copy

| Part | Value |
|---|---|
| Environment | `CartPole-v1` |
| Learner | DDQN |
| Network | MLP `64 -> 64`, ReLU |
| Optimizer | Adam, learning rate `2.3e-3` |
| Discount | `0.99` |
| Budget | `50,000` environment steps |
| Replay | capacity `100,000`; learning starts at `1,000` |
| Batch/update | batch `64`; train every `256` steps; `128` gradient steps |
| Target copy | every `10` steps, as reported |
| Exploration | `1.0 -> 0.04` over the first `0.16` of training |
| PER | proportional; `alpha=0.6`; `beta=0.4 -> 1.0`; TD floor `1e-6` in released code |
| Evaluation | 100 evenly spaced checks; 5 full episodes each; eval epsilon `0.001` |
| Success | reward threshold `475`; compare median steps to threshold |
| Replication count | 20 matched-seed runs per arm |

### Exactness limits

- The paper says fixed, matched seeds but does not publish the 20 seed values.
- The public `example.py` leaves the paper hyperparameters commented out. Running it unchanged uses library defaults and is **not** the paper protocol.
- The repository does not ship a full multi-run uniform-versus-PER driver or raw result files.

This is still the best first target. Use a declared seed list before launch and hold it fixed across both arms.

## 2. Blind Cliffwalk — tiny mechanism gate

### Evidence

The original PER paper’s Figure 2 shows both stochastic rank-based PER and stochastic proportional PER beating uniform replay by multiple orders of magnitude. The curves summarize 10 random initializations. See [Sections 3.1–3.3 and Appendix B.1](https://arxiv.org/pdf/1511.05952).

### Protocol the paper fully states

| Part | Value |
|---|---|
| States | `n = 2..16` |
| Actions | two; one advances, one terminates |
| Reward | `1` only after all correct actions; otherwise `0` |
| Action mapping | correct action alternates by state |
| Replay contents | exhaust all `2^n` action sequences in random order |
| Replay size | `2^(n+1) - 2` transitions; exactly one rewarded path |
| Learner | one-step Q-learning |
| Function model | one-hot state plus constant bias; action generalization blocked |
| Initialization | each weight from `Normal(0, 0.1)` |
| Discount | `1 - 1/n` |
| Step size | `1/4` |
| Stop rule | value MSE against ground truth below `1e-3` |
| Runs | 10 random initializations |

### Exactness limits

The paper does **not** disclose the stochastic Cliffwalk values for `alpha`, `beta`, or proportional `epsilon`. It also does not publish first-party Cliffwalk code. The [paper source bundle](https://export.arxiv.org/e-print/1511.05952) contains the figures and TeX, but no experiment script or hidden setting table.

Thus:

```text
good:    replay-rule logic check
bad:     claim of bit-for-bit paper reproduction
```

Pre-register the missing values. Do not describe them as recovered paper settings.

## 3. Asterix — first Atari target

### Evidence

The original paper’s human-start raw scores were:

| Arm | Score |
|---|---:|
| Uniform Double DQN | 16,837.0 |
| Rank-based PER | 22,484.5 |
| Proportional PER | 31,907.5 |

Those are single training runs per arm. Asterix was also one of the games used to tune global PER settings, so the original result is not an untouched game-level test. See [Tables 2, 3, and 7](https://arxiv.org/pdf/1511.05952).

DeepMind later released a best-effort JAX reproduction and five-seed results. Reading the first-party CSV data and averaging the last 10 evaluations gives about `11,806` for uniform Double DQN and `115,132` for proportional PER; PER wins all five paired seeds. These are calculations from the released data, not printed paper values. Sources: [DQN Zoo](https://github.com/google-deepmind/dqn_zoo), [result archive](https://github.com/google-deepmind/dqn_zoo/raw/master/results.tar.gz), [proportional-PER runner at commit `45061f4`](https://github.com/google-deepmind/dqn_zoo/blob/45061f4bbbcfa87d11bbba3cfc2305a650a41c26/dqn_zoo/prioritized/run_atari.py), and [uniform Double-DQN runner](https://github.com/google-deepmind/dqn_zoo/blob/45061f4bbbcfa87d11bbba3cfc2305a650a41c26/dqn_zoo/double_q/run_atari.py).

### DQN Zoo protocol

```text
Double DQN
84 x 84 grayscale
4 stacked frames
action repeat 4
reward clip [-1, 1]
replay capacity 1,000,000
warm-up 50,000 replay items
batch 32
learn every 16 raw frames
target copy every 120,000 raw frames
gamma 0.99
200,000,000 training frames
200 iterations
1,000,000 train frames + 500,000 eval frames per iteration
```

The proportional arm uses `alpha=0.6`, `beta=0.4 -> 1.0`, a `0.001` uniform sampling mixture, normalized importance weights, and max-seen priority for new items. DQN Zoo does not ship the paper’s rank-based Atari arm.

### Fairness trap

The published algorithm comparison changes learning rate as well as replay:

| Arm | Replay | Learning rate |
|---|---|---:|
| Published uniform | Uniform | `0.00025` |
| Published PER | Proportional | `0.00025 / 4` |

Therefore a replay-only reproduction needs a second matched pair:

| Arm | Replay | Learning rate |
|---|---|---:|
| Matched uniform | Uniform | `0.00025 / 4` |
| Matched PER | Proportional | `0.00025 / 4` |

The published pair answers “can we reproduce the named algorithms?” The matched pair answers “did replay sampling cause the gap?”

## 4. Space Invaders — Atari confirmation

### Evidence

The original paper’s human-start raw scores were:

| Arm | Score |
|---|---:|
| Uniform Double DQN | 2,628.7 |
| Rank-based PER | 3,912.1 |
| Proportional PER | 9,063.0 |

Space Invaders was not listed in the eight-game tuning set. The same caveat remains: each original score is one training run. See [Tables 2 and 7](https://arxiv.org/pdf/1511.05952).

The later DQN Zoo five-seed data also favor proportional PER in every paired seed. The mean of the last 10 evaluations is about `2,549` for uniform Double DQN and `7,569` for proportional PER. These values are calculations from the first-party [result archive](https://github.com/google-deepmind/dqn_zoo/raw/master/results.tar.gz), using the same [PER](https://github.com/google-deepmind/dqn_zoo/blob/45061f4bbbcfa87d11bbba3cfc2305a650a41c26/dqn_zoo/prioritized/run_atari.py) and [uniform](https://github.com/google-deepmind/dqn_zoo/blob/45061f4bbbcfa87d11bbba3cfc2305a650a41c26/dqn_zoo/double_q/run_atari.py) runners.

Use the same protocol and four-arm learning-rate design as Asterix.

## Compute gate

A full five-seed Atari replay-only pair costs:

```text
1 game
x 2 replay arms
x 5 seeds
x 200M training frames
= 2 billion training frames
```

The four-arm published-plus-matched design doubles that.

Do not start there. Use this gate:

```text
CartPole 20 matched seeds
    |
    +-- PER win reproduced? -- no --> audit logic before more compute
    |
    yes
    v
Blind Cliffwalk rank + proportional
    |
    +-- expected speed-up? --- no --> audit logic before Atari
    |
    yes
    v
Asterix short smoke
    |
    v
Asterix full matched study
    |
    v
Space Invaders confirmation
```

## Cases not selected

- **Pong:** the original paper and DQN Zoo show near-ties. It is a poor positive control.
- **MinAtar:** the primary LaBER study says plain PER was variable and did not consistently improve on uniform without game-specific tuning. Its own plots do not give a clean broad PER win. See the [paper](https://arxiv.org/pdf/2110.01528) and [first-party reproduction repository](https://github.com/SuReLI/laber).
- **More Atari games:** Gopher is another strong positive case, but Asterix plus Space Invaders already give one tuned and one untouched paper game. Adding Gopher before those finish adds cost, not much new proof.

## Decision

Run **CartPole-v1 uniform versus proportional PER** first.

Keep every listed setting fixed. Use 20 declared matched seeds. Compare median steps to the 475 threshold. Also save full evaluation curves and censored failures that do not reach the threshold by 50,000 steps.

That test is small enough to debug. It also avoids the Atari learning-rate confound. If it does not show the published direction, the replay path needs inspection before any expensive benchmark is credible.
