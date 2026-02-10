# DQN Training Architecture for SMAX

## System Overview

```mermaid
flowchart TB
    subgraph ENV["SMAX Environment (JaxMARL)"]
        direction LR
        E_RESET["env.reset(key)"]
        E_STEP["env.step(key, state, action_dict)"]
        E_OBS["Per-Agent Observations<br/>obs = {agent_0: [...], agent_1: [...], agent_2: [...]}"]
        E_REW["Team Reward<br/>damage_dealt / team_size + win_bonus"]
    end

    subgraph PREPROCESS["Observation Processing"]
        GS["get_global_state()<br/>Concatenate all agent obs<br/>[obs_0 | obs_1 | obs_2]"]
        AM["get_action_masks()<br/>env.get_avail_actions(state)<br/>shape: (3, num_actions)"]
    end

    subgraph AGENT["DQN Agent"]
        direction TB
        EG["Epsilon-Greedy<br/>ε = 0.1"]

        subgraph QNET["CentralizedQNetwork (PyTorch)"]
            direction TB
            INPUT["Global State Input<br/>(obs_dim,)"]
            FC1["FC1: Linear + ReLU<br/>(obs_dim → 256)"]
            FC2["FC2: Linear + ReLU<br/>(256 → 256)"]
            FC3["FC3: Linear + ReLU<br/>(256 → 256)"]

            subgraph HEADS["Per-Agent Q-Value Heads"]
                H0["Head 0<br/>Q(s, a₀)"]
                H1["Head 1<br/>Q(s, a₁)"]
                H2["Head 2<br/>Q(s, a₂)"]
            end

            INPUT --> FC1 --> FC2 --> FC3
            FC3 --> H0
            FC3 --> H1
            FC3 --> H2
        end

        subgraph MASK["Action Masking"]
            M0["Mask invalid → -inf<br/>argmax → action₀"]
            M1["Mask invalid → -inf<br/>argmax → action₁"]
            M2["Mask invalid → -inf<br/>argmax → action₂"]
        end

        EG -->|"90% greedy"| QNET
        EG -->|"10% random<br/>valid actions"| JOINT_ACT
        H0 --> M0
        H1 --> M1
        H2 --> M2
        M0 --> JOINT_ACT["Joint Action<br/>[a₀, a₁, a₂]"]
        M1 --> JOINT_ACT
        M2 --> JOINT_ACT
    end

    ENV --> PREPROCESS
    GS --> EG
    AM --> MASK
    JOINT_ACT -->|"action_dict = {<br/>agent_0: a₀,<br/>agent_1: a₁,<br/>agent_2: a₂}"| E_STEP
```

## Training Loop

```mermaid
flowchart TB
    START(["Episode Start"]) --> RESET["env.reset(key)<br/>Get initial obs, state"]
    RESET --> GET_STATE["global_state = concat(obs)<br/>action_masks = env.get_avail_actions()"]
    GET_STATE --> STEP_LOOP

    subgraph STEP_LOOP["Step Loop (while not done)"]
        direction TB
        SELECT["select_action(state, masks)<br/>ε-greedy with action masking"]
        ENV_STEP["env.step(key, state, action_dict)<br/>→ next_obs, rewards, dones"]
        STORE["Store transition in PER buffer<br/>{s, a, r, s', done, masks, next_masks}"]
        CHECK_Q{"total_steps<br/>% 4 == 0?"}
        UPDATE["_update()<br/>Sample & learn from buffer"]
        CHECK_T{"total_steps<br/>% 500 == 0?"}
        TARGET["Hard copy<br/>Q → Q_target"]
        ADVANCE["state ← next_state<br/>masks ← next_masks<br/>return += reward"]

        SELECT --> ENV_STEP --> STORE
        STORE --> CHECK_Q
        CHECK_Q -->|Yes| UPDATE --> CHECK_T
        CHECK_Q -->|No| CHECK_T
        CHECK_T -->|Yes| TARGET --> ADVANCE
        CHECK_T -->|No| ADVANCE
        ADVANCE -->|"not done"| SELECT
    end

    ADVANCE -->|"done"| LOG["Log episode return & length"]
    LOG --> SAVE_CHECK{"episode<br/>% 500 == 0?"}
    SAVE_CHECK -->|Yes| SAVE["Save checkpoint"]
    SAVE_CHECK -->|No| NEXT
    SAVE --> NEXT["Next episode"]
    NEXT -->|"episodes remaining"| RESET
    NEXT -->|"training complete"| DONE(["Save final model"])
```

## Q-Network Update (_update)

```mermaid
flowchart TB
    subgraph SAMPLE["1. Sample from PER Buffer"]
        PER["PER Buffer<br/>(capacity: 100,000)"]
        BATCH["Sample batch of 32<br/>P(transition) ∝ (|TD_error| + ε)^β"]
        WEIGHTS["Importance Sampling Weights<br/>w_i = P(uniform) / P(priority)"]
        PER --> BATCH --> WEIGHTS
    end

    subgraph COMPUTE["2. Compute Targets & Q-Values"]
        direction TB
        LOOP["For each transition in batch:"]

        subgraph TARGET_CALC["Target Calculation"]
            NOT_DONE{"done?"}
            TERMINAL["target = r"]
            NON_TERMINAL["Q_target(s') → mask invalid<br/>target = r + γ · Σ max_a' Q_target(s', a')<br/>(sum over all agents)"]
            NOT_DONE -->|Yes| TERMINAL
            NOT_DONE -->|No| NON_TERMINAL
        end

        subgraph Q_CALC["Q-Value Calculation"]
            Q_FORWARD["Q(s) → Q-values for all agents"]
            Q_TAKEN["Q_taken = Σ Q(s, a_i)<br/>(sum of chosen action Q-values)"]
            Q_FORWARD --> Q_TAKEN
        end

        TD["TD_error = target - Q_taken"]
        LOOP --> TARGET_CALC
        LOOP --> Q_CALC
        TARGET_CALC --> TD
        Q_CALC --> TD
    end

    subgraph LEARN["3. Update Network"]
        PRIORITIES["Update PER priorities<br/>priority_i = (|td_error_i| + 0.01)^0.25"]
        LOSS["Weighted MSE Loss<br/>L = mean(w_i · (Q_taken - target)²)"]
        BACKPROP["Backprop + Adam (lr=0.0005)<br/>Gradient clipping (max=100)"]
        PRIORITIES --> LOSS --> BACKPROP
    end

    SAMPLE --> COMPUTE --> LEARN
```

## Two-Network Architecture

```mermaid
flowchart LR
    subgraph ONLINE["Q Network (Online)"]
        O_DESC["Updated every 4 steps<br/>via gradient descent<br/>Used for action selection"]
    end

    subgraph TARGET["Q_target Network (Frozen)"]
        T_DESC["Frozen between copies<br/>Provides stable TD targets<br/>Prevents moving target problem"]
    end

    ONLINE -->|"Hard copy every<br/>500 steps"| TARGET
```

## Reward Structure (SMAX 3m)

```mermaid
flowchart LR
    subgraph REWARD["Per-Step Team Reward"]
        DMG["Damage Reward<br/>|enemy_health_decrease| / enemy_team_size"]
        WIN["Win Bonus: +1.0<br/>All enemies dead & ≥1 ally alive"]
        TOTAL["Total = damage + win_bonus"]
        DMG --> TOTAL
        WIN --> TOTAL
    end

    subgraph RANGE["Expected Returns (3m)"]
        BEST["Win: ~6.0<br/>(all damage + bonus)"]
        MID["Partial: ~2-3<br/>(damage only, lost)"]
        WORST["Quick Loss: ~0-1<br/>(minimal damage)"]
    end

    TOTAL --> RANGE
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.1 | Exploration rate |
| `alpha` | 0.0005 | Learning rate (Adam) |
| `hidden_dim` | 256 | MLP hidden layer size |
| `M` | 100,000 | Replay buffer capacity |
| `B` | 32 | Batch size |
| `C` | 500 | Target network update frequency (steps) |
| `n_steps_for_Q_update` | 4 | Steps between Q-network updates |
| `PER eps` | 0.01 | Priority smoothing constant |
| `PER beta` | 0.25 | Priority exponent |
| `n_episodes` | 2,000+ | Training episodes |

## File Structure

```
dqn/
├── config.py          # DEFAULT_CONFIG hyperparameters
├── policies.py        # CentralizedQNetwork (shared body + per-agent heads)
├── buffers.py         # PrioritizedReplayBuffer (TD-error priorities)
├── dqn.py             # DQN agent class (.learn(), .save(), .load())
├── utils.py           # SMAX helpers (env creation, state/mask/reward utils)
├── train.py           # CLI entry point (argparse → DQN.learn())
├── visualize.py       # Record trained agent gameplay as GIF
├── train_smax_dqn.sh  # SLURM sbatch for training on Rosie
├── visualize_smax_dqn.sh  # SLURM sbatch for visualization
├── models/            # Saved checkpoints
├── videos/            # Recorded gameplay GIFs
└── logs/              # SLURM job output logs
```
