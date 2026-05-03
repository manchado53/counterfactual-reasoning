# Gardner Chess (5×5) — Environment Reference

Gardner Chess is a 5×5 chess variant suggested by Martin Gardner in 1969. All standard pieces
fit on the first row of a 5-wide board, so the game preserves the full strategic richness of
chess — piece asymmetry, material trades, checkmate — in a much smaller search space.

We use the `pgx` JAX-native implementation (NeurIPS 2023):

```python
import pgx
env = pgx.make("gardner_chess")
```

---

## The Board

```
  a   b   c   d   e
┌───┬───┬───┬───┬───┐
│ ♜ │ ♞ │ ♝ │ ♛ │ ♚ │  5  (Black back rank)
├───┼───┼───┼───┼───┤
│ ♟ │ ♟ │ ♟ │ ♟ │ ♟ │  4  (Black pawns)
├───┼───┼───┼───┼───┤
│   │   │   │   │   │  3  (Empty)
├───┼───┼───┼───┼───┤
│ ♙ │ ♙ │ ♙ │ ♙ │ ♙ │  2  (White pawns)
├───┼───┼───┼───┼───┤
│ ♖ │ ♘ │ ♗ │ ♕ │ ♔ │  1  (White back rank)
└───┴───┴───┴───┴───┘

  Rook  Knight  Bishop  Queen  King
```

**Simplifications vs. original Gardner Chess** (pgx implementation):
- No pawn double-move
- No en-passant
- No castling

---

## Why This Environment for Our Research

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMAX  vs.  Gardner Chess                         │
│                                                                     │
│  SMAX                          Gardner Chess                        │
│  ─────────────────────         ─────────────────────               │
│  Cooperative multi-agent       Adversarial single-agent             │
│  Continuous health/position    Discrete board state                 │
│  Dense shaped rewards          Sparse terminal-only reward          │
│  ~100–200 actions              1,225 possible actions               │
│  Short episodes (~40 steps)    Long episodes (up to 256 steps)      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
         Together they demonstrate Algorithm 2 is general-purpose,
         not tuned to a single environment type.
```

| Property | Why It Matters for Consequence Reasoning |
|---|---|
| Deterministic transitions | Counterfactual reasoning is clean — all stochasticity comes from the opponent, not the environment itself |
| Sparse rewards only | Forces long-horizon credit assignment; exactly where consequence scoring is most useful |
| Large action space (1,225) | Tests whether consequence-weighted priority still finds informative transitions when ~95% of actions are illegal at any state |
| JAX-native | `jax.vmap` works over rollouts without Python loops; compatible with our entire training stack |
| Pre-trained baseline opponent | ~1,000 Elo AlphaZero model gives a consistent difficulty benchmark for win-rate evaluation |

---

## Observation Space

**Raw shape**: `(5, 5, 115)` float32  
**In our wrapper**: flattened to `(2875,)` float32

Think of the observation as a stack of 115 binary "photographs" of the board, each 5×5.
The network reads all 115 at once to understand the full game situation.

```
  The observation tensor:

       channel 0    channel 1    channel 2         channel 114
      ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐
      │0 0 0 0 0│  │0 0 0 0 0│  │0 0 0 0 0│  ...  │1 1 1 1 1│
      │0 0 0 0 0│  │0 0 0 0 0│  │1 0 0 0 0│       │1 1 1 1 1│
      │0 0 0 0 0│  │0 0 0 0 0│  │0 0 0 0 0│       │1 1 1 1 1│
      │1 1 1 1 1│  │0 0 0 0 0│  │0 0 0 0 0│       │1 1 1 1 1│
      │0 0 0 0 0│  │0 0 0 1 0│  │0 0 0 0 0│       │1 1 1 1 1│
      └─────────┘  └─────────┘  └─────────┘       └─────────┘
     "where are     "where is    "where is        "it is white's
      white pawns?"  white king?"  a black piece?"   turn" (all 1s)
```

### Where does 115 come from?

The encoding follows the AlphaZero paper, adapted for the 5×5 board.
The key idea: instead of just the current board, the agent sees the **last 8 board positions**
(history), so it can detect things like piece repetition and positional trends.

```
  115  =  (8 history steps  ×  14 planes per step)  +  3 game-state planes
       =        112                                  +  3
```

**Step 1 — The 14 planes per history step:**

```
  For each of the 8 most recent board positions:

  Planes 0–5   │ White's pieces — one plane per piece type
               │   plane 0: where are white's kings?   (0 or 1 per square)
               │   plane 1: where are white's queens?
               │   plane 2: where are white's rooks?
               │   plane 3: where are white's bishops?
               │   plane 4: where are white's knights?
               │   plane 5: where are white's pawns?
               │
  Planes 6–11  │ Black's pieces — same structure
               │   plane 6:  black kings
               │   plane 7:  black queens
               │   plane 8:  black rooks
               │   plane 9:  black bishops
               │   plane 10: black knights
               │   plane 11: black pawns
               │
  Plane 12     │ Has this exact position appeared before? (repetition 1)
  Plane 13     │ Has it appeared twice before?            (repetition 2)
```

**Step 2 — The 8 history steps stacked:**

```
  time step t-7  →  14 planes   channels   0 – 13
  time step t-6  →  14 planes   channels  14 – 27
  time step t-5  →  14 planes   channels  28 – 41
  time step t-4  →  14 planes   channels  42 – 55
  time step t-3  →  14 planes   channels  56 – 69
  time step t-2  →  14 planes   channels  70 – 83
  time step t-1  →  14 planes   channels  84 – 97
  time step t    →  14 planes   channels  98 – 111   ← current board
                                          ─────────
                                          112 planes so far
```

**Step 3 — The 3 remaining game-state planes:**

```
  Channel 112  │ Whose turn?     all 1s = white to move,  all 0s = black
  Channel 113  │ Move clock:     all cells = (half-moves since last capture or pawn move) / 100
  Channel 114  │ Move number:    all cells = total moves played / 512

  Note: standard chess AlphaZero also has 4 castling-rights planes here,
  but Gardner Chess has no castling → those 4 are dropped.
  That is why it is 115 and not 119 (full chess AlphaZero uses 119).
```

**Why does the agent need history?**  
A single snapshot cannot detect repetition or progress. Example: if the same position
appears 3 times the game is a draw — but you can only know that by comparing across
time steps. The 8-step history window lets the network see this directly.

pgx **automatically flips** the board so the observation is always from the current player's
perspective. Since our wrapper guarantees the DQN only ever sees white-to-move states, the
agent always views the board from white's side.

```python
CHESS_OBS_FLAT = 5 * 5 * 115  # = 2875
```

---

## Action Space

**Total actions**: 1,225 = 25 squares × 49 move types

```
  From-square (25 total)          Move type (49 total)
  ┌─────────────────────┐         ┌──────────────────────────┐
  │  0  1  2  3  4      │         │  N  NE  E  SE  S  SW  W  NW │  (8 dirs)
  │  5  6  7  8  9      │    ×    │  × distance (1..4 each)      │
  │ 10 11 12 13 14      │         │  + knight moves (8 types)    │
  │ 15 16 17 18 19      │         │  + promotion types           │
  │ 20 21 22 23 24      │         └──────────────────────────┘
  └─────────────────────┘
         25                   ×          49           =  1,225
```

**At any given state, most actions are illegal.** The environment provides a mask:

```
legal_action_mask:  [ 0 0 0 1 0 0 1 0 ... 0 1 0 ]   shape (1225,)  bool
                                ↑       ↑       ↑
                          only ~20–35 entries are True per state

Q-values:           [ q q q q q q q q ... q q q ]   shape (1225,)
                                ↓
After masking:      [ -∞ -∞ -∞ q -∞ -∞ q -∞ ... ]
                                ↓
action = argmax  →  picks the highest-Q legal move
```

```python
state.legal_action_mask         # (1225,) bool
q_masked = jnp.where(legal_action_mask, q_values, -jnp.inf)
action   = jnp.argmax(q_masked)
```

---

## Reward Structure

Rewards are **sparse** — zero on every step, non-zero only at game end.

```
  Episode timeline:

  Step:   1    2    3    4   ...   T-1    T
  ──────────────────────────────────────────────►  time
  Reward: 0    0    0    0   ...    0    ±1 or 0
                                          │
                              ┌───────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Terminal reward   │
                    │                     │
                    │  White wins  → +1   │
                    │  White loses → -1   │
                    │  Draw        →  0   │
                    └─────────────────────┘

  state.rewards  shape: (2,)
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
          rewards[0]            rewards[1]
          White's reward        Black's reward
          (what DQN sees)       (ignored)
```

**Why sparse rewards stress-test consequence scoring:**
Every move before the final one has reward 0, so standard TD learning assigns equal
(zero) priority to all non-terminal transitions. Consequence scoring breaks this tie by
measuring how much the future outcome distribution shifts after each action — identifying
the tactically decisive moves even though their immediate reward is 0.

---

## Termination Conditions

```
                       ┌──────────────────┐
                       │   Game Running   │
                       └────────┬─────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
   │  Checkmate  │      │     Draw     │      │  Truncation  │
   │             │      │              │      │              │
   │ King in     │      │ • Stalemate  │      │ 256 total    │
   │ check, no   │      │ • Insufficient│      │ steps        │
   │ legal moves │      │   material   │      │ (hard limit) │
   │             │      │ • Threefold  │      │              │
   │ reward: ±1  │      │   repetition │      │ reward: 0    │
   └─────────────┘      │ • 50-move    │      └──────────────┘
                        │   rule       │
                        │              │
                        │ reward: 0    │
                        └──────────────┘

   state.terminated = True              state.truncated = True
```

---

## Single-Agent MDP Wrapper

pgx exposes a two-player alternating interface. Our `GardnerChessEnv` wrapper collapses
it into a standard single-agent MDP: **the DQN is always white**.

```
  ┌────────────────────────────────────────────────────────────┐
  │  env.step(state, action)  — one call from DQN's perspective │
  │                                                            │
  │   pgx state                                                │
  │   current_player=0  ──►  White plays action                │
  │                               │                           │
  │                               ▼                           │
  │                          pgx state s1                     │
  │                          current_player=1                  │
  │                               │                           │
  │                          done? ──YES──► return (obs, r, done=True)
  │                               │                           │
  │                               NO                          │
  │                               ▼                           │
  │                         Opponent plays                     │
  │                         (random or baseline)              │
  │                               │                           │
  │                               ▼                           │
  │                          pgx state s2                     │
  │                          current_player=0  ◄── invariant  │
  │                               │                           │
  │                          return (obs, r, done)            │
  └────────────────────────────────────────────────────────────┘

  reward = s1.rewards[0] + s2.rewards[0]
         = white's reward from both half-moves
           (at most one is non-zero)

  KEY INVARIANT: every state stored in the replay buffer
                 has current_player == 0  (white to move)
```

---

## Opponent Modes

```
  ┌──────────────────────────────────────────────────────────┐
  │                      Opponent                            │
  │                                                          │
  │  'random'                    'baseline'                  │
  │  ────────                    ─────────                   │
  │  Sample uniformly from       pgx AlphaZero model         │
  │  legal_action_mask           "gardner_chess_v0"          │
  │                              ~1,000 Elo strength         │
  │  Good for:                                               │
  │  • Early training checks     Good for:                   │
  │  • Verifying env wrapper     • Real evaluation           │
  │  • Fast iteration            • Win-rate benchmarking     │
  │                              • Consequence rollouts      │
  │                              • Requires dm-haiku         │
  └──────────────────────────────────────────────────────────┘

  Both opponents are JAX-traceable → safe inside jax.jit / jax.vmap
```

---

## Q-Network Architecture

```
  Input
  ┌────────────────────┐
  │  (2875,) float32   │   ← flattened board observation
  └────────┬───────────┘
           │ reshape
           ▼
  ┌────────────────────┐
  │  (5, 5, 115)       │   ← spatial board tensor
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │  Conv(32, 3×3)     │   ← detect local piece patterns
  │  + ReLU            │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Conv(64, 3×3)     │   ← higher-order spatial features
  │  + ReLU            │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Conv(64, 1×1)     │   ← channel mixing / compression
  │  + ReLU            │
  └────────┬───────────┘
           │ flatten
           ▼
  ┌────────────────────┐
  │  (1600,)           │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Dense(512) + ReLU │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Dense(256) + ReLU │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │  Dense(1225)       │   ← one Q-value per possible action
  └────────┬───────────┘
           │ expand_dims(-2)
           ▼
  ┌────────────────────┐
  │  (1, 1225)         │   ← (n_agents, actions_per_agent)
  └────────────────────┘

  Why conv layers?  The (5,5) grid has spatial structure —
  nearby squares interact (attacks, defenses, pawn chains).
  Convolutions detect these patterns translationally.
```

---

## Consequence Scoring in Chess Context

Algorithm 2 applies identically to chess with two small adaptations:

```
  SMAX rollout step:                Chess rollout step:
  ─────────────────                 ──────────────────
  N agents act simultaneously  →    White acts, then opponent responds
  1 env.step() call            →    2 env.step() calls per "step"

  cf_horizon = 10 in chess
             = 10 move-pairs
             = 20 half-moves total
```

**Batched rollout shape** (identical to SMAX):

```
  _compiled_batched_fn(params, states, actions, keys)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
           B transitions    K candidate        N stochastic
           from buffer       actions            rollouts
              │                 │                 │
              └────────────────►▼◄────────────────┘
                           output: (B, K, N)
                           cumulative returns
                                │
                                ▼
                     consequence_metric()
                     Wasserstein distance between
                     actual-action and alt-action
                     return distributions
                                │
                                ▼
                     priority weight for replay
```

**Key chess-specific difference** — actions are scalars, not per-agent arrays:

```python
# SMAX: action is a tuple of N agent actions
actual_action = tuple(transition['a'])           # e.g. (3, 7, 2)

# Chess: action is a 1-tuple (single agent)
actual_action = (int(transition['a'][0]),)       # e.g. (42,)

# actions_array shape:
# SMAX:  (B, K, n_agents)
# Chess: (B, K)             ← scalar actions, no agent dimension
```

---

## Key Constants

```python
CHESS_OBS_FLAT = 2875    # 5 × 5 × 115
CHESS_ACTIONS  = 1225    # 25 squares × 49 move types
N_AGENTS       = 1       # white only (single-agent MDP)
```

## Evaluation Metrics

```
  Cumulative episode return  →  outcome classification

     return > 0  ──►  Win   ──►  counted in win_rate
     return < 0  ──►  Loss  ──►  counted in loss_rate
     return = 0  ──►  Draw  ──►  counted in draw_rate
                  also: avg_return, avg_episode_length
```
