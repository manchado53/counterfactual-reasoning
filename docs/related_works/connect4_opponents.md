# Connect Four: Opponent Options for Training and Evaluation

Survey of available opponent strategies for Connect Four, assessed for use in CCE Claim 2 experiments.
Covers implementation complexity, JAX compatibility, strength, and suitability as a training/eval baseline.

---

## Literature Benchmarks (Verified)

The following win-rate hierarchy is confirmed across multiple sources (primary: arXiv:2405.16595, arXiv:2204.13307):

| Approach | vs Random | vs Heuristic | vs Minimax |
|---|---|---|---|
| MCTS (decisive-move rollout) | 100% | 94.75% | Wins 69% |
| Minimax depth-6 (alpha-beta) | ~99.5% | 75% | — |
| Tabular Q-Learning (200k games) | 75.5% | 1% | Loses 100% |

**Note:** No peer-reviewed head-to-head numbers exist for *neural-network* DQN/PPO vs. MCTS or minimax on canonical 7×6 Connect Four. All Q-Learning results above use tabular methods.

Best peer-reviewed result: **TCL-wrap** (Scheiermann & Konen, IEEE Transactions on Games, arXiv:2204.13307) — TD learning + MCTS at evaluation time only. Achieves 66.3% tournament win rate, beats AB-DL baseline 99/0/1 with 500+ MCTS iterations. Required 6,000,000 training episodes.

---

## Option 1 — Random Opponent (current baseline)

**Strength:** Floor  
**JAX compatible:** Yes — trivial  
**Training overhead:** Zero

Selects uniformly from legal columns. Used universally as the minimum competency check. Any trained agent should reach >90% win rate; the signal disappears quickly and high win rates here do not distinguish algorithms.

**Verdict:** Too weak for Claim 2. Useful only as a sanity check.

---

## Option 2 — Rule-Based Heuristic: Win → Block → Fork → Center (current opponent)

**Strength:** Weak-medium  
**JAX compatible:** Yes — fully JIT + vmap  
**Training overhead:** ~1 array op per move

Priority scoring: immediate win (1000) → block opponent win (100) → create double threat (10) → center column bias (0–3) → random tiebreak. No lookahead beyond one move.

This is essentially the "decisive move" heuristic used as the rollout policy inside the best MCTS agents in the literature (arXiv:2405.16595). The gap vs. MCTS is the *lookahead*, not the per-move scoring logic.

**Verdict:** Currently in use. Academically reproducible but not a standard citable baseline. Creates some tactical moments but misses multi-step threats entirely.

---

## Option 3 — MCTS with Decisive-Move Rollouts (recommended upgrade)

**Strength:** Medium-strong (94.75% vs heuristic, 69% vs minimax depth-6)  
**JAX compatible:** Yes — via `google-deepmind/mctx`  
**Training overhead:** ~3–20× slower per chunk depending on sim count

Standard Monte Carlo Tree Search where each simulation uses the decisive-move heuristic (Option 2) as the rollout policy. This is the top-ranked approach in arXiv:2405.16595.

`mctx` is Google DeepMind's JAX-native MCTS library, explicitly designed for pgx environments. The `recurrent_fn` wraps `env.step` directly — pgx `State` is the embedding. Fully JIT + vmap compatible.

**Simulation count tradeoffs:**

| Sims | Strength | Overhead vs rule-based |
|---|---|---|
| 8–16 | Noticeably stronger, 1–2-step threats | ~3–5× |
| 32–64 | Medium-strong, reliable tactical play | ~8–12× |
| 128–256 | Approaches near-perfect short-game play | ~20–40× |
| 1000+ | Near-optimal (used in TCL-wrap) | impractical for training |

**Implementation:** ~50 lines in a new `opponent_mcts.py`. The `recurrent_fn` is a thin wrapper around `env.step`; no neural network needed for opponent use. Prior logits = uniform over legal moves; value = 0 (or random rollout).

**Verdict:** Best option for Claim 2. Citable, well-known algorithm, adjustable difficulty, stays inside the JAX training loop. Recommended sim count: **16–32 for training**, **64–128 for evaluation**.

---

## Option 4 — Fixed-Depth Negamax (no alpha-beta)

**Strength:** Medium (depth 4 ≈ minimax depth-4; depth-6 minimax beats heuristic 75%)  
**JAX compatible:** Partial — requires fixed-depth unrolled tree, no pruning  
**Training overhead:** 7^4 = 2,401 leaf evals at depth 4; feasible but complex

Classic game-tree search. Alpha-beta pruning cannot be JIT-compiled (data-dependent branching), so a pure negamax with a fixed tree depth must be fully enumerated. At depth 4, all 2,401 board positions in the lookahead tree are evaluated simultaneously via JAX `vmap`. Requires a hand-crafted board evaluation function (piece counts, threat counts, center bias).

Depth-6 minimax is the second-ranked approach in arXiv:2405.16595 and a well-established benchmark in the game AI literature.

**Verdict:** Viable but non-trivial to implement in JAX. More complex than mctx for similar or slightly lower strength. Not recommended unless mctx is ruled out.

---

## Option 5 — Frozen Pre-Trained DQN (self-generated baseline)

**Strength:** Medium-strong (our own runs achieve 76–90% vs random)  
**JAX compatible:** Yes — just a forward pass  
**Training overhead:** Same cost as running our own network

Train one DQN agent to convergence, freeze its weights, use it as a fixed opponent for all subsequent experiments. Fully JAX-native, no additional libraries needed.

**Downsides:** Less "objective" as a baseline — a reviewer may ask why the baseline is the paper's own trained model. Strength depends on how well the training run went. Creates a potential circularity concern if CCE is what produced the frozen agent.

**Verdict:** Acceptable if MCTS is impractical, but weaker scientifically as a cited baseline.

---

## Option 6 — Perfect Solver (Pascal Pons)

**Strength:** Perfect — first player wins 100% with optimal play  
**JAX compatible:** No — C++ binary  
**Training overhead:** N/A (cannot be used in training loop)

Connect Four is a solved game (Allis, 1988). Pascal Pons' open-source solver (github.com/PascalPons/connect4) plays perfectly and is the gold-standard reference in academic benchmarking (Elo ~2000). Used to measure how close a trained agent gets to optimal play.

**Verdict:** Not usable for training. Useful as a final evaluation checkpoint to report in the paper ("our best agent achieves X% against perfect play"). No JAX integration exists.

---

## Option 7 — AlphaZero-Style Neural MCTS (self-play)

**Strength:** Strong to near-perfect (AlphaZero.jl beats depth-5 minimax after 75k self-play games)  
**JAX compatible:** Yes — `a0-jax` (NTT123/a0-jax) implements AlphaZero in JAX for Connect Four  
**Training overhead:** Very high; requires alternating policy improvement + self-play loops

AlphaZero combines MCTS with a learned policy/value network, iteratively improving both through self-play. After 75,000 self-play games with 600 sims/move, AlphaZero.jl beats depth-5 minimax using the network alone at inference.

**Critical problem for Claim 2:** The opponent gets stronger as the agent trains, making the performance metric non-stationary. Win rate vs. self is not a comparable metric across algorithms. See discussion in `CLAIM2_METRICS.md`.

**Verdict:** Not suitable for Claim 2 sample efficiency evaluation. Strong future work candidate.

---

## Recommendation Summary

| Option | Use for Training | Use for Eval | Priority |
|---|---|---|---|
| Random | Sanity check only | No | Low |
| Rule-based (Win→Block→Fork) | ✅ Currently running | ✅ Currently running | Keep as secondary |
| **MCTS (mctx, 16–32 sims)** | **✅ Recommended** | **✅ Recommended** | **Primary** |
| Negamax depth-4 | Possible | Possible | Fallback if mctx blocked |
| Frozen DQN | Possible | Possible | Last resort |
| Perfect solver | ❌ No JAX | ✅ Final evaluation only | Optional appendix |
| AlphaZero self-play | ❌ Confounds metrics | ❌ Non-stationary | Future work |

**Decision:** Use MCTS via `mctx` with 16–32 simulations as the unified training + evaluation opponent. This eliminates the train/eval mismatch concern, provides a citable standard baseline, and is meaningfully stronger than the current heuristic while remaining JAX-compatible.

---

## Key Sources

- Taylor & Stella (2024). *An Evolutionary Framework for Connect-4 as Test-Bed for Comparison of Advanced Minimax, Q-Learning and MCTS.* arXiv:2405.16595
- Scheiermann & Konen (2022). *AlphaZero-inspired General Game Learning.* arXiv:2204.13307. IEEE Transactions on Games.
- Wang et al. (2019). *Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play.* IEEE CoG 2019.
- Laurent, J. AlphaZero.jl Connect Four Tutorial. https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/
- Google DeepMind. mctx: Monte Carlo tree search in JAX. https://github.com/google-deepmind/mctx
