# Validating Consequence Scores: Methodology

## The Core Question

Given a consequence score and an oracle label for each decision point, how do we
determine whether the scores are *good*?

Three metrics, each catching a distinct failure mode.

---

## Metric 1 — Spearman Rank Correlation (Primary)

**What it tests:** Does the overall ordering of states by consequence match the
oracle's ordering?

**Why it's the headline number:**
- Scale-invariant: doesn't matter that our scores are in [0, ∞) and oracle
  values are in [-1, 1]
- Robust to outliers compared to Pearson
- Directly comparable across environments (chess vs FrozenLake)
- A single interpretable number: ρ ∈ [-1, 1], p-value for significance

**Failure mode it catches:** The method is ordering states wrong overall — high
scores on unimportant states, low scores on important ones.

**How to compute:**

For chess (sampled decision points — ~50 games, ~15–25 moves each, ~750–1250 points):
- oracle label = `mean_k≠chosen |v_chosen - v_k|` where v = AlphaZero value head
  (negated to white's perspective), chosen = move the baseline model played
- consequence score = Wasserstein score from `ChessCounterfactualAnalyzer`
- `ρ = spearmanr(consequence_scores, oracle_scores)`

For FrozenLake (full state space — all states enumerated, not sampled, works
for any grid size):
- oracle label = `mean_{a≠chosen} |Q*(s, chosen) - Q*(s, a)|` from value iteration,
  where chosen = argmax_a Q*(s, a)
- consequence score = Wasserstein score from the JAX consequence analyzer
- `ρ = spearmanr(consequence_scores, oracle_scores)` over all states

**Expected result:** ρ > 0, p < 0.05 for both environments.

---

## Metric 2 — Precision@K / Top-K Agreement (Secondary)

**What it tests:** Are the states our method flags as *most* consequential the
same ones the oracle flags?

**Why this matters:** The whole point of consequence-weighted replay is
prioritization — we care most about whether we correctly identify the top tier
of important states, not whether we rank the bottom half perfectly.

**Failure mode it catches:** Good Spearman overall but the top states are wrong.
A method could have high rank correlation yet completely miss the most critical
positions — this metric exposes that.

**How to compute:**

Let `K` be a threshold (top 5%, 10%, 20% of states by score):

```
top_oracle_K  = set of states in top K% by oracle label
top_ours_K    = set of states in top K% by our consequence score
Precision@K   = |top_oracle_K ∩ top_ours_K| / |top_ours_K|
```

Report for K = 5%, 10%, 20%. Random baseline = K (e.g. for N=100 states and K=10%,
random expected overlap = 10/100 = 0.10, so Precision@K baseline = 0.10 = K).

**Expected result:** Precision@K > random baseline at all thresholds, largest
gap at K=5% (hardest, most meaningful).

---

## Metric 3 — Sampling Distribution Agreement (Tertiary)

**What it tests:** Would an agent using our scores sample consequential states
as often as one using the oracle would?

**Why this matters:** The buffer samples transition `i` with probability
`p_i ∝ consequence_score_i ^ beta`. This is the mechanism that actually drives
learning — it doesn't matter if scores are correctly ordered if the resulting
sampling probabilities don't concentrate on the right states. This metric tests
the end-to-end prioritization behaviour directly.

**Failure mode it catches:** Good Spearman + good Precision@K but the scores
are too uniform — the most important states aren't getting substantially higher
sampling probability. The prioritization signal exists in the ordering but is
too weak to matter in practice.

**How to compute:**

Using the same `beta` as the buffer (chess: beta=0.4, FrozenLake: TBD when
trainer is built). Add epsilon before raising to avoid zeros:

```python
eps = 1e-8
p_ours   = (consequence_scores + eps) ** beta
p_ours  /= p_ours.sum()

p_oracle = (oracle_spreads + eps) ** beta
p_oracle /= p_oracle.sum()

kl_div = scipy.special.rel_entr(p_oracle, p_ours).sum()  # KL(oracle || ours)
r_samp, p_samp = pearsonr(p_oracle, p_ours)
```

Report both KL divergence (lower = better, 0 = identical distributions) and
Pearson r between the two probability vectors (higher = better). KL is the
primary number; Pearson gives a scale-comparable summary alongside the other
two metrics.

**Expected result:** KL close to 0, Pearson r > 0 and consistent with
Spearman ρ from Metric 1.

---

## Summary Table

| Metric | What it tests | Failure mode |
|---|---|---|
| Spearman ρ | Overall ordering correct | Wrong rank structure |
| Precision@K (5%, 10%, 20%) | Most important states identified | Top states missed |
| Sampling KL + Pearson | Sampling distribution matches oracle | Scores too uniform to drive prioritization |

A method that scores well on all three is genuinely capturing consequence
structure. A method that scores well on Spearman but poorly on Precision@K is
failing in a way Spearman alone wouldn't reveal. A method that passes both but
has high sampling KL is correctly ordered but too compressed to actually
prioritize effectively.

---

## Environment — Claim Assignment

| Environment | Claim 1 (scoring) | Claim 2 (sample efficiency) |
|---|---|---|
| Gardner chess | Yes — primary | Yes |
| FrozenLake (planned) | Yes — secondary | Yes |
| SMAX | No | Yes — primary |

SMAX has no independent oracle. The only available label (per-state perturbation)
uses the same rollout mechanism as the algorithm itself, making any correlation
nearly circular. SMAX is therefore used exclusively for Claim 2.

## Ground Truth Labels

### Gardner chess
**Source:** AlphaZero value head (`pgx.make_baseline_model("gardner_chess_v0")`)

For each decision point, the baseline model plays the game (chosen action) and
K candidate alternatives are evaluated via the value head. The oracle mirrors
the algorithm's structure — chosen action as reference, diverge against each
alternative, aggregate:

```
v_k        = -value(step(s, a_k))          # white's value after candidate move k
                                            # negation: pgx returns from current
                                            # player's perspective (black after white moves)
v_chosen   = -value(step(s, a_chosen))

oracle_score(s) = mean_k≠chosen |v_chosen - v_k|
```

High score = the baseline model's chosen move leads to a board evaluation that
is substantially different from what the alternatives would have produced — the
move genuinely mattered by the oracle's standards.

### FrozenLake
**Source:** Optimal Q-function Q*(s,a) via dynamic programming (slippery=True)

FrozenLake exposes its full transition model via `env.P[state][action]`, which
gives exact `(probability, next_state, reward, done)` tuples for any grid size.
Run value iteration to convergence to get V*(s), then extract Q*(s,a):

```
Q*(s, a)   = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]

chosen     = argmax_a Q*(s, a)             # optimal action at this state

oracle_score(s) = mean_{a ≠ chosen} |Q*(s, chosen) - Q*(s, a)|
```

This directly mirrors the algorithm: chosen action as reference, average
absolute difference against each alternative. High score = the optimal action
is substantially better than the alternatives — acting suboptimally here costs
a lot.

Using slippery=True: actions are stochastic (1/3 probability of intended
direction, 1/3 each perpendicular). The oracle reflects expected value
differences under the true stochastic dynamics, making importance more graded
than in a deterministic grid.

This is analytically exact — no trained model, no approximation, derived
entirely from environment structure. Works for any grid size since the only
input is `env.P`, which Gymnasium generates automatically.

---

## Set Aside: Perturbation Experiment

*Not for immediate implementation — future work or supplementary result.*

Take the states flagged as most consequential by our method. Force the agent to
act *badly* at exactly those states (e.g., take the worst-scoring action instead
of the best). Measure reward drop.

**Output:** A curve — fraction of states perturbed (x-axis) vs. reward drop
(y-axis). Steeper = the method found a small set of states that genuinely matter.
Each scoring method gets its own curve. Random selection is the baseline.

**Why it's compelling:** It's a direct causal test. Not "do the scores correlate
with importance" but "does acting on these scores actually matter for outcomes."

**Why it's set aside:** Requires a trained agent + controlled perturbation
infrastructure that doesn't exist yet. Best done after the correlation metrics
are validated.
