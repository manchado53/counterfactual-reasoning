# CCE Diagnostics — Is the Consequence Score a Good Importance Signal?

**TL;DR.** Counterfactual Consequence Estimation (CCE) is a **precise, sparse,
zero-false-positive decision-fork detector**. It produces large sample-efficiency
gains in **deterministic** environments, but its total-variation metric is
**saturated by environmental stochasticity**, which neutralizes the signal. An
offline diagnostic that measures how well CCE tracks exact ground-truth importance
**predicts the win-rate** in every scenario tested.

---

## 1. Motivation

CCE scores each replay transition by *how much the action choice matters*, then
uses that score to prioritize experience replay. Across environments CCE failed to
robustly beat plain Prioritized Experience Replay (PER, which prioritizes by TD
error), and we had no visibility into *why* — we were tuning hyperparameters blind.

The diagnostic interrogates the **mechanism** instead of the outcome. It loads
trained checkpoints and, for thousands of visited states, compares three numbers:

| signal | what it is |
|---|---|
| **CCE score** | our method's consequence score (total variation of per-action return distributions) |
| **\|TD error\|** | the free signal PER already uses |
| **ground-truth stakes** | from a strong/exact oracle: how much the action choice *truly* changes the optimal outcome |

It then asks three questions:

```
Q1  Is there structure?      — does CCE vary, or say the same thing for everything?
Q2  Is CCE just TD?          — rank-correlation between CCE and |TD|
Q3  Who matches truth?       — correlation of each with the ground-truth stakes
```

Ground-truth oracle per environment:
- **Connect Four** — MCTS with 200 simulations (`opponent_mcts`), per-action root Q-values.
- **FrozenLake** — **exact** optimal Q\*(s,a) by value iteration on the known MDP
  (`analysis/diagnostics/value_iteration.py`). Stakes = `max_a Q* − min_a Q*`.

---

## 2. Results

All correlations are Spearman ρ. "Per-state" groups by unique state (each counted
once) to remove visit-frequency bias — it is the fairest apples-to-apples number.

| Environment | ρ(CCE, truth) | ρ(TD, truth) | ρ(CCE, TD) | CCE behavior |
|---|---|---|---|---|
| Connect Four (MCTS) | 0.29 | 0.36 | 0.09 | weak, distinct from TD |
| FrozenLake **slippery** | **−0.25** | 0.77 | −0.02 | **saturated ~0.7 — useless** |
| FrozenLake **deterministic** | **+0.38** | 0.50 | 0.07 | **sparse fork-detector** |

**The headline:** removing stochasticity flips CCE from −0.25 (useless) to +0.38
(genuinely good), nearly closing the gap with TD. Stochasticity is the killer.

### The money figure — `d5_cce_vs_truth.png`

x-axis = true stakes, y-axis = CCE score.

- `docs/figures/diagnostics_fl/d5_cce_vs_truth.png` (**slippery**) — a flat smear:
  CCE spans 0→1 at every stake level. No relationship.
- `docs/figures/diagnostics_fl_det/d5_cce_vs_truth.png` (**deterministic**) — dots
  snap to the corners: low-stakes→CCE 0, high-stakes→CCE 1, and the **top-left is
  empty** (CCE never fires on a low-stakes state — zero false positives). Its only
  error is occasional shyness (high-stakes→CCE 0), never crying wolf.

---

## 3. Win-rate corroboration — the diagnostic predicts the outcome

**Deterministic FrozenLake** (`claim2_no_slip` experiment, 25 seeds, 8×8, % of seeds
reaching ≥90% within 15k episodes — deterministic FL is solvable, so this is a
sample-efficiency / reliability measure):

| algorithm | % seeds solved | median episodes → 90% |
|---|---|---|
| DQN-uniform | 40% | never (>15000) |
| DQN+PER | 48% | never (>15000) |
| CCE additive μ=1.0 | 56% | 13,713 |
| CCE additive μ=0.25 | 67% | 10,626 |
| CCE multiplicative μ=0.25 | **80%** | **8,980** |

**Every CCE variant beats both baselines.** The best (multiplicative) solves 80% of
seeds vs PER's 48%.

**Slippery FrozenLake** (`claim2_main`): CCE additive μ=1.0 IQM 0.677 vs PER 0.654,
`P(improvement) = 0.525 [0.31, 0.73]` — a coin flip. Null result, as the diagnostic
predicts (the signal is saturated garbage there).

```
                   DIAGNOSTIC ρ(CCE,truth)     WIN-RATE vs PER
Slippery (stoch)   −0.25  saturated, useless    null (coin-flip)
Deterministic      +0.38  precise fork-detect    clear win (80% vs 48%)
```

The science and the outcome agree in both directions.

---

## 4. Mechanism

**Why stochasticity saturates CCE.** CCE measures the total variation between the
per-action *return distributions*. Under slippery dynamics every action produces a
broad, overlapping blur of outcomes, so the distributions always look "different" —
TV maxes out at a near-constant ~0.7 for almost every state. The metric detects
*that* actions differ, never *by how much*. In deterministic dynamics the return
distributions are clean point masses, so TV cleanly resolves real forks.

**Why CCE has false negatives (even when deterministic).** CCE rolls out under the
**agent's own greedy policy**, not the optimal one. At a fork the agent has not yet
mastered, all rollouts fail to reach the goal → all returns 0 → CCE sees no
consequence, even where the optimal stakes are large. TD error does not share this
limitation (it reads value error directly), which is why TD edges CCE on raw
correlation while CCE wins on *precision*.

**Why precision beats correlation for replay.** CCE is sparse and never cries wolf;
it surgically flags the exact fork transitions and wastes no priority elsewhere.
That focus is what drives the sample-efficiency win, and raw correlation
under-credits it.

---

## 5. Conclusion

> CCE is a precise decision-fork detector that delivers large gains in deterministic
> environments and is neutralized by environmental stochasticity, which saturates its
> total-variation metric.

This is a sharp, *predictive* claim: the importance-signal quality (measurable
offline, before any training) tells you whether CCE will help.

---

## 6. Limitations & follow-ups

1. **SMAX diagnostic (highest priority).** SMAX is stochastic, so this finding
   predicts the observed SMAX lead (CCE-additive 71.7% vs ~64–65%) may be fragile.
   Running the diagnostic there (needs a ground-truth oracle — e.g., Monte-Carlo
   action values under a strong frozen policy) would confirm or refute it.
2. **A slip-robust metric.** Replace total variation of the return *distributions*
   with a magnitude-preserving statistic (e.g., spread of the *mean* returns across
   actions, `max_a E[return] − min_a E[return]`). This directly targets "stakes" and
   may rescue CCE under stochasticity.
3. The diagnostic correlation is computed over *visited* states; the per-state
   grouping mitigates but does not eliminate distribution bias.

---

## 7. Reproduce

Code: `src/counterfactual_rl/analysis/diagnostics/`

```
compute_diagnostics.py      Connect Four compute stage (MCTS-200 oracle)
compute_diagnostics_fl.py   FrozenLake compute stage (slippery & deterministic)
value_iteration.py          exact optimal Q* oracle for FrozenLake
plot_diagnostics.py         env-agnostic plotting (figures d1–d9)
run_diagnostics.sh          SLURM wrapper (Connect Four)
run_diagnostics_fl.sh       SLURM wrapper (FrozenLake)
```

```bash
# FrozenLake — slippery (jobs 255495-497) and deterministic (257556/572/571)
sbatch --export=ALL,DIAG_RUN_IDS="255495 255496 255497",\
DIAG_OUT_NPZ=".../docs/figures/diagnostics_fl/diagnostics.npz" run_diagnostics_fl.sh

sbatch --export=ALL,DIAG_RUN_IDS="257556 257572 257571",\
DIAG_OUT_NPZ=".../docs/figures/diagnostics_fl_det/diagnostics.npz" run_diagnostics_fl.sh

# plot (cheap, local)
python -m counterfactual_rl.analysis.diagnostics.plot_diagnostics \
  --npz .../diagnostics_fl_det/diagnostics.npz \
  --out .../diagnostics_fl_det --truth-label "optimal Q*"
```

Outputs (figures `d1`–`d9` + `diagnostics.npz`):
```
docs/figures/diagnostics/        Connect Four
docs/figures/diagnostics_fl/     FrozenLake slippery
docs/figures/diagnostics_fl_det/ FrozenLake deterministic
```
