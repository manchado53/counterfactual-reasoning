# Instrumentation build — make the buffer tell us what it is doing (2026-09-03)

## Why

CCE is a null on JaxNav. We think we know why: only ~1.6% of the replay buffer ever
gets a real CCE score, so the CCE arms have been doing uniform replay at ~5x the cost.

**That 1.6% is arithmetic on three config numbers. Nothing in the code has ever
recorded which buffer entries hold a real score.** Every question this week needed its
own GPU job because the buffer is never saved and never described.

This build adds the instruments. It changes no algorithm and no default behaviour.

## What we are adding

**In the shared buffer** — a `scored_at` array that remembers, per entry, the Q-update
at which it last received a measured score (`-1` = never). From that: `scored_frac`
(settles the 1.6%) and staleness percentiles. Plus a log-spaced histogram of the score
distribution with a separate exactly-zero counter, logged at every eval so the same data
gives a heatmap, a ridgeline and an animation without re-running. Plus a compressed npz
snapshot of the raw buffer at ~10 checkpoints, so the buffer is inspectable after the
fact — it never has been.

**In the eval** — how episodes actually END: goal / crash / timeout, and episode length
split by outcome. Today's single `avg_length` averages crashes and successes together, so
it moves for two opposite reasons at once. Written to a new `eval.jsonl` for every arm,
including uniform and PER, so crash rates are comparable across arms.

**In the CCE rollouts** — how each counterfactual rollout ends, same four buckets. CCE
assumes its rollouts predict reality and nobody has ever checked that. Putting rollout
outcomes next to eval outcomes is the cheapest real test of the method's core assumption.

**Double-sampling** — every so often, draw a second batch using TD priority alone, do
not train on it, and log how much it overlaps the batch we actually used. Asked for in
July, never built. It answers "is CCE picking different transitions at all?" directly,
instead of inferring it from concentration statistics.

## What we are NOT doing

No beta sweep (`(0+eps)**beta` is one constant for 98% of the buffer — no exponent can
rank it). No coverage sweep until `scored_frac` exists and reports a real number. No
priority normalisation — that is issue #16, deferred. No new algorithm.

## How we will know it worked

A 10-minute single-arm smoke run, before anything overnight. `scored_frac` is fixed by
config, so the smoke run is what confirms or kills the 1.6% — the overnight run buys the
trajectory, the figures and the extra arm, not the headline number.

The overnight run is 3 seeds. That is deliberate: this is a mechanism run, debugging the
instruments. JaxNav's paired sd is 30pp, so no seed count we can afford would show that
any of this changes learning, and none of it will be reported as if it could.
