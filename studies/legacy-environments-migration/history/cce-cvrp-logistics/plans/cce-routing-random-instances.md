# Random-instance routing for Claim 2

## Why

Every routing Claim-2 result so far — roughly 3,100 runs — was trained on **one fixed
map**. The same ten customers, the same coordinates, the same demands, every episode,
every seed. That is not what the field does; Kool et al. and the papers that follow it
draw a fresh instance every episode.

Three explanations survive for the routing null, and the current data cannot tell them
apart:

1. `beta = 0.25` made all five arms sample almost uniformly (issue #4, measured: top
   transition only 1.20x more likely than bottom, ESS 0.999).
2. On one fixed map the agent sees the same handful of situations thousands of times,
   so *which* transition you replay cannot matter much. Argued, never measured.
3. CCE genuinely does not help routing.

Reading 3 is the interesting one, and we cannot claim it while 1 and 2 are open. This
plan closes 2 and folds in a check on 1.

It also removes a real reviewer objection: right now every environment in the paper is
one we built, and routing is one map inside it.

## What gets randomized

An **instance bank**: K pre-generated maps, one drawn per episode.

Randomized per instance:

- **customer coordinates** — uniform in the unit square, the field's convention
- **demands** — integers, re-checked so the instance stays feasible

Held fixed on purpose, so difficulty does not drift with geometry:

- 10 customers, capacity 10, `allow_stranding=True`, `reward_shape='terminal'`
- **budget set per instance** as `0.95 x that instance's own exact optimal
  all-customers tour`. This is the important one. A budget fixed in absolute units
  would make some maps trivial and others impossible, and the comparison would measure
  instance luck instead of replay strategy.

Not randomized: number of customers. Varying it changes the action space and the state
count, and buys nothing for this question.

## The constraint that shapes the design

The trainer is **tabular-indexed**. `env.state_features` is an `(n_states,
feature_dim)` array, the buffer stores integer state indices, and updates gather with
`features[states]`. A different map every episode makes a bare state index meaningless.

So: pre-build K instances, keep each one's tables, and store `(instance_id,
state_index)` in the buffer. Everything stays tabular, the exact oracle still works
per instance, and we can still measure stakes.

Cost is fine — 34.3 MB of tables per instance:

```
 8 instances   0.27 GB
16 instances   0.54 GB
32 instances   1.07 GB
```

16 is the working default. Roughly 10 s to build each, so ~3 min of startup.

The alternative — computing features on the fly and dropping enumeration — is the
"proper" fix but it is a rewrite and it costs us the exact oracle. Not now.

## How it gets measured

Fixed from what this environment already taught us:

- **Continuous metric, not binary.** Mean `opt_ratio` across the bank. The binary
  "solved" metric collapsed a real 7.5-point spread into `0/40 vs 2/40` at the floor,
  and into `38/40 vs 40/40` at the ceiling.
- **Measure inside the window.** On the fixed map the arms only separate between
  roughly 15k and 25k episodes; before that nothing solves, after that everything
  does. Gate 2 relocates that window for the bank, and the sweep runs to the top of
  it rather than to 50k.
- **Log ESS every eval.** Without it we cannot tell "CCE did not help" from "the
  sampler never concentrated". This also answers issue #9 directly, and dropping the
  raw score histogram alongside it answers issue #8.

Secondary metric: mean `opt_ratio` on a held-out set of instances the agent never
trained on. That is the field's generalization measure. Reported, not primary — Claim 2
is about sample efficiency, not generalization.

## Gates

**Gate 1 — build the bank, no training.** Generate K instances, solve each exactly,
and check: every instance is feasible, the per-instance optimum is comparable across
the bank, and the three-band stakes distribution still looks like the fixed map's
(middle band under ~25%). If randomizing geometry destroys the bimodal stakes that
Option A bought us, stop and report that — it is a real finding and it costs no
compute.

**Gate 2 — one arm, ~20 seeds.** Plain DQN-uniform on the bank. Find where the learning
curve crosses 50% and confirm there is a window at all. Set the episode budget from
that, and write it into the notebook before the sweep runs.

**Sweep.** 5 arms x 2 beta levels x 20 seeds at the Gate-2 budget. `beta = 0.25` is the
shipped value and the control; the second level comes from the ESS measured in Gate 2,
picked to land ESS near 0.6 rather than 0.999.

## Registered prediction, written before anything runs

To be filled in at the end of Gate 2, in the notebook, with a direction and a number.
The honest prior right now: randomizing instances raises the value of prioritized
replay for *every* arm including PER, so the thing to watch is whether CCE's gap over
PER widens, not whether CCE beats uniform.

## What this does not do

- Does not fix the zero-CCE-score problem (issue #7). Routing has a dense per-step
  reward, so it should not have that failure — but that is an assumption, and Gate 1
  is the place to check it.
- Does not touch Claim 1. Claim 1 is merged, needs the fixed map, and stays fixed.
- Does not adopt an external benchmark. That is still the right long-term move; this is
  the cheap version that reuses everything we already have.
