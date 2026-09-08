# The JaxNav story — 7 figures, one argument

Written 2026-09-04 off array 274678 (mid-flight, ~178k of 250k episodes) plus the
finished bootstrap sweep 274476. Every number here is measured; nothing is projected.

## The framing that makes this a contribution, not a failure

Do **not** open with "CCE didn't work on JaxNav." That is a negative result about one
environment and it invites the obvious question we cannot answer (why should anyone care
about JaxNav?).

Open with this instead:

> **We can now measure, from three config values and a buffer snapshot, whether CCE is
> capable of prioritising at all in a given environment — before spending a single GPU-hour
> on the comparison.**

That is Jeremy's 07/10 ask, verbatim: *"additional metrics beyond SNR to predict when the
method will succeed."* JaxNav is the environment that made it findable, because it is where
the method is at its most broken and the breakage is arithmetic rather than statistical.

The 230 JaxNav runs stop being a sunk cost and become the negative control that calibrated
a diagnostic.

---

## Beat 1 — A real fix to the score changed nothing

`fig_bootstrap_curves.png` · `fig_why_null.png`

Bootstrap (issue #7) is implemented, correct, and measured. Within each panel the only
difference between the two lines is `cf_bootstrap`.

```
   mean CCE score      +187%
   ess_frac              +1%
   final win rate        +0%     (20 seed-matched pairs, Wilcoxon p = 0.837)
```

The score got much better. Learning did not move. **Why not** is the rest of the story.

## Beat 2 — Because almost none of the score reaches the sampler

`fig_signal_funnel.png` — the single strongest slide

```
   100,000  transitions in the buffer
       887  have a MEASURED consequence score     x 0.0089   <- coverage
       283  and that score is NON-ZERO            x 0.32     <- 68% come back 0
   ----------------------------------------------------------------
     0.28%  of the buffer carries usable CCE signal
```

Two bottlenecks, and they **multiply**. Coverage was the one we knew about; the zero-rate
is the second and it halves the first again.

Coverage is set by three config values and nothing else:

```
                 n_score_sample                64
   coverage  =  ------------------------  =  --------  =  1.6% per transition LIFETIME
                score_interval x steps        250x16       ->  0.85% on the belt

   FrozenLake (CCE WINS)   128/(100x4)  =  32%  lifetime  ->  14.4%
```

**Buffer capacity cancels out.** Measured directly: coverage was 0.85% whether the buffer
held 65k or 100k entries, and it is flat from episode 5k to 178k.

## Beat 3 — And what the other 99% carries is a clock

`fig_score_ridgeline.png` · `fig_score_distribution.png`

An entry that is never scored is handed `mean(consequence_scores)` at insertion
(Algorithm 1 line 7). The first entry ever gets a hard-coded `0.0`. Everything after is an
average of a buffer that is ~99% entries which received the average:

```
   new value = 0.9915 x (old inherited) + 0.0085 x (measured mean)
             = 0.9915 x 0.00442 + 0.0085 x 0.00208 = 0.004400
                                          (the buffer shows 0.004400)
```

Time constant `1/0.0085 = ~118 buffer turnovers`; a whole 250k run is ~101. **The value
never catches up within a run.** Consequences, measured on the unscored subset only:

```
   corr(entry age, inherited value)      -0.71   <- newer = higher, the mean is climbing
   spread across the whole 99%           1.25%
   after the beta=0.25 exponent          0.31%
```

So for 99% of the buffer, the CCE priority encodes **when the transition arrived** and
nothing about the transition, at a strength of 0.31%. That is uniform replay with a
rounding error of recency bias — and it is exactly what `ess_frac` 0.91-1.00 says.

## Beat 4 — Therefore no fix to score QUALITY can matter

`fig_bootstrap_side_by_side.png`

Same config, same seeds, only `cf_bootstrap` differs:

```
   CHANGED                     OFF        ON
     scores exactly 0         0.84%     0.09%
     mean score               0.004     0.156      39x
     bins occupied               14        28
   UNCHANGED
     coverage                 0.84%     0.87%
     top-2 bins hold         99.16%    99.37%
```

Everything bootstrap improved lives inside the 1% that has a measurement. Everything that
decides sampling lives in the 99% that does not.

**This argument is not about bootstrap.** It applies to every future improvement to score
quality, including issue #16 (rank-vs-min-max normalisation). Coverage is the only lever
that moves the 99%.

## Beat 5 — What not knowing this has cost

`prob_and_cost.txt` (array 274476, no new compute)

```
   arm         hours/run   vs uniform          P(arm > uniform)   90% interval
   uniform         1.12       1.0x                    --              --
   per             1.67       1.5x                  0.040        [0.000, 0.120]
   CCE arms   8.20-9.02   7.3-8.1x        best: cce_only 0.800   [0.520, 1.000]
```

`cce_only` at 0.800 is a max over 8 arms at n=5 — not a result. **Uniform is the best arm
and the cheapest.** Uniform beats PER at P = 0.960.

And the shape of the cost:

```
   64 states x 15 actions x 20 rollouts x 2,535 scoring calls x 60 steps
     =  2,919,920,792  IMAGINED env steps
   vs        10,138,614  REAL env steps
     =  288 imagined steps per real step
```

Jeremy 08/13 asked for P(method > baseline) with one-tailed 90% confidence rather than
two-tailed p-values. Both interval widths are now reported. Jeremy 07/10 asked for the
compute-time analysis; the totals are measured, and the scoring-vs-training SPLIT is
explicitly NOT available because the JaxNav trainer records only `total` and `eval`.

## Beat 6 — What the instruments now let us see at all

`fig_instrumentation.png` · `fig_learning_curves.png`

Six quantities that did not exist before 2026-09-03: coverage, score staleness, the stored
score histogram (issue #8), rollout terminal reasons, eval crash/timeout rates, and
length-conditional-on-outcome (Jeremy 08/27).

Two things they immediately found, both new:

**The rollouts model a policy the agent does not have.** `cf_rollout_temperature=0.5`
(softmax) for the rollouts, `argmax` for evaluation. While the agent is bad the two behave
alike; once it is competent they diverge hard:

```
                  imagined      real
   crash            57%          11%
   goal             18%          45%
```

CCE scores counterfactuals under a policy that crashes 5x more and reaches the goal 2.5x
less than the one being trained, **and the error grows as the agent improves.** Not tested
past 178k; hold until the run lands.

**`cce_max` gives back the most of its peak — but this is weaker than it first looked.**
RETRACTED AND RESTATED 2026-09-08 on the finished 250k data. At the mid-flight 178k read
it was 3/3 seeds dropping ~55pp with nothing else above 9pp, which I called the one
outcome claim that survives n=3. On the finished run:

```
   arm            median drop     per-seed
   cce_max          51.6pp        [-1.6, 62.5, 51.6]   <- one seed does NOT collapse
   per              27.6pp        [30.6, 10.2, 27.6]   <- was 4.6pp at the 178k cutoff
   cce_only         15.2pp        [33.6, 15.2,  5.0]
   cce_wmean        12.1pp        [12.1, 50.5,  8.4]
   cce_wmean_bs      8.7pp        [ 6.3,  8.7, 17.7]
   uniform           4.9pp        [ 2.5,  4.9,  8.7]
```

The ORDERING survives — cce_max worst, uniform best — but the gap to the next arm is 1.9x,
not 6x, and the within-arm seed spread swamps it exactly like every other arm in that
figure. It is not an exception to the n=3 rule. It is still worth mentioning against the
08/20 decision to move Mean to the appendix, as a caution rather than as evidence.

**Second time a mid-flight read flattered a claim** (the first was the rollout/reality
crash gap). Do not quote outcome numbers off a running sweep.

## Beat 7 — The honest bottom line, and the next move

```
   JaxNav is settled as a NULL and the reason is now arithmetic, not statistical.
   Paired seed sd is 30pp; a 5pp effect needs ~290 seed-pairs. Do not add seeds.
```

The next experiment is the **coverage sweep**, and this session changes its design:

```
   at fixed compute:  coverage / cost ~ 1/(rollouts x horizon)
   20x60 -> 5x20 buys 1.6% -> 19% for free

   BUT: raising coverage alone slides along the top edge of the funnel --
        20x more measurements, still 68% of them ZERO.
   SO:  run it on FrozenLake too, where the zero-rate is already 0
        and there is a positive control.
```

Open, filed, not built: **#16** (`normalise_priorities` should use rank, not min-max —
min-max is a near-no-op because rescaling `td` factors out of `(td+eps)**beta` except
through the eps floor) and the **initialization** question (PER hands a new entry
`max_priority` two lines above in the same function; CCE hands it the mean — and no
constant fixes it, since any constant self-averages the same way).

## Figure order for the meeting

```
   1  fig_bootstrap_curves        the problem: a real fix, no effect
   2  fig_why_null                the mechanism: score moved, replay didn't
   3  fig_signal_funnel           THE SLIDE: 100,000 -> 887 -> 283
   4  fig_score_ridgeline         what the other 99% carries
   5  fig_bootstrap_side_by_side  why no score fix can help
   6  fig_instrumentation         what we can now measure
   7  fig_learning_curves         and why the curves are not the result
      prob_and_cost.txt           the two analyses owed since July
```

Regenerate 3-7 with `bash slurm/refresh_figures.sh`. Do **not** show
`MOCKUP_planned_figures.png` — synthetic data.
