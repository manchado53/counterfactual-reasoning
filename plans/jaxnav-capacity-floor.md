# JaxNav capacity floor — where does a small buffer actually break?

## Why

Every small-buffer run we have ever done was a CCE run:

```
  consequence-dqn      500 /  1,000 /  2,500 /  8,192 / 10,000 / 25,000 / 100,000
  dqn-uniform          nothing below 100,000
  dqn (PER)            nothing below 100,000
```

So the lab-notebook's "32 runs, all collapse, DO NOT shrink the buffer" is confounded.
We never asked whether plain replay collapses at 8,192 too. If it does, small buffers are
simply bad on JaxNav and CCE is innocent. If it does not, then shrinking hurts *only when
you prioritise*, which would explain all 32 runs and is a finding in its own right.

There is also a 12x hole in the ladder: we have 8,192 and 100,000 and nothing between.

## Why it matters now

The next CCE design (score-on-insert) reaches 100% coverage at a cost that does not depend
on capacity — so capacity stops buying coverage. What it still buys is *freshness*:

```
  capacity 100,000  ->  mean staleness 3,125 gradient updates
  capacity  25,000  ->  mean staleness   781        (today's measured value is 1,788)
```

So a smaller buffer would let score-on-insert land at 100% coverage *and* fresher scores
than we have now — but only if learning survives down there. This sweep is the gate.

## What we run

A capacity ladder, no CCE at all. Six capacities x two algorithms x five seeds = 60 runs.

```
  capacity    bursts held    why this rung
     8,192        1.0        hard floor: below it, data is overwritten unread
    12,288        1.5        first rung above the floor
    16,384        2.0        power-of-two step
    25,000        3.1        the staleness target for score-on-insert
    50,000        6.1        halfway to today
   100,000       12.2        today's default, re-run in-sweep as the anchor
```

`uniform` isolates the diversity problem on its own. `PER` adds prioritised sampling, which
is what CCE actually does, so it is the arm that can show a prioritise-times-small-buffer
interaction. PER is the weaker baseline on JaxNav (30.8% vs uniform 62.8% at 100,000), so
it has less room to fall — that is why both arms run, not just PER.

100,000 is included even though we already have 21 uniform and 13 PER runs there, because
those came from other sweeps with other RNG streams. An in-sweep anchor makes the ladder an
internal comparison.

## Cost

~90 GPU-h, roughly 4 hours wall at 30 concurrent. That is a rounding error next to the
530 GPU-h score-on-insert plan it gates.

## What each outcome means

- **Both collapse** — small buffers are just bad here. The notebook's rule is right and now
  uncontaminated. Score-on-insert runs at 100,000 and accepts staleness 3,125.
- **Only PER collapses** — prioritising out of a small buffer is the killer. Explains all 32
  CCE runs, and it is a real result.
- **Neither collapses** — the floor is lower than believed. Score-on-insert can run at
  25,000 and get 100% coverage with better freshness than today.

All three are useful, which is unusual.

## Read-out

Bimodal seeds, so report the fraction of seeds reaching the good mode, plus the median and
the per-seed values. Not a mean. Compare arms only at matched episode counts, and never
take a final number from a run that has not finished — the fast seeds are the good seeds.
