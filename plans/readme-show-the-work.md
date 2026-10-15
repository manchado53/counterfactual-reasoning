# README: show the work behind each step

Rewrite the README so a reader sees the method, the real math, the real results,
and what didn't work — same recipe as AI-Driven-Soccer-Analysis and revit-to-robot.
The project is still running, so the README says "in progress" and states only
claims backed by real runs.

## What changes

- **Method section with real math.** Return distribution, CCE score, PER priority,
  additive/multiplicative mixing, IS weight, TD target — copied from `paper/paper.tex`,
  each next to a diagram from `docs/figures/diagrams/` (all real, none used today).
- **Results, honest.** Claim 1 (CCE tracks the exact oracle: FrozenLake rho
  0.32 → 0.77 → 0.89, CVRP 0.58 → 0.77 → 0.74) and Claim 2 (FrozenLake
  deterministic 20/25 vs 12/25 vs 10/25 seeds; JaxNav mid-training lead only).
- **"What didn't work."** Stochastic FrozenLake tie, graded slip (win only at
  slip 0), CVRP null (~2,000 runs), DoorKey saturation, JaxNav lead fading by 500k
  plus the obstacle-fill bug, chess/Connect Four dropped, 2–3x wall-clock cost.
- **Effort, in numbers.** ~6,600 CCE cluster jobs, ~12.6k GPU-hours on Rosie, Jan–Oct 2026.
- **Known caveats, stated plainly.** Historical "weighted_mean" ran as `max`
  (result holds under `mean`); PER inserted at priority 1.0 until PR #34; current
  recipe differs from the paper's.
- **Shorter how-to.** CLI usage moves to `docs/USAGE.md`; README keeps a short quickstart.

## Decisions (2026-10-15)

- **One story:** idea → FrozenLake win → Claim 1 holds → win fades on harder
  envs → what we learned.
- **Branch figures:** pull in ~6–8 of the best (JaxNav collapse / why-null,
  graded-slip curves, DoorKey, CVRP Claim 2), each with a provenance line.
- **Caveats:** short box stating the `max` and PER-insert bugs.
- **Visibility:** repo stays private for now; make public after the README is done
  and Jeremy agrees, then share with Nick.

## Guard rails

- Never use mock figures (`docs/figures/fig1-9_*`, mock `claim1/*`) or the broken
  `fig_c1_map_cvrp.png`.
- Keep out: the co-authored theory PDF and its theorems, chat/session IDs, cluster
  node names, third-party figures, anything Direct Supply.
- Figures from unmerged branches get copied in only with a provenance line.
