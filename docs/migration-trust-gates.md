# Migration trust gates

Checked on 2026-10-12 before merging PR #30.

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Clean Python 3.12 install | PASS | `pip install -e '.[research,dev]'` in a new virtual environment |
| Full test suite | PASS | 116 passed, 2 skipped |
| Recipe catalog | PASS | 24 training recipes and 10 study configs across 9 environment adapters |
| Real bounded environment smokes | PASS | 9 of 9 completed on CPU |
| JaxNav frozen-function and recipe tests | PASS | 24 passed |
| JaxNav replay and decay parity | PASS | 45 cases, 144 exact comparisons each, 6,480 total |
| Full-size JaxNav GPU parity | NUMERIC MATCH; INCOMPLETE | Two candidate tasks matched; a third stopped at tiny old-versus-old float noise before its candidate ran |
| Experiment-bundle structure | PASS | Clean install, catalog, 116 tests, and all 9 environment smokes rerun after the move |

## Environment smokes

Each row used the normal training entry point, now `python -m training.run train`, with
`--smoke`, a CPU device, and a four-minute wall limit.

| Environment | Recipe | Task | Result |
| --- | --- | ---: | --- |
| JaxNav | `jaxnav/ranked-td-decay/training.json` | 13 | completed |
| FrozenLake | `frozenlake/smoke/training.json` | 2 | completed |
| DoorKey | `doorkey/fixed-transfer/training.json` | 4 | completed |
| Connect Four | `connect_four/fixed-transfer/training.json` | 4 | completed |
| SMAX | `smax/historical/training.json` | 3 | completed |
| Chess | `chess/historical/training.json` | 3 | completed |
| CVRP | `cvrp/historical/training.json` | 3 | completed |
| RP-DQN | `rpdqn/historical/training.json` | 3 | completed |
| CVRP budget | `cvrp_budget/budget/training.json` | 0 | completed |

The original run manifests are under `runs/trust-gates/20261012/`. The same nine
smokes were rerun after the structure move under
`runs/trust-gates/20261012-structure/`. Run data is ignored by Git. This page is
the tracked result.

## Parity boundary

The retained replay-and-decay report passed for old and fixed collectors, all
three CCE arms, decay on and off, and seeds 0 through 4.

Under the approved `1e-7` raw-score tolerance, tasks 1 and 7 matched the
candidate. Task 13 stopped at the old-versus-old repeatability check before its
candidate ran. Its largest raw CCE-score difference was
`5.090493640567788e-08`, which is below that tolerance. This is float noise, not
evidence of a behavior change. The full-size gate is still incomplete because
task 13 never reached its candidate comparison and the full production budget
was not replayed.

The detailed migration workspace is preserved at commit `d0c2945` and tag
`migration-audit-20261008`.
