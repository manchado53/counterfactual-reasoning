# Connect Four: fixed tests across training

Approved October 14, 2026. This extends the v2 replay experiment with the
agreed evaluation bank. The earlier jobs and their source stay untouched.

| Suite | Starts | Opponents | Roles | Cases |
|---|---|---|---|---|
| Empty | Empty board, 50 case seeds per cell | Minimax depths 1–5 | Both | 500 |
| Opening | 50 unique legal four-ply boards | Minimax depths 1–5 | Both | 500 |

Every checkpoint, arm, and training seed sees the identical bank. Its canonical
SHA-256 is checked at startup and each evaluation, and saved with every result
and checkpoint. Per-case moves and outcomes are saved in compressed files.
The bank is committed, never regenerated during a run.

Bank SHA-256: `15921cffa6df3e13751a1303bf0d02ce17213d60022651ecd7ca0f3240976cb2`.
The same policy gives identical full paths and outcomes. Different policies may
change moves; the foe reacts to the resulting board. Ties use only the case seed
and absolute board ply, never training/checkpoint/batch identity or global RNG.

Minimax uses a fixed integer window heuristic: unopposed windows with one, two,
three stones score 1, 4, 16; center stones score 3. Values are symmetric for both
players. Terminal wins dominate the heuristic. Depth is individual board moves.
Root moves are scored with full bounds, so pruning cannot produce false root
ties. The foe samples uniformly among best-scoring legal root moves. These are
limited-search estimates, not proof of optimal play. Depth is not a guarantee
that one foe beats every shallower foe on every board.

Report the two suites separately: wins/draws/losses, win rate W/N, score
(W+0.5D)/N, and role/depth cells. Log actual distinct paths. There are 51 distinct
board starts across 1,000 cases, not 1,000 unique boards. Different seeds need not
produce distinct games. Generalization is scoped to this frozen bank. Compare
replay arms using five matched training seeds, not games/checkpoints as extra
independent trained models. No stopping or model selection based on test scores.

## Training controls retained from v2

Uniform, raw-TD PER, and ranked CCE-to-ranked-TD decay; five seeds per arm.
The fixed training foe is still win/block/center, with deterministic tie order.
The minimax opponents are test foes only. This isolates the test protocol change.
Connect Four CNN, Adam .001, discount .999, 7,526,400 learner decisions,
evaluation at zero/every 107,520 decisions/end, no early stop.
Score each CCE arrival before replay; rank scores; fade between 100k and 150k
completed games; preserve collection carry and all legal masks.
See ../jaxnav-transfer-v2/README.md and the committed plan for full controls.

## Source precedents and our choices

[Thill et al. 2012, section 2.4](https://www.researchgate.net/publication/235219697_Reinforcement_Learning_with_N-tuples_on_the_Game_Connect-4)
uses random choices among optimal foe moves to avoid identical games.
[Bjornsson et al. 2023, sections 4.2 and 4.6.1](https://iris.ru.is/ws/files/217767220/FAIA-372-FAIA230279.pdf)
uses 25 four-ply openings played from both sides.
Our 50 openings, 500/500 split, depths 1–5, heuristic, seed schedule, and fixed
bank hash are explicit choices, not an exact reproduction of either paper.

## Runtime and launch

Search uses isolated pinned numba 0.65.0 and llvmlite 0.47.0 under .runtime/.
The shared environment used by running jobs is not changed. Recreate with:

```sh
PYTHONNOUSERSITE=1 /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python -m pip install --no-deps --target .runtime numba==0.65.0 llvmlite==0.47.0
```

The job sets PYTHONPATH to this local runtime. Source/package hashes are saved
by the common runner. Run CPU tests, then full-network/full-bank GPU smokes for
tasks 0, 5, 10. Use sbatch --test-only before submission. Submit all 15 tasks
only after all smokes pass. Keep combined active old/new experiment jobs at
most eight, preserving DoorKey and v2. Start training/monitor_connect_four_v3.py.
Smoke measurements are runtime/validity checks, never experimental results.
