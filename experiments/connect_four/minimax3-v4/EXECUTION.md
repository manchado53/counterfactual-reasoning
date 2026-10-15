# Minimax-3 transfer launch

The user approved the full 15-run launch without a learning pilot. Training uses depth-3 minimax with uniform random choice among tied best legal moves, in both roles. The main score uses the frozen bank's 200 depth-3 cases; the other 800 cases measure transfer across opponent depths. All 1,000 cases remain fixed across checkpoints, arms, and seeds.

CPU validation: 26 tests passed (14 v4, 9 shared replay/target, 3 registry/source checks). Coverage includes legal actions, stochastic CCE, collector carry, decay boundaries, bank identity, RNG isolation, and matched configurations.

GPU smoke array 284008 passed on tested code commit bef4b283f8d73b8f74d54287e31dcdd8ab83c497: Uniform task 0, PER task 5, CCE task 10. Each ran 21,504 decisions and 5,376 updates. Each matched the independent CPU minimax oracle on 51 boards, preserved evaluation RNG, changed network weights with finite diagnostics, and passed checkpoint reload. Initial evaluation outcomes and first training collections matched across arms. CCE scored all new transitions before replay. These are runtime checks, not learning results.

The bank SHA256 is 15921cffa6df3e13751a1303bf0d02ce17213d60022651ecd7ca0f3240976cb2. Smoke gate evidence is saved in ignored runs/smoke-gate.json. Full submission metadata and monitor output are saved under runs/ after submission.

The full array contains 5 matched seeds for each of Uniform, PER, and ranked-CCE decay. Limit: 5 active experiment GPUs, leaving room for the 3 active DoorKey jobs within an 8-job total. sbatch --test-only must pass before full submission. Old incorrect Connect Four arrays 283882 and 283926 were stopped; their records were retained. DoorKey jobs were not changed.

New assumptions: four stochastic counterfactual paths per action, common random numbers across alternate actions, and scoring chunks of 64. Four paths are a fixed practical choice, not a tuned optimum. The main depth-3 subset has 200 cases, not 1,000 independent depth-3 games. Shared openings and opponent rules also mean cases are not independent training replicates; the five training seeds are the replication unit.

Warm smoke blocks took 9.5 seconds for Uniform, 9.8 for PER, and 22.5 for CCE. A simple 700-block extrapolation gives about 1.9 hours per control run and 4.4 hours per CCE run before full-buffer effects. Allow roughly 10–20 hours for the complete array after GPUs become available; update this estimate from full runs. Each run has a 48-hour scheduler limit.
