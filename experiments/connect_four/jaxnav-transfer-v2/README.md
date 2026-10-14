# Connect Four: JaxNav replay transfer v2

Compare Uniform, raw-TD PER, and multiplicative CCE fading to ranked TD.
Five matched seeds (0–4), 15 runs. Only replay differs within each seed.

The CCE sampler is the shared JaxNav `DecayBuffer`: score every arrival before
its first replay draw, rank both signals, blend normalized probabilities from
100k to 150k completed games, then stay at ranked TD. Continue scoring after
150k. One deterministic rollout per legal action; horizon 60, discount .99,
Wasserstein difference, action-weight temperature .5. Invalid moves are excluded.
The score uses the fixed training opponent, not the evaluation opponents.

The learner uses the Connect Four board CNN, LeakyReLU, Adam .001, discount
.999, batch 256, update every four decisions, target copy each 1000 decisions,
and 100k replay slots. Total budget: 700 x 256 x 42 = 7,526,400 learner decisions.
No early stop. Evaluate at zero and every ten blocks, through block 700.
A learner decision is one move plus the opponent reply, when still live.

The training bank is the fixed win/block/center rule with both roles. Evaluation
uses 1000 unique legal start boards, 500 per role, crossed with five fixed foe
rules (100 per cell). Seeded random actions use only the case's own seed.
Starts contain 4–12 played moves; this tests play from openings, not empty-board
win rates. The JSON bank is committed, copied into each run, and SHA-256 checked
at every evaluation. Repeated checkpoints are not independent trials. Report
paired seed-level differences and per-role/per-foe results. Five seeds constrain
uncertainty; smoke results are never evidence that replay improves learning.

`training.json` is the resolved 15-task recipe. `job.sbatch` requires C4_ROOT,
C4_MODE (smoke/train), and C4_COMMIT. Run `sbatch --test-only` before each
submission. Smoke tasks are 0,5,10; each runs two full-size collection blocks,
the full network and all 1000 cases. Require all three completion records before
full submission. Full tasks are 0–14 with at most eight active GPUs, reducing
the cap if other user jobs require it. Start `python -m training.monitor_connect_four_v2` after each submission.
No DoorKey job or source is modified.

Each attempt bundles source and package identity. Progress records include
Q/loss/TD, CCE distribution/age/coverage, live-slot draw coverage, ESS, top-1%
probability mass, fade, step/game counters and wall time. Checkpoints are for
evaluation and explicitly not resumable. Failed or timed-out runs must remain
visible. Training results are pending until full runs actually finish.
