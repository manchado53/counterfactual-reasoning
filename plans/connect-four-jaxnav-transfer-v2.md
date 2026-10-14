# Connect Four replay transfer v2

Plan for approval. No training code changed or jobs submitted yet.

Test whether the working JaxNav replay design helps on Connect Four.
Use Uniform, raw-TD PER, and multiplicative CCE with ranked-TD decay.
Use seeds 0–4 for each arm: 15 runs in all.
Only replay differs between arms. Keep DoorKey jobs and files untouched.

## Base and audit

The remote default branch is `master`; there is no remote `main`.
This branch, `experiment/connect-four-jaxnav-transfer-v2`, starts from freshly
fetched `5f2d313edf9eb4dc934a8a1d6461f47d5fe3f940` in its own worktree.
The old checkout has merge conflicts and is left alone.

Read `experiments/jaxnav/ranked-td-decay`, the JaxNav build hook and shared
`replay/decay.py`, plus the Connect Four learner and `fixed-transfer` recipe.
The old Connect Four evaluator explicitly repeats two paths 500 times each.
Its runner has no replay decay, only seeds 0–1, and a copied JaxNav MLP.
Its carry and legal masks are useful, but need regression tests.

DoorKey's `vlmrb-paper-parity` is absent from this base. Read its README and
launcher in `.codex/worktrees/doorkey-paper-parity` as a read-only reference.
Its 12x12 grid CNN, learning rate 4e-5, discount .95 and step cap do not transfer.

JaxNav fixed/multiplicative/decay full-run completion records were found for
tasks 13, 15, 17, 41 and 59 in the two `decay_20261007_v1*` work directories.
They reach at least 200k games, record zero terminal-start ghosts and all
three blend stages. This verifies completion and the working path, not that
multiplicative CCE beats every other JaxNav arm. Choosing it for this single
CCE arm is an explicit assumption; do not select it using Connect Four results.

## What carries over

Score all arrivals before any draw, including after the fade ends.
Use legal alternatives only, Wasserstein return distance, weighted mean,
one greedy rollout per action, rollout discount .99, horizon 60,
action-weight temperature .5, and terminal-safe bootstrap.
Keep score-on-insert with no random refresh. Score in chunks of at most 1280;
smaller chunks may be used for GPU memory without changing the score.

Use the shared JaxNav DecayBuffer. Start with multiplicative ranked CCE and
ranked TD, exponents 1 and 1. Keep replay epsilon .01, exponent beta .25,
initial maximum priority 1, and the source importance-weight behavior.
Raw-TD PER stays raw TD; it is distinct from the CCE arm's ranked-TD endpoint.

For completed training games e:

    fade = clip((e - 100000) / 50000, 0, 1)
    draw probability = (1 - fade) * CCE-mix probability
                       + fade * ranked-TD probability

Do not fade raw scores or switch to raw-TD PER. Apply the fade before each
update, as in JaxNav. Preserve state, pending reset, player role, return and
length across collection batches. Keep training, replay, scoring and evaluation
random streams separate so extra scoring or evaluation cannot shift exploration.

## What stays Connect Four specific

Use the existing Connect4QNetwork: board shape 6x7x2; convolutions 32/64/32
with kernels 5/5/1; dense layers 256/128; seven outputs; LeakyReLU; no LayerNorm.
Adapt its output shape only. Use the local Connect Four defaults: Adam .001,
discount .999, batch 256, replay size 100000, one update per four learner
decisions, target copy each 1000 decisions, gradient clip 10.
One decision means a learner move plus the opponent reply if the game is live.
Keep terminal rewards from the learner's player ID, not hard-coded player zero.

Use 256 environments, 42 decisions per collection batch, and 700 batches:
7,526,400 decisions and 1,881,600 scheduled update attempts per arm.
Epsilon goes from 1 to .05 across the first half of that decision budget.
This retains the local Connect Four budget; it does not use JaxNav's episode
budget. Log game counts and fade so the actual transition can be checked.
No early stopping. Evaluate at step zero, every ten batches (107,520 decisions),
and the end. All arms use these exact points and the same update schedule.

Keep the old transfer's fixed training opponent, win/block/center with column
tie order [3,2,4,1,5,0,6], as a one-policy training bank shared by all runs.
Keep both roles. Freeze and hash this training-bank spec. This is an explicit
choice to isolate replay and keep the source one-rollout CCE score well defined.

## A real fixed test bank

Build and commit 1000 legal, nonterminal opening cases once, before training.
Use 500 per learner role and five fixed opponent rules, 100 per role/rule cell:
seeded random legal moves, center-first, win-first, win/block/center, and
win/block with a fixed alternate column order. These are policy variants;
do not claim a proven strength order.

Use varied openings of 4–12 plies (a ply is one player's move), with the
right parity for the learner's first/second role and the learner to move.
Reject duplicate board-plus-side-to-move positions, not just duplicate move
lists. Reject wins, illegal moves and full boards. Persist opening moves,
board identity, role, opponent rule/version and per-case random seed.
Random-opponent draws use the case seed and ply, never the training RNG.
Use disjoint bank seeds from training and verify no duplicate start states.
The bank tests play from these starts, not full games from empty boards.

Hash canonical bank bytes with SHA-256; verify that hash at run start and
each evaluation. Store it in configs, logs and checkpoints. Report win/draw/loss
by role and opponent, plus the overall mean. Track unique starts and actual
trajectory counts; do not confuse 1000 fixed cases with 1000 independent seeds.
Use paired differences across five training seeds for claims. Never count
repeated checkpoints or matching game paths as extra independent trials.

## Checks and launch gate

Implement a new v2 protocol and recipe while retaining the old recipe's meaning.
Test legal actions in collection, targets, rollouts and evaluation; player
reward signs; carry across split batches; tactical CCE scores, single-legal-move
zero scores and score-before-draw; exact decay boundaries and weight/cache
behavior; bank legality, uniqueness, hash and role balance; evaluation identity
after reload; all RNG streams unchanged by evaluation; and matched configs.

Log Q values, loss, absolute TD error, CCE score distribution and age, scored
fraction, actual replay-draw coverage, ESS (effective sample size), top-1-percent
priority mass, maximum draw probability, fade, steps, games and wall time.

Run CPU checks first. Run sbatch --test-only before each submission, then GPU
smokes for all three arms using the full network and all 1000 evaluation cases.
Force boundary checks within a separate smoke fixture; do not change production
timing. Require finite updates, changed weights, full scoring coverage, stable
bank identity, reload parity and a real CUDA backend. Smoke is only a code gate.

If all pass, freeze source/config/package hashes and submit 15 tasks with an
array cap of eight. Use a stricter temporary cap if needed to keep total active
user GPU jobs at eight while DoorKey runs. Never cancel or change DoorKey jobs.
Check account/QOS and actual scheduler state after submission. Start a persistent
monitor immediately after smoke and full submission. Monitor only these new jobs.
Derive the runtime estimate from warm GPU timings, including scoring, evaluation
and checkpoint cost; report queue time separately as unknown.

Commit and push this branch, verify the remote commit, and open a PR against
master. Report tests, source commit, smoke IDs, full array ID, active/queued
tasks, timing estimate and these assumptions. Full runs may still be active;
report their real state without calling a smoke a result.
