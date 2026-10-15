# Depth-3 minimax replay comparison

Train Uniform, PER and CCE on a fixed depth-3 minimax policy, with uniform random
choice among tied best legal moves and both learner roles. Five matched seeds per
arm. This replaces the mistaken rule-bot training in arrays 283882 and 283926.
The user approved launch without a learning pilot. CPU and GPU smoke checks
remain required; they are code/runtime checks, not model-selection trials.

Use the exact same heuristic and depth unit as the frozen CPU evaluation foe.
GPU training uses 64-bit bitboards, verified against independent CPU alpha-beta
root scores. Each reply consumes an explicit collection key. No global RNG.
Network inputs, parameters, loss weights and update calculations stay float32.
Random tie selection has the same uniform law in training and evaluation, but
uses separate JAX and saved NumPy seed streams respectively.

Keep the v3 bank byte-for-byte: SHA-256
15921cffa6df3e13751a1303bf0d02ce17213d60022651ecd7ca0f3240976cb2.
The 200 depth-3 cases are primary: 100 empty-board and 100 opening-board games,
each balanced across roles. Report each suite too. Depths 1,2,4,5 are separate
transfer tests. The full bank remains 1,000 cases and 51 distinct board starts.
Same policy/actions and case seeds imply identical full evaluation games at all
checkpoints. Never count duplicate game paths as independent training runs.

CCE uses four stochastic return samples for each legal action, with independent
keys across the four paths and shared keys across counterfactual actions.
Sort each action's four returns and average absolute paired differences to
estimate Wasserstein-1 distance. Keep the source legal softmax action weighting,
rank normalization, multiplicative CCE/TD priorities, score-before-draw, and
100k–150k-game probability fade to ranked TD. No score refresh. Four paths is a
new declared sampling choice, not a claim that four is optimal or sufficient for
all stochastic settings. Score chunks of 64 bound GPU memory.

All other v2 controls remain: Connect Four CNN 32/64/32 plus 256/128 dense,
Adam .001, discount .999, replay 100k, batch 256, update each four decisions,
256 collectors, 42 steps per block, 700 blocks (7,526,400 decisions), test at zero,
every ten blocks, and end. No early stopping. One decision is a learner move plus
the foe reply. This study is not an exact paper reproduction.

Job runtime uses isolated pinned numba/llvmlite from .runtime, as in v3.
Submit three full-bank GPU smokes before the 15-run full array. Use sbatch
--test-only before all submissions. Preserve DoorKey jobs. Start the monitor
training/monitor_connect_four_v4.py after submission. Save all source hashes,
configs, bank copies, evaluation paths and outcomes through the shared runner.
