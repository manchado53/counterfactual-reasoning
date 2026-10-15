# Correct the training foe

User approved depth-3 minimax training and the existing fixed tests, and explicitly
asked to skip the learning pilot. Stop only the superseded Connect Four arrays
283882 and 283926; retain their artifacts and preserve every DoorKey job.

Train all three replay arms against depth-3 minimax with uniform random ties.
Use the same window heuristic as the frozen evaluation opponents. Verify GPU
search values against the independent CPU minimax implementation before launch.
Keep both roles, the existing CNN, optimizer, budget, collector carry, replay
formula, and evaluation points. The 200 depth-3 cases are the primary test;
report the other depths separately. Keep the original 1,000-case bank unchanged.

CCE now estimates distributions from four stochastic paths per legal action.
Use empirical Wasserstein distance on sorted return samples, retaining legal
softmax weights and the ranked CCE-to-ranked-TD schedule. Use common random keys
across counterfactual actions, independent keys across the four paths, and a
scoring RNG separate from collection/evaluation. Four paths is an explicit new
sampling choice, not a validated optimum. Chunk size is a memory/runtime choice.

Run CPU tests and three full-bank GPU smoke checks, not a learning pilot.
Then submit all 15 runs after sbatch --test-only. Cap new jobs at five while
three DoorKey GPU jobs exist; do not alter them. Start a read-only monitor,
push the branch, and open a PR. Report real timings and any runtime limitations.
