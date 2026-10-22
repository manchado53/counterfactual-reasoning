# PER positive-control reproduction

Reproduce a published environment where proportional PER beats uniform replay before judging the repository's past Uniform-versus-PER gaps. Start with the ReaPER paper's CartPole-v1 DDQN protocol because it is cheap, uses 20 matched runs, and holds the learner settings fixed across Uniform and canonical PER.

Build the CartPole study through this repository's shared TD and experiment tooling. Freeze the paper's environment, network, optimizer, exploration, replay, update, evaluation, stopping, and seed protocol. Run three clearly named arms: Uniform, the paper's canonical proportional PER, and the repository's current proportional-PER rule under the same CartPole learner. Tests must prove each arm's probabilities, importance weights, schedules, new-entry rule, TD targets, and loss path before any full run.

Report full curves, failures that never reach the reward threshold, median steps to threshold, uncertainty, wall time, and exact provenance. Treat CartPole as the first gate. If it reproduces the known PER direction, add the original paper's Blind Cliffwalk mechanism study. Do not start Atari until both cheap gates pass; Asterix and Space Invaders are later candidates with multi-billion-frame full protocols.
