# Shared PER-style study

Set up one comparison protocol for replay methods across environments, inspired by the presentation of results in Schaul et al., *Prioritized Experience Replay* (ICLR 2016). Each environment keeps its own score and evaluation cases, while all methods within that environment use matched learners, training budgets, evaluation checkpoints, and seeds. The shared analysis will check run provenance, keep incomplete runs visible, and produce consistent per-environment learning curves plus a cross-environment summary of early learning, final performance, and cost.

This setup will reuse valid observed runs where their protocols match. It will not treat existing smoke checks, unfinished runs, or unmatched baselines as completed comparisons. New training and paper claims wait for the ongoing PER audit and blend comparison to settle the exact replay arms and controls.
