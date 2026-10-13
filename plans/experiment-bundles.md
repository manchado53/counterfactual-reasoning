# Experiment bundles

Replace the vague experiment-code package with three clear homes. Shared CCE
math stays in `src/counterfactual_rl`. Reusable execution code moves to
`training`, and reusable result-reading code stays in `analysis`. Actual research
questions become self-contained folders under `experiments/<environment>/<name>`
with a short README plus training and analysis recipes.

Move every current recipe into this shape, update commands, Slurm launchers,
tests, docs, packaging, and source capture, then run the full trust gate. Preserve
old learner variants under each environment's `legacy` folder so past recipes
remain runnable. Do not move generated run data into Git.
