# Historical analysis drivers

33 tests pass. Four full-driver smoke commands pass on real inputs.
No learner training or full study sweep ran.

| Driver | What ran | Limit |
|---|---|---|
| Claim 2 raw rebuild | Real job 257499, saved manifest and full log header | Full125 check fails on missing257574 |
| Graded raw rebuild | Real job 269926, completion filter, per-slip cache | Full200 not run |
| Covariance | Real seed0/trained checkpoint, all636 transitions, gradient JVP and SGD check | Two rollouts; one checkpoint; saved horizon retained |
| Suitability | Real checkpoint, returns, six metrics, visits, horizon check, evaluation, scorecard | Two rollouts; horizon2; one visit/eval episode |

The earlier 11 analysis configs and 19 plots remain recorded in verification.json.
Their evidence is separate from these real-input driver checks.

## Run

The public entry point is experiments.studies.run.
Internal copied source modules preserve old driver code for comparison.
Use the public entry point to get explicit paths and the wall-time cap.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=.:src \
  /home/ad.msoe.edu/manchadoa/.conda/envs/counterfactual/bin/python \
  -m experiments.studies.run \
  --config configs/studies/covariance-checkpoints.json \
  --data-root /path/to/repo-data --out /tmp/new-covariance-output --smoke
```

Other configs:
- configs/studies/claim2-rebuild.json
- configs/studies/graded-slip-rebuild.json
- configs/studies/suitability-rollouts.json

All output dirs must be fresh.
The CLI caps CPU affinity at four cores and wall time at180 seconds.
--max-wall may lower that cap. Timeout kills the child process group and returns124.
--validate-inputs checks exact paths before costly math; raw rebuild also checks headers.
--smoke uses explicit smoke jobs or the first listed checkpoint.
No synthetic input is substituted for missing real data.

--data-root wins. Else config.data_root is resolved from the config file location.
The installed module location is never used as a data root.
The library hook is run(config, fresh_out, data_root=explicit_root).
Main owns any shared runner or recipe integration; those files were not edited here.

## Proof and source records

- catalog.json: current code/evidence status and gaps per family.
- full-driver-verification.json: four real-input smoke reports, artifact hashes, 33 tests.
- full-driver-sources.json: preserved full source modules and import rewrites.
- checkpoint-loader-source.json: exact strict loader methods from the theorem branch.
- source-report.json / evidence-sources.json: earlier sources and paper provenance.
- manifest-coverage.json: 39 manifest variants linked into main's full discovery.

Each successful run reports data hashes and package versions.
It also hashes imported project sources and writes project-sources.tar.gz.
The entry-point source is included even when run as __main__.
These are current source bytes, not proof of the original historical launch source.

Final outputs: /tmp/historical-full-final-vaf3az3b.
Report JSON and artifact hashes are retained in full-driver-verification.json.
Temp files can expire; configs rebuild them from supplied data.

The checks compare original function bodies and execute saved math functions against
new-path functions on the same real inputs and current dependencies.
This proves reduced source equivalence, not a full old-environment reproduction.

## Exact remaining gaps

Full no-slip input validation stops on:
src/counterfactual_rl/agents/frozen_lake/runs/257574/metrics.log
under the main repo data root.
The driver does not silently omit this seed.

The raw driver keeps the source's step clock and last-value padding.
historical_all is explicit; complete_only uses the source's completion rule.
A rebuilt cache is not a training port.
The no-slip125, stochastic50 and graded200 manifests are override rows.
Main owns logged training-recipe recovery and missing May slip_prob proof.

Covariance full stage/seed sweeps and aggregate theorem figures have not run.
Suitability's FL driver is runnable; C4/SMAX, dashboard injection and realized
replay-sampling CLI wiring remain incomplete.
Chess real-data orchestration and the older PPO/SMAC notebook paths remain partial.
The bootstrap source still gets no explicit RNG object; bitwise CI replay is unproven.
Historical max fallback and undefined-correlation warnings remain visible.
Conda emits a CUDA plugin warning; all these runs execute on CPU.

## Legacy worker handoff

RPDQN learner source lives under:
../cce-rpdqn/src/counterfactual_rl/agents/cvrp_neural/
The key files are dqn.py, consequence_dqn.py, consequence_dqn_vectorized.py,
and eval_testset.py. Several learner files were dirty at the initial audit.
The legacy worker must preserve those dirty bytes before a learner port.
This worker changes only RPDQN analysis; its dirty plot_training.py is snapshotted.

Main's discovery.json remains the shared46-ref/23-tree/22-worktree index.
No second full discovery scan was done.
Main reported T4 gate281087 failed oldold task13; two of nine passed.
These analysis checks do not change that gate or its tolerance.
