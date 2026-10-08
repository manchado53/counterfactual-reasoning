# Rebuild the migration runtime

Tested platform: Linux x86_64, Python 3.12.12. The migration CLI is installed as
`cce-run`; neither `PYTHONPATH` nor a private worktree is needed at run time.
The recipes and analysis scripts remain research files in the checkout.

## Historical dependency exception (not a resolver-clean stack)

The saved JaxNav runs used JAX 0.9.1, SciPy 1.16.3, and JaxMARL 0.1.0.
JaxMARL's release metadata declares `jax<=0.4.38` and `scipy<=1.12`.
Changing these versions would change the parity reference. We therefore preserve
the tested historical stack, explicitly install its full pinned dependency list
with `--no-deps`, and retain the two `pip check` conflicts in the check report.
Do not call this a clean dependency-resolution pass. A resolver-compatible modern
stack needs a separate behavior check before replacing this historical runtime.

The installed JaxMARL release's 93 Python files matched the sealed October 7
vendor source byte-for-byte in the initial audit. No simulator fork is needed.
Recheck hashes for each retained run; a package version alone is insufficient.

## Isolated CPU install

Use an empty virtual environment (no system-site packages):

```bash
python3.12 -m venv /path/to/new-venv
/path/to/new-venv/bin/python -m pip install --no-deps -r requirements/migration-cpu.lock
/path/to/new-venv/bin/python -m pip install --no-deps --no-build-isolation .
/path/to/new-venv/bin/cce-run validate \
  --recipe configs/experiment/jaxnav_decay_20261007.json --task 13
/path/to/new-venv/bin/cce-run train \
  --recipe configs/experiment/jaxnav_decay_20261007.json --task 13 \
  --smoke --device cpu --max-wall-seconds 180 --output /path/to/new-attempt
/path/to/new-venv/bin/python analysis/plot_migrated_runs.py \
  --runs /path/to/new-attempt --bin-size 1 --output /path/to/new-figure.png
```

The smoke is six training episodes and two eval maps, NOT a research result or the
1,000-map GPU gate. A run saves config, source bundle, simulator hashes, eval-bank
identity, logs, and completion/failure state. Exact resume remains unsupported.

Core installation without research extras does not install every simulator or
PyTorch. Old PyTorch callers can request the `legacy` extra. The `research` extra
provides the main learner libraries, but the pinned historical JaxNav stack uses
the lock above because of the declared JaxMARL conflict.

For a repeatable installed-package check, run the probe from outside the checkout:

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /path/to/new-venv/bin/python -m experiments.install_probe \
  --recipe /path/to/checkout/configs/experiment/jaxnav_decay_20261007.json --task 13 \
  --plot-script /path/to/checkout/analysis/plot_migrated_runs.py \
  --output /path/to/new-probe
```

It rejects source-checkout imports and system-site packages, runs two bounded
smokes, checks complete source bundles, rebuilds figures, and compares scientific
data. Each training subprocess has a hard 180-second timeout. It expects exactly
the two known JaxMARL metadata conflicts and fails on any changed exception list.
For FrozenLake use `configs/experiment/frozenlake-smoke.json`, task 2.

## GPU record

`migration-cuda12.lock` records the CUDA packages present in the historical GPU
runtime on top of the CPU lock. It is a reconstruction record, not a claim that a
new GPU environment has passed parity. NVIDIA workers request `JAX_PLATFORMS=cuda`
explicitly. Device model and driver must be recorded with each GPU gate. A T4 pass
does not establish H100 parity, and a bounded prefix is not a full-budget rerun.
