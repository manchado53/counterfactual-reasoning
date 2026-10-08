# Historical JaxNav plots

Two plots now pass an exact data check. No training was run.

```text
saved old logs + old output
             |
       explicit manifest
             |
       new legacy reader
             |
    exact data check -> PNG
```

## Files

- `analysis/historical_reporting.py`: decay reader, exact gate, render, CLI.
- `analysis/freeze_historical_decay.py`: capture old plot data from the old script.
- `analysis/historical_five_arm.py`: five-arm reader, saved-audit capture, render, CLI.
- `tests/test_historical_reporting.py`: public reader and gate tests.
- `analysis/historical_reproduction/decay-20261007-2224-v2/`: frozen decay bundle.
- `analysis/historical_reproduction/five-arm-20261007/`: frozen five-arm bundle.

Each bundle has `manifest.json`, copied old source, old numeric output, and compressed
raw logs. `verified/` has `report.json`, `plotted-data.json`, and `reproduced.png`.
The manifest pins each log's byte count and SHA256. SHA256 means a hash of the file
bytes. Changed bytes fail the check. Log growth cannot extend this saved view.
New readers use the copied logs; they do not need the old worktrees to exist.
The capture tools need the original files. They refuse an existing output folder.

## Exact rules

| Rule | Decay snapshot | Five-arm audit | Existing new-run plotter |
|---|---|---|---|
| Source | Oct 7, 22:24 CDT | Oct 7 audit | selected new run folders |
| Range | 0–200k episodes | 0–250k episodes | observed bins |
| Bins | `[lo, hi)`, width 5k | `(lo, hi]`, width 5k | floor division, width 5k |
| Last bin | keep only if `hi <= last episode` | all bins required, last episode >=250k | partial bin can count |
| X axis | bin center, thousands | bin center, thousands | bin lower edge |
| Seed count | shown seeds, out of 5 planned | exactly seeds 0–4 in each arm | number of selected runs |
| Main line | arithmetic mean | arithmetic mean and median | arithmetic mean and median |
| Band | none | descriptive SEM, sample SD / sqrt(5) | none |
| Thin lines | each shown seed, with its own end | none | none |
| Mean tail | all shown seeds must have the bin | all five seeds must have the bin | shared observed bins |
| Smooth/fill | none | none | none |

SEM is a spread summary. It is not a 95% confidence band.
The protocols stay separate. Neither old plot is routed through `plot_migrated_runs.py`.
No median was added to the decay plot. No source run was extended or replaced.

## Old sources and evidence

The source folder was:
`/home/ad.msoe.edu/manchadoa/.codex/visualizations/2026/09/19/01a0bbe7-7aaa-7202-a8e7-a15007ce9594/`.

The decay source was `decay-training-curves.py`, hash
`942ae0c4bd30e6981011d075319980401c9c7b175efc2cdacaecb8810d38738b`.
Its saved `decay-training-curves-with-baselines-sources.json` names 52 inputs.
All 52 saved byte prefixes were recovered with exact hashes.
The old script was run on those prefixes. Only its final three output calls were
removed, and its display timestamp was restored. Its selection, config checks,
bins, means, labels and plot calls ran unchanged. The capture refuses a different
script hash. Old source was not edited. Old outputs were not overwritten.
The old source's source list, counts, bank hashes and end summaries matched the
saved snapshot. Frozen plot values came from the old Matplotlib line objects.
Selection metadata came from the old manifests and snapshot, not the new reader.

The five-arm source was `five-arm-audit.py`, hash
`dd720b24eccf36c73972dda86023d6afc02b530574364333efa7bdafeb1e6036`.
Its existing `jaxnav-five-arms-audit.json` already held means, medians and SEM.
Those old values are the expected output. They were not made by the new reader.
All 25 raw evaluation logs matched their saved audit hashes.
The old helper `fixed_eval_log_curves.py` was also copied from the
`research+cce-robotics-transfer` worktree. Its generic default needs at least three
seeds per bin; the five-arm audit overrides that with all five seeds and 50 bins.

Each manifest retains selected run IDs, seeds, source hashes and known configs.
The decay bundle retains the full original manifest and old baseline config reads.
Historical CCE config fields not checked by that old plot are not newly certified.
Fixed-map evaluations remain development evidence. No new-map claim is made.
Raw-TD PER and the ranked-TD decay endpoint retain their distinct labels.

## Pass and missing-data table

| Decay arm/group | Shown seeds | Mean ends at |
|---|---|---:|
| Mul / historical old | 0,1,2,3,4 | 200k |
| Mul / old decay | 0,1,2,3,4 | 195k |
| Mul / fixed constant | 0,1,2,3 | 10k |
| Mul / fixed decay | 0,1,2 | 200k |
| Add / historical old | 0,1,2,3,4 | 200k |
| Add / old decay | 0,1,2 | 200k |
| Add / fixed constant | 0,1,2 | 200k |
| Add / fixed decay | 0,1,2 | 200k |
| Only / historical old | 0,1,2,3,4 | 200k |
| Only / old decay | 0,1,2 | 85k |
| Only / fixed constant | 0,1,2 | no full bin |
| Only / fixed decay | none | waiting, 0/5 |

Decay: all three panels pass exactly, including labels, styles, seed curves,
bin centers, means, null tails, seed IDs and saved source selection.
The two baseline means also pass in each panel. The snapshot has 27 of 45 planned
treatment logs, 15 historical CCE logs, and 10 historical baseline logs.
The 18 absent treatment files are listed in the manifest and report. This says
absent at the saved snapshot; it does not say missing now or failed training.

Five-arm: Uniform, PER, CCE Mul, CCE Only and CCE Add 50/50 all pass exactly.
The check includes all 25 selected runs and configs, all 50 bin centers, and every
mean, median and SEM. Labels and plot notes follow the copied source.

The decay frozen-output hash is
`d280c42c5c90f1e510a3b225b28145cf0eb31bd86e2b6b74e8b21702d6ec25be`.
The five-arm frozen-output hash is
`70c5e765ff0267aac6e41eb957df40db5fcbe8853a634ad22547cb38bbe8d368`.
Reports record candidate/old source hashes, manifest hash, runtime and zero tolerance.
These are numeric checks, not image-pixel equality claims.

## Run from this checkout

```bash
timeout 180s taskset -c 0-3 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 PYTHONPATH=src:. python -m pytest -q \
  tests/test_historical_reporting.py tests/test_migrated_plot.py

timeout 180s taskset -c 0-3 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 PYTHONPATH=src:. python -m analysis.historical_reporting \
  --manifest analysis/historical_reproduction/decay-20261007-2224-v2/manifest.json \
  --output /path/to/fresh-decay-output

timeout 180s taskset -c 0-3 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 PYTHONPATH=src:. python -m analysis.historical_five_arm \
  --manifest analysis/historical_reproduction/five-arm-20261007/manifest.json \
  --output /path/to/fresh-five-arm-output
```

Use four allowed CPU IDs if 0–3 are not available on another host.
Tests were written at the public manifest reader and equality gate first.
Observed red checks: missing reader; missing gate; missing seed selection;
unsupported gzip; missing five-arm reader; invalid identity accepted.
Then implementation made each check pass. Latest focused result: 17 passed,
11 historical tests and six existing new-run plot tests, in 1.11 seconds.
TDD kept expected data separate from new reader outputs.

## Limits still open

- This closes these two historical plot checks, not training parity or all history.
- `cce-confidence.py` and its saved bootstrap report were found and read.
  That inference plot is not ported here. No new bootstrap claim is made.
- The generic two-panel fixed-eval plot and HTML curve books have no new gate here.
- Original historical training code/package identity remains uncertified.
- A current live view needs a new named snapshot. It must not replace this one.
- No jobs, source-run edits, core edits, commits or branches were made.
