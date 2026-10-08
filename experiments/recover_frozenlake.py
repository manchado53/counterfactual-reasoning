"""Recover logged settings; never infer missing fields from current defaults.

A runnable recipe reruns those settings through the explicitly named legacy port.
It does not assert that a Git ref proves the source used by a historical job.
"""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.history import read_metrics_config
from experiments.learners.frozenlake.build import validate_config


def recover(manifest, run_root):
    source = json.loads(Path(manifest).read_text())
    rows, blocked = [], []
    for job, overrides in source.items():
        log = Path(run_root) / str(job) / 'metrics.log'
        if not log.is_file():
            blocked.append(dict(job=job, reason='metrics.log missing', overrides=overrides))
            continue
        cfg = read_metrics_config(log)
        mismatch = {k: [v, cfg.get(k)] for k,v in overrides.items() if cfg.get(k) != v}
        if mismatch:
            blocked.append(dict(job=job, reason='manifest/header mismatch', fields=mismatch))
            continue
        # These are run-interface labels only, not historical learner defaults.
        collector = 'sequential' if cfg.get('vectorized') is False else 'old'
        resolved = dict(cfg, collector_variant=collector, training_budget_kind='episodes')
        try:
            validate_config(resolved)
        except ValueError as exc:
            blocked.append(dict(job=job, reason=str(exc), logged_config=cfg))
            continue
        header = '\n'.join(line for line in log.read_text().splitlines() if line.startswith('#')) + '\n'
        arm = cfg['algorithm']
        if arm == 'consequence-dqn':
            arm += f"/{cfg['priority_mixing']}/mu={cfg['mu']}/c={cfg['mu_c']}/td={cfg['mu_delta']}"
        rows.append(dict(task=len(rows), original_job=job, seed=cfg['seed'], arm=arm,
                         collector=collector, config=resolved,
                         provenance=dict(log=str(log.resolve()), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                                         original_header=header,
                                         evidence_status='logged_settings_not_launch_source_proof')))
    return dict(schema_version=1, environment='frozenlake',
                source_manifest=str(Path(manifest).resolve()),
                source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                original_manifest=source,
                code_status='runnable_logged_settings_on_named_legacy_port',
                evidence_status='historical_completion_and_launch_source_not_revalidated',
                warnings=['Do not pool duplicate job IDs across recovered manifests.',
                          'Old weighted_mean without action probabilities still falls back to max.',
                          'No completion, statistical superiority, or full trajectory parity implied.'],
                rows=rows, blocked=blocked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = recover(args.manifest, args.runs)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(rows=len(result['rows']), blocked=len(result['blocked']))))


if __name__ == '__main__':
    main()
