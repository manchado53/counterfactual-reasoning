"""Run explicit historical analysis configs, without private worktree imports.

CLI: python -m experiments.studies.run --config configs/studies/claim2.json --out NEW_DIR
Shared-runner hook: run(config: dict, out: pathlib.Path) -> JSON-safe report.
Data paths use --data-root or config.data_root relative to the config file.
Library callers can supply data_root; otherwise their current directory is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import signal
import tarfile


def _path(value, data_root):
    p = Path(value)
    return p if p.is_absolute() else Path(data_root) / p


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    """Preserve undefined and censored historical values in strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def validate(config):
    if config.get('schema_version') != 1:
        raise ValueError('Expected study schema_version=1')
    allowed = {'claim1_frozen_lake', 'claim1_cvrp', 'claim2', 'graded_slip',
               'covariance', 'suitability_precision', 'rpdqn_logs', 'priority_flatness',
               'claim2_rebuild', 'covariance_checkpoints', 'suitability_rollouts'}
    if config.get('study') not in allowed:
        raise ValueError('Unknown analysis study')
    if not isinstance(config.get('seed'), int):
        raise ValueError('Explicit integer seed required')
    reps = config.get('reps', 32)
    if not isinstance(reps, int) or not 2 <= reps <= 50000:
        raise ValueError('reps must be 2..50000')


def run(config, out, data_root=None):
    validate(config)
    data_root = Path(data_root or Path.cwd()).resolve()
    # CLI supervisor enforces wall time; library callers own their execution budget.
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    np.random.seed(config['seed'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = []

    def source(value):
        p = _path(value, data_root)
        if not p.is_file():
            raise FileNotFoundError(f'Analysis input missing: {p}')
        inputs.append({'path': str(p), 'sha256': _hash(p)})
        return p

    study = config['study']
    evidence = 'fresh_analysis_no_training'
    if study in ('claim2_rebuild', 'covariance_checkpoints', 'suitability_rollouts'):
        from .full_drivers import execute
        result, evidence = execute(config, out, data_root, source)
    elif study in ('claim1_frozen_lake', 'claim1_cvrp'):
        if study == 'claim1_frozen_lake':
            from .frozen_lake_oracle import compute_oracle
            q, scores, states = compute_oracle(**config['oracle'])
        else:
            from .cvrp_oracle import compute_oracle
            q, scores, states = compute_oracle(**config['oracle'])
        np.savez_compressed(out / 'oracle.npz', Q=q, states=states,
                            scores=[scores[s] for s in states])
        result = {'n_states': len(states), 'q_shape': q.shape}
        if config.get('cache'):
            with np.load(source(config['cache']), allow_pickle=False) as cache:
                result['cache_max_abs_q_error'] = float(np.max(np.abs(q-cache['Q'])))
                result['cache_state_order_equal'] = np.array_equal(states, cache['non_terminal'])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(states, [scores[s] for s in states], '.')
        ax.set(xlabel='State', ylabel='Exact action gap', title=study)
        fig.savefig(out / 'oracle.png'); plt.close(fig)
    elif study in ('claim2', 'graded_slip'):
        from .claim2_metrics import final_iqm, steps_to_threshold, iqm_curves, prob_improvement_curves
        from .claim2_figures import fig1_iqm_curves, fig2_final_iqm, fig4b_prob_improvement_curves
        cache_path = source(config['cache'])
        evidence = 'cached_reanalysis_no_training'
        groups = {}
        if study == 'claim2':
            with np.load(cache_path, allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                labels = meta['labels']
                groups[meta['env']] = ({a:z[f'raw_{i}'] for i,a in enumerate(labels)},
                                      {a:z[f'steps_{i}'] for i,a in enumerate(labels)})
            threshold = meta['threshold']
        else:
            from .graded_slip import _load_cache, slip_entropy
            raw, steps, submitted, dropped = _load_cache(cache_path)
            groups = {str(p): (raw[p], steps[p]) for p in raw}
            threshold = config['threshold']
            meta = {'n_submitted': submitted, 'dropped_runs': dropped,
                    'entropy': {str(p): slip_entropy(p) for p in raw}}
        result = {'cache_metadata': meta, 'groups': {}}
        final = {}
        for label, (raw, steps) in groups.items():
            final[label] = final_iqm(raw, reps=config['reps'])
            result['groups'][label] = {'final_iqm': final[label],
                'steps_to_threshold': steps_to_threshold(raw, steps, threshold),
                'shapes': {a:v.shape for a,v in raw.items()}}
            if config.get('curves', False):
                curves = iqm_curves(raw, reps=config['reps'])
                probs = prob_improvement_curves(raw, reps=config['reps'])
                axis = next(iter(steps.values()))
                fig1_iqm_curves({label:curves}, {label:axis}, {label:threshold},
                                str(out / f'iqm_{label}.png'))
                fig4b_prob_improvement_curves({label:probs}, {label:axis},
                                              str(out / f'prob_{label}.png'))
        # One figure per group keeps the historical plotting layout readable.
        for label, values in final.items():
            fig2_final_iqm({label:values}, str(out / f'final_{label}.png'))
    elif study == 'priority_flatness':
        from .priority_flatness import summarise
        data = json.loads(source(config['input']).read_text())
        result = {name:summarise(values) for name,values in data.items()}
        evidence = config['input_evidence']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for name,values in data.items():
            ax.hist(values, bins=20, alpha=0.5, label=name)
        ax.set(xlabel='Saved CCE score', ylabel='Count'); ax.legend()
        fig.savefig(out / 'scores.png'); plt.close(fig)
    elif study in ('covariance', 'suitability_precision'):
        data = json.loads(source(config['input']).read_text())
        evidence = config['input_evidence']
        if study == 'covariance':
            from .covariance import predicate, deployed_predicate, check_identities
            c, d, u = (np.asarray(data[k], dtype=float) for k in ('c','d','u'))
            result = {'linear': predicate(c,d,u), 'deployed': deployed_predicate(c,d,u),
                      'identities': check_identities(c,d,u)}
        else:
            from .suitability import stakes_C, concentration, snr, distinct_td, need, horizon_fit
            from .precision import precision_at_k
            returns = np.asarray(data['returns'], dtype=float)
            stakes = stakes_C(returns)
            result = {'stakes': stakes, 'concentration': concentration(stakes),
                      'snr': snr(returns), 'distinct_td': distinct_td(data['cce'],data['td']),
                      'need': need(stakes,data['visits']),
                      'horizon_fit': horizon_fit(data['cf_horizon'],
                                                {int(k):v for k,v in data['means_by_h'].items()}),
                      'precision': {str(k):precision_at_k(data['oracle'],data['cce'],k)
                                    for k in data['ks']}}
    else:
        from . import rpdqn_logs
        root = _path(config['input'], data_root)
        rpdqn_logs.R = str(root)
        result = {}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for job in config['jobs']:
            source(str(root / str(job) / 'metrics.log'))
            record = rpdqn_logs.load(str(job))
            if record is None:
                raise ValueError(f'RPDQN {job}: fewer than three rows')
            result[str(job)] = record
            ax.plot(record['ep'], rpdqn_logs.smooth(record['ret']), label=str(job))
        ax.set(xlabel='Logged episode', ylabel='Return (negative tour length)')
        ax.legend(); fig.savefig(out / 'rpdqn.png'); plt.close(fig)
        evidence = config['input_evidence']
    versions = {}
    for package in ('numpy','scipy','jax','matplotlib','rliable'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = []
    bundle = out / 'project-sources.tar.gz'
    project_modules = sys.modules.copy()
    project_modules['experiments.studies.run'] = sys.modules[__name__]
    with tarfile.open(bundle, 'w:gz') as archive:
        for name, module in sorted(project_modules.items()):
            if name not in ('experiments','counterfactual_rl') and not name.startswith(('experiments.', 'counterfactual_rl.')):
                continue
            path = getattr(module, '__file__', None)
            if path and Path(path).is_file() and path.endswith('.py'):
                member = name.replace('.', '/') + '.py'
                archive.add(path, arcname=member, recursive=False)
                sources.append({'module':name, 'path':path, 'sha256':_hash(path),
                                'bundle_member':member})
    report = _json({'schema_version':1, 'study':study, 'config':config,
                    'data_root':str(data_root), 'imported_project_sources':sources,
                    'source_bundle':{'path':str(bundle), 'sha256':_hash(bundle)},
                    'evidence_status':evidence, 'inputs':inputs, 'python':sys.version,
                    'versions':versions, 'results':result})
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--input', help='Explicit replacement for input/cache path')
    p.add_argument('--data-root', help='Base for data paths; else config data_root relative to config location')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--validate-inputs', action='store_true')
    p.add_argument('--max-wall', type=float, default=180)
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 < args.max_wall <= 180:
        p.error('--max-wall must be >0 and <=180 seconds')
    if not args._worker:
        child = subprocess.Popen([sys.executable, '-m', 'experiments.studies.run',
                                  *sys.argv[1:], '--_worker'], start_new_session=True)
        try:
            code = child.wait(timeout=args.max_wall)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            print(f'Analysis exceeded {args.max_wall}s; process group stopped.', file=sys.stderr)
            code = 124
        raise SystemExit(code)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    data_root = Path(args.data_root).resolve() if args.data_root else (config_path.parent / config.get('data_root', '.')).resolve()
    config['smoke'] = args.smoke
    config['validate_inputs_only'] = args.validate_inputs
    if args.input:
        config['cache' if 'cache' in config else 'input'] = args.input
    try:
        run(config, Path(args.out), data_root=data_root)
    except Exception as exc:
        print(json.dumps({'status':'failed', 'study':config.get('study'),
                          'error_type':type(exc).__name__, 'error':str(exc),
                          'data_root':str(data_root)}, allow_nan=False), file=sys.stderr)
        raise SystemExit(1)
    print(Path(args.out) / 'report.json')


if __name__ == '__main__':
    main()
