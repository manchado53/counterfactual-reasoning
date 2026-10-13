"""Check an installed runner from an empty directory; preserve all evidence.

Run with the isolated environment's Python, not with PYTHONPATH pointed at source.
This checks runtime reproduction and reports dependency conflicts separately.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


KNOWN_CONFLICTS = {
    'jaxmarl 0.1.0 has requirement jax<=0.4.38, but you have jax 0.9.1.',
    'jaxmarl 0.1.0 has requirement scipy<=1.12, but you have scipy 1.16.3.',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', required=True, type=Path)
    p.add_argument('--task', required=True, type=int)
    p.add_argument('--plot-script', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    recipe, plot = args.recipe.resolve(), args.plot_script.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', production_parity='NOT_TESTED', commands=[])
    env = dict(os.environ, PYTHONPATH='', PYTHONNOUSERSITE='1', JAX_PLATFORMS='cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               JAX_ENABLE_COMPILATION_CACHE='false')

    def run(label, command, timeout=180, check=True):
        proc = subprocess.run(command, cwd=out, env=env, capture_output=True,
                              text=True, timeout=timeout)
        (out/(label+'.log')).write_text(proc.stdout+proc.stderr)
        report['commands'].append(dict(label=label, argv=command, returncode=proc.returncode))
        if check and proc.returncode:
            raise RuntimeError(f'{label} failed: see {label}.log')
        return proc

    try:
        import training
        import counterfactual_rl
        prefix = Path(sys.prefix).resolve()
        if sys.prefix == sys.base_prefix:
            raise ValueError('Use an isolated virtual environment')
        for module in (training, counterfactual_rl):
            if not Path(module.__file__).resolve().is_relative_to(prefix):
                raise ValueError('Module came from a source checkout, not the installed wheel')
        report['python'] = sys.version
        report['prefix'] = str(prefix)
        report['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
        cfg = (prefix/'pyvenv.cfg').read_text()
        if 'include-system-site-packages = false' not in cfg:
            raise ValueError('System-site packages must be disabled')
        check = run('dependency-check', [sys.executable, '-m', 'pip', 'check'], check=False)
        conflicts = set(check.stdout.strip().splitlines()) if check.returncode else set()
        report['dependency_conflicts'] = sorted(conflicts)
        if conflicts != KNOWN_CONFLICTS:
            raise ValueError('Historical dependency exceptions changed; inspect before accepting')
        report['dependency_resolution_passed'] = False
        hashes = []
        for copy in ('a', 'b'):
            target = out/copy
            run('train-'+copy, [sys.executable, '-m', 'training.run', 'train',
                '--recipe', str(recipe), '--task', str(args.task), '--output', str(target),
                '--smoke', '--device', 'cpu', '--max-wall-seconds', '180'])
            attempt = json.loads((target/'attempt.json').read_text())
            if attempt['status'] != 'completed' or not attempt.get('eval_bank_hash'):
                raise ValueError('Smoke incomplete or evaluation identity missing')
            with zipfile.ZipFile(target/'source.zip') as bundle:
                if 'project/src/counterfactual_rl/replay/decay.py' not in bundle.namelist():
                    raise ValueError('Source bundle did not capture core code')
                for name, expected in attempt['source_hashes'].items():
                    if hashlib.sha256(bundle.read('project/'+name)).hexdigest() != expected:
                        raise ValueError('Source bundle hash mismatch')
            run('plot-'+copy, [sys.executable, str(plot), '--runs', str(target),
                '--bin-size', '1', '--output', str(out/(copy+'.png'))])
            plotted = json.loads((out/(copy+'.json')).read_text())
            # Paths and wall times differ; compare scientific values and source identity.
            evaluations = [json.loads(line) for line in (target/'eval.jsonl').read_text().splitlines()]
            for row in evaluations:
                row.pop('seconds', None)
            hashes.append(dict(config=attempt['resolved_config'], bank=attempt['eval_bank_hash'],
                source=attempt['source_hashes'], simulator=attempt['simulator_source_hashes'],
                eval=evaluations, plotted=plotted['curves']))
        if hashes[0] != hashes[1]:
            raise ValueError('Two installed CPU smoke runs differ')
        report.update(status='runtime_passed_with_dependency_exception',
                      repeated_smoke_passed=True, plot_data_repeated=True,
                      source_bundle_verified=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report['status'], report=str(out/'report.json'))))
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
