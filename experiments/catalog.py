"""List runnable recipes and analysis configs without implying result parity."""
import argparse
import json
from pathlib import Path
from experiments.recipes import validate_row
from experiments.registry import get_environment


def catalog(root):
    root = Path(root)
    training = []
    sources = list((root/'configs/experiment').glob('*.json'))
    sources += list((root/'configs/historical/runnable').glob('*.json'))
    for path in sorted(sources):
        manifest = json.loads(path.read_text())
        failures = []
        if manifest.get('schema_version') != 1:
            raise ValueError(f'Unsupported recipe schema: {path}')
        if len({r['task'] for r in manifest['rows']}) != len(manifest['rows']):
            raise ValueError(f'Duplicate tasks: {path}')
        adapter = get_environment(manifest['environment'])
        for row in manifest['rows']:
            try:
                validate_row(row, adapter)
            except (ValueError, KeyError) as exc:
                failures.append(dict(task=row['task'], error=str(exc)))
        training.append(dict(path=str(path.relative_to(root)), environment=manifest['environment'],
                             rows=len(manifest['rows']), invalid=failures,
                             evidence=manifest.get('evidence_status', 'see per-study verification; validation is not parity'),
                             blocked_recovery_rows=len(manifest.get('blocked', []))))
    analyses = []
    from experiments.studies.run import validate
    for path in sorted((root/'configs/studies').glob('*.json')):
        config = json.loads(path.read_text())
        try:
            validate(config)
            error = None
        except (ValueError, KeyError) as exc:
            error = str(exc)
        analyses.append(dict(path=str(path.relative_to(root)), study=config.get('study'), invalid=error))
    return dict(schema_version=1, training=training, analyses=analyses,
                warning='Validation is not execution, old-job completion, or parity. Read the study evidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Checkout/data root, not an implicit installed-package path')
    args = p.parse_args()
    result = catalog(args.root)
    print(json.dumps(result, indent=2))
    if any(x['invalid'] for x in result['training'] + result['analyses']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
