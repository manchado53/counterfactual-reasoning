"""Recover logged configs without reading checkpoints or treating logs as success."""
import ast
import hashlib
import json
from pathlib import Path
from port_sources import ROOT, STUDY, add


def main():
    main_root = ROOT.parents[2]
    roots = [(main_root, 'smax'), (main_root, 'chess'),
             (ROOT.parent / 'cce-cvrp-logistics', 'cvrp'),
             (ROOT.parent / 'cvrp-claim1', 'cvrp'),
             (ROOT.parent / 'cce-rpdqn', 'cvrp_neural')]
    rows, scans = [], []
    total = 0
    for root, env in roots:
        directory = root / 'src/counterfactual_rl/agents' / env / 'runs'
        paths = sorted(directory.glob('*/metrics.log'))
        scans.append(dict(root=str(directory), logs=len(paths)))
        for path in paths:
            lines = []
            with path.open() as f:
                for line in f:
                    if not line.startswith('#'):
                        break
                    lines.append(line)
                    if sum(map(len, lines)) > 64_000:
                        raise ValueError(f'Header exceeds 64KB: {path}')
            header = ''.join(lines)
            total += len(header.encode())
            if total > 12 * 1024**2:
                raise ValueError('12 MiB header cap')
            config = {}
            for line in lines:
                key, sep, value = line[1:].strip().partition(': ')
                if sep and key.isidentifier() and key == key.lower():
                    try:
                        config[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        config[key] = value
            rows.append(dict(environment=env, run=path.parent.name, source=str(path),
                header_sha256=hashlib.sha256(header.encode()).hexdigest(), header=header,
                seed=config.get('seed'), algorithm=config.get('algorithm'),
                priority_mixing=config.get('priority_mixing'), mu=config.get('mu'),
                code_status='logged config recovered; source revision at execution unbound',
                evidence_status='header only; no completion or superiority claim'))
    add({STUDY / 'run-headers.json': json.dumps(dict(scan=scans, source_header_bytes=total, rows=rows,
            scope='Immediate per-run metrics.log headers only. Full logs and checkpoints stay at source.'), ensure_ascii=False) + '\n'})
    print(f'{len(rows)} log headers, {total} source bytes')
    print(json.dumps(scans))


if __name__ == '__main__':
    main()
