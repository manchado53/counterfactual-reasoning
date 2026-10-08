"""Bounded read-only history scan; preserve selected historical sources via apply_patch."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from port_sources import ROOT, STUDY, add


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True, timeout=30)


def main():
    refs = git('for-each-ref', '--format=%(refname)', 'refs/heads', 'refs/remotes').splitlines()
    pattern = re.compile(r'smac|routing_budget|cvrp.*budget|budget.*cvrp|routing.*random|cvrp_neural', re.I)
    coverage = {}
    for ref in refs:
        coverage[ref] = [p for p in git('ls-tree', '-r', '--name-only', ref).splitlines() if pattern.search(p)]
    files, rows = {}, []
    ref = 'refs/remotes/origin/refactorization'
    for p in coverage[ref]:
        if not p.endswith(('.py', '.sh', '.md')):
            continue
        text = git('show', f'{ref}:{p}')
        dest = STUDY / 'history' / 'smac-refactorization' / p
        files[dest] = text
        rows.append(dict(source=f'{ref}:{p}', commit=git('rev-parse', ref).strip(), archive=str(dest),
                         sha256=hashlib.sha256(text.encode()).hexdigest(), status='archive-only'))
    worktrees = ['cce-cvrp-logistics', 'cvrp-claim1', 'cce-rpdqn']
    status = {}
    for name in worktrees:
        root = ROOT.parent / name
        status[name] = dict(commit=git('rev-parse', 'HEAD', cwd=root).strip(), dirty=git('status', '--porcelain', cwd=root))
        paths = git('ls-files', '--cached', '--others', '--exclude-standard', cwd=root).splitlines()
        selected = [p for p in paths if (p.startswith('src/counterfactual_rl/agents/cvrp/') or
                    pattern.search(p)) and p.endswith(('.py', '.sh', '.md', '.json'))]
        for p in selected:
            source = root / p
            if not source.is_file() or source.stat().st_size > 200_000:
                continue
            text = source.read_text()
            dest = STUDY / 'history' / name / p
            files[dest] = text
            rows.append(dict(source=str(source), commit=status[name]['commit'], archive=str(dest),
                             sha256=hashlib.sha256(text.encode()).hexdigest(), status='archive-only'))
    size = sum(len(s.encode()) for s in files.values())
    if size > 5 * 1024**2:
        raise ValueError(f'History source cap exceeded: {size}')
    files[STUDY / 'history-manifest.json'] = json.dumps(dict(ref_coverage=coverage, worktrees=status,
         files=rows, bytes=size, limitations='Local known refs only; no fetch. Selected text sources <=200KB each. '
         'Run roots, checkpoints and notebook contents not copied. Archives are not verified ports.'), indent=2) + '\n'
    add(files)
    print(f'Preserved {len(rows)} files, {size} bytes; scanned {len(refs)} refs.')


if __name__ == '__main__':
    main()
