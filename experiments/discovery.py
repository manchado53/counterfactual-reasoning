"""Bounded, read-only source discovery across known refs and worktrees.

This index finds leads. It does not certify run provenance or archive their data.
Ignored research code is included; model weights and raw training logs are not read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

SKIP = {'.git', '.claude', '.codex', '__pycache__', 'node_modules', 'venv', '.venv',
        'runs', 'logs', 'vendor', 'build', 'dist', 'checkpoints', '.pytest_cache'}
SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.sbatch', '.sh', '.ipynb'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=30)


def scan_tree(root, max_files=20000, max_bytes=64*1024**2):
    entries, skipped, total, inspected = [], [], 0, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(directory, d).is_symlink())
        for name in sorted(files):
            path = Path(directory, name)
            if path.suffix not in SUFFIXES or path.is_symlink():
                continue
            inspected += 1
            if inspected > max_files:
                return dict(entries=entries, skipped=skipped, bytes=total, truncated='file-count cap')
            size = path.stat().st_size
            if size > 2*1024**2 or total + size > max_bytes:
                skipped.append(dict(path=str(path.relative_to(root)), bytes=size, reason='byte cap'))
                continue
            payload = path.read_bytes()
            total += len(payload)
            entries.append(dict(path=str(path.relative_to(root)), bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest()))
    return dict(entries=entries, skipped=skipped, bytes=total, truncated=None)


def discover(root):
    refs = []
    trees = {}
    for line in git(root, 'for-each-ref', '--format=%(refname) %(objectname)',
                    'refs/heads', 'refs/remotes').splitlines():
        name, commit = line.split()
        refs.append(dict(ref=name, commit=commit))
        if commit not in trees:
            trees[commit] = [p for p in git(root, 'ls-tree', '-r', '--name-only', commit).splitlines()
                             if Path(p).suffix in SUFFIXES]
    worktrees = []
    for block in git(root, 'worktree', 'list', '--porcelain').strip().split('\n\n'):
        fields = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        path = Path(fields['worktree'])
        if not path.is_dir():
            worktrees.append(dict(path=str(path), error='worktree path missing'))
            continue
        worktrees.append(dict(path=str(path), head=fields.get('HEAD'), branch=fields.get('branch'),
                              status=git(path, 'status', '--short'), **scan_tree(path)))
    return dict(schema_version=1, refs=refs, trees=trees, worktrees=worktrees,
                scope='Known local and remote-tracking refs; bounded worktree source files, including ignored work/',
                exclusions=sorted(SKIP),
                limits='20k candidate files, 64MiB read per worktree, 2MiB per file; omitted files listed',
                gaps=['No fetch or claim that local remote refs are current',
                      'No exhaustive traversal of training logs/checkpoints or external scratch directories',
                      'No claim that source hash establishes which code produced an old run'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = discover(args.repo.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(refs=len(result['refs']), worktrees=len(result['worktrees']))))


if __name__ == '__main__':
    main()
