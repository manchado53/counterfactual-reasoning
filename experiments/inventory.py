"""Read-only Git/worktree inventory. Writes only the requested new JSON report."""
import argparse
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def inventory(root):
    records = []
    blocks = git(root, "worktree", "list", "--porcelain").strip().split("\n\n")
    for block in blocks:
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        path = Path(fields["worktree"])
        tracked = git(path, "ls-files", "-z").split("\0")
        source_paths = [p for p in tracked if p.endswith(".py")]
        records.append(dict(path=str(path), head=fields.get("HEAD"), branch=fields.get("branch"),
            status=git(path, "status", "--short"),
            tracked_python=len(source_paths),
            research_paths=[p for p in tracked if p.startswith(("plans/", "analysis/", "paper/repro/manifests/"))],
            entrypoints=[p for p in source_paths if Path(p).name in
                ("train.py", "run_experiments.py", "transfer.py", "transfer_run.py")
                or p.endswith("_suite.py")],
            evidence_status="requires_run_to_source_audit",
            code_status="not_migrated"))
    return dict(schema_version=1, scope="git-index-and-worktrees-only",
                warning="Not a run inventory or a backup. Untracked code is listed, not archived.",
                worktrees=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = inventory(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"{len(result['worktrees'])} worktrees indexed: {args.output}")


if __name__ == "__main__":
    main()
