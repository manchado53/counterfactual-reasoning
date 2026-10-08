"""Static guards on the gate's budget and preservation of production settings."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_full_size_profile_does_not_override_config():
    tree = ast.parse((ROOT/"experiments/parity/training_worker.py").read_text())
    branches = [n for n in ast.walk(tree) if isinstance(n,ast.If)
                and ast.unparse(n.test) == "args.profile == 'full-size'"]
    assert any(any(isinstance(s,ast.Assign) and ast.unparse(s) == "overrides = {}"
                   for s in branch.body) for branch in branches)


def test_prefix_cap_rejects_unbounded_run(tmp_path):
    worker = ROOT/"experiments/parity/training_worker.py"
    proc = subprocess.run([sys.executable,str(worker),"--source",str(tmp_path),
        "--reference",str(tmp_path),"--kind","reference","--task","1",
        "--output",str(tmp_path/"out"),"--chunks","4"],capture_output=True,text=True)
    assert proc.returncode == 2
    assert "limited to 1..3" in proc.stderr
    assert not (tmp_path/"out").exists()


def test_scheduler_script_uses_sealed_candidate():
    text = (ROOT/"slurm/parity.sbatch").read_text()
    assert '$suite/candidate' in text
    assert "--profile full-size --device gpu" in text
    assert "200000" not in text
