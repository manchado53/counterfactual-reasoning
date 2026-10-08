"""Seal and queue one H100 parity gate after an existing array finishes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.parity.run import snapshot, digest, dump, MAX_SOURCE_BYTES


def watch(job, output):
    previous = None
    deadline = time.monotonic() + 172800
    terminal = {"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE"}
    while time.monotonic() < deadline:
        proc = subprocess.run(["sacct","-X","-n","-P","-j",job,
            "--format=JobID,State,ExitCode"],capture_output=True,text=True)
        rows = [line.split("|") for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode == 0 and rows and rows != previous:
            record = dict(job=job,rows=rows,observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
            dump(output/"monitor.json",record)
            print(json.dumps(record),flush=True)
            previous = rows
        if rows and all(row[1].split()[0].rstrip("+") in terminal for row in rows):
            return
        time.sleep(45)
    dump(output/"monitor.json",dict(job=job,status="monitor_expired"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", required=True,type=Path)
    p.add_argument("--after-job")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--watch")
    args = p.parse_args()
    out = args.output.resolve()
    if args.watch:
        watch(args.watch,out)
        return
    if args.reference is None or not args.after_job or not args.after_job.isdigit():
        p.error("reference and numeric after-job are required")
    if any(c.isspace() for c in str(out)):
        p.error("SLURM output paths cannot contain whitespace")
    out.mkdir(parents=True,exist_ok=False)
    snapshot(args.reference.resolve(),out/"reference")
    root = Path(__file__).resolve().parents[2]
    files = [f for directory in ("src","experiments") for f in (root/directory).rglob("*.py")]
    files += [root/"slurm/parity.sbatch"]
    if sum(f.stat().st_size for f in files) > MAX_SOURCE_BYTES:
        raise ValueError("Candidate source cap exceeded")
    hashes = {}
    for f in files:
        rel = f.relative_to(root)
        target = out/"candidate"/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,target)
        hashes[str(rel)] = digest(target)
    dump(out/"candidate-hashes.json",hashes)
    cmd = ["sbatch","--parsable","--job-name=cce-full-parity","--partition=dgxh100",
        "--account=undergrad_research","--qos=normal","--gres=gpu:h100:1",
        "--cpus-per-task=8","--mem=60G","--time=10:00:00",
        "--dependency=afterany:"+args.after_job,
        "--output="+str(out/"job.out"),"--error="+str(out/"job.err"),
        str(out/"candidate/slurm/parity.sbatch"),str(out),sys.executable]
    dry = subprocess.run(cmd[:1]+["--test-only"]+cmd[1:],capture_output=True,text=True,check=True)
    dump(out/"dry-run.json",dict(command=cmd,stdout=dry.stdout,stderr=dry.stderr))
    print(dry.stdout+dry.stderr,flush=True)
    if not args.submit:
        print("Dry-run only; no job submitted",flush=True)
        return
    for name, expected in hashes.items():
        if digest(out/"candidate"/name) != expected:
            raise ValueError("Candidate changed after sealing")
    done = subprocess.run(cmd,capture_output=True,text=True,check=True)
    job = done.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unknown scheduler response: "+done.stdout)
    with (out/"monitor.log").open("x") as log:
        env = dict(os.environ,PYTHONPATH=str(out/"candidate/src")+os.pathsep+str(out/"candidate"))
        child = subprocess.Popen([sys.executable,"-m","experiments.parity.submit",
            "--watch",job,"--output",str(out)],cwd=out/"candidate",
            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(out/"job.json",dict(job=job,dependency=args.after_job,gpus=1,wall_hours=10,
        monitor_pid=child.pid,source_commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()))
    print("SUBMITTED "+job,flush=True)


if __name__ == "__main__":
    main()
