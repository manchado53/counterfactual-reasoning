"""Fill in each array task's RAW job id -- the name of its run directory.

A SLURM array task is addressed as <array_id>_<task>, but SLURM allocates it a
distinct JobIDRaw, and that raw id is what the trainer names its run directory
after. So a manifest that records only "273775_10" points at a directory that
does not exist. This resolves the mapping from sacct and writes `run_dir` next
to each entry. Safe to re-run: tasks not yet allocated are skipped and picked
up on a later pass.

    python slurm/resolve_manifest.py [array_id]
"""
import json, os, subprocess, sys

MANIFEST = "docs/figures/real/claim2/jaxnav/data/manifest_factorial.json"
array_id = sys.argv[1] if len(sys.argv) > 1 else None

man = json.load(open(MANIFEST))
arrays = {v["array_id"] for v in man.values()} if not array_id else {array_id}

raw = {}
for aid in arrays:
    out = subprocess.run(["sacct", "-j", aid, "--format=JobID,JobIDRaw",
                          "--noheader", "-X", "-P"],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 2 and "_" in parts[0]:
            raw[parts[0].strip()] = parts[1].strip()

n = 0
for key, rec in man.items():
    r = raw.get(rec["job_id"])
    if r and rec.get("run_dir") != r:
        rec["run_dir"] = r
        n += 1
json.dump(man, open(MANIFEST, "w"), indent=2)
resolved = sum(1 for v in man.values() if v.get("run_dir"))
print(f"resolved {n} new; {resolved}/{len(man)} entries now have run_dir")
