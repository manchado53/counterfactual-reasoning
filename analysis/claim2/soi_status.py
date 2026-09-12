"""Health box for the score-on-insert sweep (array 275761).

We skipped the smoke, so the gates it would have checked are checked HERE, on the real
runs, from their first eval onward. Everything this prints comes from files the runs write
anyway -- nothing extra is computed and no GPU is touched.

    python analysis/claim2/soi_status.py

The three questions, and where each is answered:

    does score-on-insert actually reach 100%?   ess.jsonl  scored_frac   -> want >= 0.99
    is H=200 mostly zeros, as predicted?        ess.jsonl  score_n_zero/scored_n
    is the reward arm doing anything?           eval.jsonl crash_rate, win_rate

A run is HEALTHY if it is progressing and its coverage gate passes. Anything else is
called out by name so it can be killed early instead of burning days.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import greedy_learning_curves as G

# Two arrays: 275761 kept 9 healthy runs; 275791 resubmitted the 21 that died on the
# broken DGX nodes (dh-dgx1-1 / dh-dgx1-2 cannot init CUDA for this env).
ARRAY = "275761,275791"
MANIFEST = os.path.join(G.DATA, "manifest_score_on_insert.json")
TARGET_EPISODES = 250_000


def _raw_ids():
    out = subprocess.run(["sacct", "-j", ARRAY, "-X", "-n", "-P",
                          "--format=JobID,JobIDRaw,State,Elapsed,NodeList"],
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) >= 5:
            d[p[0]] = dict(raw=p[1], state=p[2].split()[0], elapsed=p[3], node=p[4])
    return d


def _last(path, pred=lambda r: True):
    if not os.path.exists(path):
        return None
    last = None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if pred(r):
            last = r
    return last


def main():
    if not os.path.exists(MANIFEST):
        print("no manifest yet"); return
    man = json.load(open(MANIFEST))
    info = _raw_ids()

    rows = []
    for key, v in sorted(man.items()):
        jid = v["job_id"]
        meta = info.get(jid, {})
        rid = meta.get("raw", jid)
        rd = os.path.join(G.RUNS, rid)
        ev = _last(os.path.join(rd, "eval.jsonl"))
        es = _last(os.path.join(rd, "ess.jsonl"), lambda r: r.get("scored_frac") is not None)
        ep = ev.get("episode") if ev else 0
        cov = es.get("scored_frac") if es else None
        zer = (es["score_n_zero"] / es["scored_n"]
               if es and es.get("scored_n") else None)
        rows.append(dict(key=key, arm=v["arm"], seed=v["seed"], state=meta.get("state", "?"),
                         elapsed=meta.get("elapsed", "-"), node=meta.get("node", "-"),
                         ep=ep or 0, cov=cov, zer=zer,
                         win=(ev or {}).get("win_rate"), crash=(ev or {}).get("crash_rate")))

    w = 104
    print("+" + "-" * w + "+")
    title = f" SCORE-ON-INSERT  array {ARRAY}   {len(rows)} runs "
    print("|" + title.center(w) + "|")
    print("+" + "-" * w + "+")
    print(f"| {'arm':<15}{'sd':>3}{'state':>11}{'elapsed':>10}{'episodes':>12}"
          f"{'kappa':>8}{'zeros':>8}{'win':>7}{'crash':>7}{'gate':>7} |")
    print("|" + "-" * w + "|")

    bad = []
    for r in rows:
        pct = 100 * r["ep"] / TARGET_EPISODES
        cov = f"{100*r['cov']:.1f}%" if r["cov"] is not None else "-"
        zer = f"{100*r['zer']:.0f}%" if r["zer"] is not None else "-"
        win = f"{100*r['win']:.0f}%" if r["win"] is not None else "-"
        cra = f"{100*r['crash']:.0f}%" if r["crash"] is not None else "-"
        # Gate: CCE arms must reach >=99% coverage once scoring has started. Baselines
        # never score, so they are exempt -- an "-" there is correct, not a failure.
        if r["state"] in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            gate, why = "DEAD", r["state"]
        elif r["arm"].startswith("soi_") and r["cov"] is not None and r["cov"] < 0.99:
            gate, why = "LOW-COV", f"kappa {100*r['cov']:.1f}% < 99%"
        elif r["state"] == "PENDING":
            gate, why = "queued", None
        elif r["cov"] is None and r["arm"].startswith("soi_") and r["ep"] > 0:
            gate, why = "no-ess", "scoring produced no ess.jsonl row yet"
        else:
            gate, why = "ok", None
        if why:
            bad.append((r["key"], why))
        print(f"| {r['arm']:<15}{r['seed']:>3}{r['state'][:10]:>11}{r['elapsed']:>10}"
              f"{r['ep']:>8,}{pct:>4.0f}%{cov:>8}{zer:>8}{win:>7}{cra:>7}{gate:>7} |")
    print("+" + "-" * w + "+")

    soi = [r for r in rows if r["arm"].startswith("soi_") and r["cov"] is not None]
    if soi:
        print(f"\nCOVERAGE CLAIM: {sum(1 for r in soi if r['cov'] >= 0.99)}/{len(soi)} "
              f"CCE runs reporting >= 99%   "
              f"(min {100*min(r['cov'] for r in soi):.1f}%)")
        for h in (60, 200):
            g = [r for r in soi if f"h{h}_" in r["arm"] and r["zer"] is not None]
            if g:
                print(f"  H={h:<4} zero-score fraction {100*sum(r['zer'] for r in g)/len(g):5.1f}%"
                      f"   (predicted: H=60 ~28.5%, H=200 ~89%)")
    if bad:
        print("\nNEEDS ATTENTION")
        for k, why in bad:
            print(f"  {k:<24} {why}")
    else:
        print("\nno failures, no low-coverage runs")


if __name__ == "__main__":
    main()
