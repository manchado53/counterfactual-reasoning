"""Scorecard I/O: JSON schema, the 6-panel figure, and best-effort injection of real numbers
into the interactive mock dashboard (docs/figures/mock_preview/dashboard.html)."""

import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (metric key, sub-field, pass-threshold) — drives both the figure and the dashboard inject.
PANELS = [
    ("concentration", "gini", 0.5),
    ("snr", "value", 3.0),
    ("distinct_td", "value", 0.5),
    ("gain_fidelity", "value", 0.5),
    ("need", "value", 0.7),
    ("horizon_fit", "value", 0.8),
]
# dashboard.html keys each metric object by a unique `viz:"..."`.
VIZ_MAP = {  # viz -> (metric key, sub-field, decimals)
    "lorenz": ("concentration", "gini", 2),
    "snr": ("snr", "value", 1),
    "distinct": ("distinct_td", "value", 2),
    "fidelity": ("gain_fidelity", "value", 2),
    "horizon": ("horizon_fit", "value", 2),
    "need": ("need", "value", 2),
}


def save_json(scorecard: dict, path: str):
    with open(path, "w") as f:
        json.dump(scorecard, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get(ckpt_metrics, mk, sub):
    m = ckpt_metrics.get(mk)
    if not m:
        return None
    return m.get(sub)


LOG_PANELS = {"snr", "horizon_fit"}  # span orders of magnitude → log y-axis for readability


def plot_scorecard(scorecard: dict, png: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (mk, sub, thr) in zip(axes, PANELS):
        all_y = []
        for env, ed in scorecard["envs"].items():
            xs = [c["phase"] for c in ed["checkpoints"]]
            ys = [_get(c["metrics"], mk, sub) for c in ed["checkpoints"]]
            ys = [np.nan if y is None else y for y in ys]
            all_y += [y for y in ys if not np.isnan(y)]
            ax.plot(xs, ys, "o-", label=env)
        ax.axhline(thr, ls=":", color="gray")
        # log-scale only when every plotted value is strictly positive
        if mk in LOG_PANELS and all_y and min(all_y) > 0:
            ax.set_yscale("log")
            ax.set_title(f"{mk}.{sub}  (pass {thr}, log scale)", fontsize=10)
        else:
            ax.set_title(f"{mk}.{sub}  (pass {thr})", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.suptitle("CCE suitability scorecard  ·  " + ", ".join(scorecard["envs"].keys()),
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


# foe strength order for the Connect Four dose-response sweep (noisy -> clean)
_FOE_ORDER = {"random": 0, "rule_based": 1, "mcts": 2}


def plot_opponent_sweep(scorecard: dict, png: str):
    """C4 dose-response: SNR / DISTINCT-TD / NEED vs foe strength, one line per checkpoint phase.

    Reads only `C4-<foe>` envs; no-op if none present. The x-axis is the foe noise axis (the C4
    analog of the FrozenLake slip sweep): random (noisiest) -> rule_based -> mcts (cleanest)."""
    c4 = {k: v for k, v in scorecard["envs"].items() if k.startswith("C4-")}
    if not c4:
        return
    foes = sorted((k[len("C4-"):] for k in c4), key=lambda f: _FOE_ORDER.get(f, 99))
    phases = [c["phase"] for c in next(iter(c4.values()))["checkpoints"]]
    panels = [("snr", "value", "SNR", True), ("distinct_td", "value", "DISTINCT-TD", False),
              ("need", "value", "NEED", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (mk, sub, title, logy) in zip(axes, panels):
        pos_only = True
        for phase in phases:
            ys = []
            for foe in foes:
                cks = {c["phase"]: c for c in c4[f"C4-{foe}"]["checkpoints"]}
                y = _get(cks[phase]["metrics"], mk, sub) if phase in cks else None
                ys.append(np.nan if y is None else y)
            if any(not (isinstance(y, float) and y > 0) for y in ys if not np.isnan(y)):
                pos_only = False
            ax.plot(range(len(foes)), ys, "o-", label=phase)
        ax.set_xticks(range(len(foes))); ax.set_xticklabels(foes, rotation=15)
        ax.set_xlabel("foe (noise: high -> low)"); ax.set_title(title, fontsize=11)
        if logy and pos_only:
            ax.set_yscale("log")
        ax.legend(fontsize=8, title="player")
    fig.suptitle("C4 dose-response — metrics vs foe strength (the slip-sweep analog)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def inject_dashboard(scorecard: dict, html_in: str, html_out: str,
                     env_keys=("FL-det", "FL-stoch")) -> list:
    """Replace the FL-det/FL-stoch numbers in dashboard.html's hardcoded METRICS array with the
    FINAL (last) checkpoint's real values. Best-effort: returns the list of metrics injected."""
    with open(html_in) as f:
        html = f.read()

    def final_val(env, mk, sub):
        cks = scorecard["envs"].get(env, {}).get("checkpoints", [])
        if not cks:
            return None
        return _get(cks[-1]["metrics"], mk, sub)

    def fmt(v, dec):
        return "null" if v is None else f"{round(float(v), dec)}"

    injected = []
    for viz, (mk, sub, dec) in VIZ_MAP.items():
        vals = {e: final_val(e, mk, sub) for e in env_keys}
        if all(v is None for v in vals.values()):
            continue
        # Rebuild the WHOLE vals object: real FL-det/FL-stoch, C4-mcts = null (never measured).
        obj = ('{"FL-det":%s,"FL-stoch":%s,"C4-mcts":null}'
               % (fmt(vals.get("FL-det"), dec), fmt(vals.get("FL-stoch"), dec)))
        pat = re.compile(r'(viz:"' + re.escape(viz) + r'".*?vals:)\{[^}]*\}', re.DOTALL)
        html, n = pat.subn(lambda m: m.group(1) + obj, html, count=1)
        if n:
            injected.append(mk)
    # Deferred metric (precision@k / ESS, viz "ess") — not computed → all n/a, no fake "ok".
    html = re.sub(r'(viz:"ess".*?vals:)\{[^}]*\}',
                  r'\1{"FL-det":null,"FL-stoch":null,"C4-mcts":null}', html, count=1, flags=re.DOTALL)
    # Forecast strip: C4 was never run → n/a (FL-det WIN / FL-stoch NULL stay — real known outcomes).
    html = re.sub(r'(fcell" data-col="C4-mcts" style=")[^"]*(">)[^<]*(<)',
                  r'\1background:#23262d;color:#9aa3b2\2n/a\3', html, count=1)
    # Relabel: this generated file holds REAL run data, not the mock template's fake numbers.
    det = scorecard.get("config", {}).get("det_run", "")
    stoch = scorecard.get("config", {}).get("stoch_run", "")
    tag = "REAL DATA"
    if det or stoch:
        tag = f"REAL DATA · det {os.path.basename(det)} / stoch {os.path.basename(stoch)}"
    html = html.replace("MOCK · FAKE DATA", tag)
    html = html.replace(".fake{display:inline-block;color:#ff6b6b;",
                        ".fake{display:inline-block;color:#5fd97a;")  # red→green badge
    with open(html_out, "w") as f:
        f.write(html)
    return injected
