"""README figures for counterfactual-reasoning, in a plain paper style.

1. cce_core.png      - the core algorithm on real FrozenLake 8x8 (slippery) rollouts
2. cce_priority.png  - from scores to sampling probabilities (beta = 1 vs 0.25)
3. jaxnav_curve.png  - latest JaxNav fixed-bank learning curves from the raw eval logs

usage: python make_readme_figs.py <eval_log_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EVAL_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK = "#222222"
GRAY = "#8a8a8a"
LIGHT = "#d9d9d9"
ACCENT = "#c0502e"   # CCE
BLUE = "#3d6a9e"     # PER

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------- FrozenLake
MAP = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
       "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
N = 8
GAMMA = 0.99
ACTIONS = ["left", "down", "right", "up"]
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def cell(s):
    return divmod(s, N)


def step_to(s, a):
    r, c = cell(s)
    dr, dc = MOVES[a]
    r2, c2 = min(max(r + dr, 0), N - 1), min(max(c + dc, 0), N - 1)
    return r2 * N + c2


def terminal(s):
    r, c = cell(s)
    return MAP[r][c] in "HG"


def outcomes(s, a):
    """Slippery: intended direction or either perpendicular, 1/3 each."""
    return [step_to(s, (a + k) % 4) for k in (-1, 0, 1)]


def reward(s2):
    r, c = cell(s2)
    return 1.0 if MAP[r][c] == "G" else 0.0


S = N * N
Q = np.zeros((S, 4))
for _ in range(2000):
    V = Q.max(1)
    Qn = np.zeros_like(Q)
    for s in range(S):
        if terminal(s):
            continue
        for a in range(4):
            Qn[s, a] = np.mean([reward(s2) + (0 if terminal(s2) else GAMMA * V[s2])
                                for s2 in outcomes(s, a)])
    if np.abs(Qn - Q).max() < 1e-10:
        Q = Qn
        break
    Q = Qn
policy = Q.argmax(1)

rng = np.random.default_rng(7)


def rollout(s, a, horizon=200):
    """Take action a in s, then follow the greedy policy. Return (G, path)."""
    path = [s]
    G, disc, act = 0.0, 1.0, a
    for _ in range(horizon):
        s = outcomes(s, act)[rng.integers(3)]
        path.append(s)
        G += disc * reward(s)
        disc *= GAMMA
        if terminal(s):
            break
        act = policy[s]
    return G, path


BINS = np.linspace(0, 1, 51)


def tv(x, y):
    p, _ = np.histogram(x, BINS)
    q, _ = np.histogram(y, BINS)
    return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()


def cce_score(s, n=200):
    rets = {a: np.array([rollout(s, a)[0] for _ in range(n)]) for a in range(4)}
    taken = policy[s]
    alts = [a for a in range(4) if a != taken]
    return np.mean([tv(rets[taken], rets[a]) for a in alts]), rets


# Pick the most and least consequential states by a first pass, then
# re-score those two with many rollouts so histogram noise stays small.
nonterm = [s for s in range(S) if not terminal(s)]
first = {s: cce_score(s, n=300)[0] for s in nonterm}
HIGH = max(first, key=first.get)
LOW = min(first, key=first.get)
score_hi, rets_hi = cce_score(HIGH, n=1000)
score_lo, rets_lo = cce_score(LOW, n=1000)


def draw_map(ax, s):
    for r in range(N):
        for c in range(N):
            ch = MAP[r][c]
            fc = {"H": "#3b3b3b", "G": "#e8e3c8", "S": "white", "F": "white"}[ch]
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec=LIGHT, lw=0.6))
            if ch == "G":
                ax.text(c + .5, r + .5, "goal", ha="center", va="center", fontsize=7, color=INK)
    r, c = cell(s)
    ax.add_patch(Rectangle((c, r), 1, 1, fc="#fbe9e2", ec=ACCENT, lw=2))
    taken = policy[s]
    for a in range(4):
        dr, dc = MOVES[a]
        col = ACCENT if a == taken else GRAY
        ax.annotate("", xy=(c + .5 + dc * .95, r + .5 + dr * .95), xytext=(c + .5, r + .5),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2 if a == taken else 1.2,
                                    mutation_scale=11))
    ax.text(0, N + 0.15, "black squares = holes", ha="left", va="top", fontsize=7, color=GRAY)
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal"); ax.axis("off")


def draw_outcomes(ax, s, rets, score):
    taken = policy[s]
    order = [taken] + [a for a in range(4) if a != taken]
    for i, a in enumerate(order):
        goal = (rets[a] > 0).mean()
        col = ACCENT if a == taken else GRAY
        ax.barh(-i, goal, color=col, height=0.6)
        ax.barh(-i, 1 - goal, left=goal, color="#efefef", height=0.6)
        ax.text(goal + 0.01 if goal < 0.85 else goal - 0.01, -i, f"{goal:.0%}",
                ha="left" if goal < 0.85 else "right", va="center", fontsize=9,
                color=INK if goal < 0.85 else "white")
        ax.text(-0.02, -i, "go " + ACTIONS[a] + ("  (policy's pick)" if a == taken else ""),
                ha="right", va="center", fontsize=9, color=ACCENT if a == taken else INK)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(-3.5, 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of 1,000 tries that reach the goal")
    ax.text(1.0, 0.62, f"CCE score = {score:.2f}", ha="right", va="bottom", fontsize=10,
            color=INK, weight="bold")


fig = plt.figure(figsize=(10, 6.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], hspace=0.5, wspace=0.45)
for row, (s, rets, score, label) in enumerate([
        (HIGH, rets_hi, score_hi, "A state where the choice matters"),
        (LOW, rets_lo, score_lo, "A state where it doesn't")]):
    axm = fig.add_subplot(gs[row, 0]); draw_map(axm, s)
    axr = fig.add_subplot(gs[row, 1]); draw_outcomes(axr, s, rets, score)
    axm.set_title(label, loc="left", fontsize=11, color=INK, pad=8)
fig.savefig(OUT / "cce_core.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print("cce_core", round(score_hi, 3), round(score_lo, 3))
for s, rets in ((HIGH, rets_hi), (LOW, rets_lo)):
    print(cell(s), ACTIONS[policy[s]], {ACTIONS[a]: round((rets[a] > 0).mean(), 3) for a in range(4)})

# ---------------------------------------------------------- priority figure
scores = np.array([first[s] for s in nonterm])
order = np.argsort(scores)[::-1]
sc = scores[order]
eps = 0.01


def probs(beta):
    w = (sc + eps) ** beta
    return w / w.sum()


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
a1.bar(range(len(sc)), sc, color=GRAY, width=0.85)
a1.set_xlabel("states in the buffer, sorted by score")
a1.set_ylabel("CCE score")
a1.set_xticks([])
uni = 1 / len(sc)
x = np.arange(len(sc))
a2.plot(x, probs(1.0) / uni, color=INK, lw=1.4)
a2.plot(x, probs(0.25) / uni, color=ACCENT, lw=1.8)
a2.axhline(1, color=GRAY, lw=0.8, ls=(0, (3, 3)))
a2.text(len(sc) * 0.30, probs(1.0)[int(len(sc) * 0.30)] / uni + 0.25, "β = 1", color=INK, fontsize=9)
a2.text(len(sc) * 0.62, probs(0.25)[int(len(sc) * 0.62)] / uni + 0.18, "β = 0.25 (used)",
        color=ACCENT, fontsize=9)
a2.text(len(sc) - 1, 1.08, "uniform", color=GRAY, fontsize=8, ha="right")
a2.set_xlabel("same states, same order")
a2.set_ylabel("replay rate vs. uniform")
a2.set_xticks([])
ratio = probs(0.25).max() / probs(0.25).min()
fig.text(0.01, -0.04,
         f"Real scores for all {len(sc)} non-terminal FrozenLake states. With β = 0.25 the most "
         f"consequential state is replayed only {ratio:.1f}x as often as the least.",
         fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "cce_priority.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("cce_priority ratio b=0.25:", round(ratio, 2), "b=1:", round(probs(1).max() / probs(1).min(), 2))

# ------------------------------------------------------------ JaxNav curves
ARMS = [("Uniform", ["277734", "277735", "277736"], GRAY),
        ("PER", ["277751", "277752", "277753"], BLUE),
        ("CCE + ranked-TD decay", ["281092", "281093", "280874"], ACCENT)]
BIN = 5000
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axvspan(100, 150, color="#f2f2f2", lw=0)
ax.text(125, 0.03, "100k–150k window", ha="center", fontsize=8, color=GRAY)
window = {}
for name, ids, col in ARMS:
    per_seed = []
    for rid in ids:
        rows = [json.loads(l) for l in open(EVAL_DIR / f"{rid}.jsonl")]
        ep = np.array([r["episode"] for r in rows]); wr = np.array([r["win_rate"] for r in rows])
        b = ep // BIN
        per_seed.append({k: wr[b == k].mean() for k in np.unique(b)})
        m = (ep >= 100_000) & (ep < 150_000)
        window.setdefault(name, []).append(wr[m].mean())
    keys = sorted(set.intersection(*[set(d) for d in per_seed]))
    M = np.array([[d[k] for k in keys] for d in per_seed])
    x = (np.array(keys) + 0.5) * BIN / 1000
    mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(ids))
    ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    ax.plot(x, mu, color=col, lw=1.8 if col == ACCENT else 1.3)
    ax.text(x[-1] + 2, mu[-1], name, color=col, fontsize=9, va="center")
ax.set_xlabel("training episodes (thousands)")
ax.set_ylabel("success rate on 1,000 fixed maps")
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
wm = {k: np.mean(v) for k, v in window.items()}
c, u, p = wm["CCE + ranked-TD decay"], wm["Uniform"], wm["PER"]
fig.text(0.01, -0.02,
         "JaxNav 8x8, 3 matched seeds per arm, every arm evaluated on the same 1,000 maps. "
         "Mean ± SEM in 5k-episode bins, from the raw evaluation logs.",
         fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "jaxnav_curve.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("jaxnav window means", {k: round(v, 4) for k, v in wm.items()})
