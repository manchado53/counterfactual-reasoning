"""MOCK / FAKE-DATA preview plots for the CCE 'will-it-transfer' predictor.
All numbers are fabricated by hand to show the FORMAT, not real results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

OUT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
CCE_C, TD_C, TRUTH_C = "#4CAF50", "#2196F3", "#FF9800"
FAKE = "ILLUSTRATIVE — FAKE DATA"

def stamp(fig):
    fig.text(0.99, 0.01, FAKE, ha="right", va="bottom",
             fontsize=8, color="#b00", style="italic", alpha=0.8)

# ----------------------------------------------------------------------------
# PLOT 1 — the transferable quadrant map: every env placed by 2 cheap numbers
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7))
# envs: (name, SNR(x), concentration(y), outcome, color)
envs = [
    ("FL-det",   8.0, 0.55, "WIN",      "#2e7d32"),
    ("FL-stoch", 1.2, 0.46, "NULL",     "#c62828"),
    ("SMAX-3m",  2.2, 0.30, "marginal", "#f9a825"),
    ("ConnectFour", 6.5, 0.40, "TD wins\n(attribution)", "#6a1b9a"),
]
# win-region shading (high SNR + high concentration)
ax.axvspan(3.0, 11, ymin=0.0, ymax=1.0, color="#e8f5e9", zorder=0)
ax.axhspan(0.40, 0.75, xmin=0.0, xmax=1.0, color="#e8f5e9", zorder=0, alpha=0.0)
# draw a soft "likely win" box
ax.add_patch(Rectangle((3.0, 0.40), 8, 0.35, color="#c8e6c9", alpha=0.55, zorder=0))
ax.text(7.0, 0.72, "likely CCE WIN\n(clean + concentrated)", ha="center",
        va="top", fontsize=10, color="#1b5e20", fontweight="bold")
ax.text(1.3, 0.13, "likely NO GAIN\n(noisy or flat)", ha="center", va="bottom",
        fontsize=10, color="#8e0000")

for name, x, y, out, c in envs:
    ax.scatter([x], [y], s=260, color=c, zorder=5, edgecolor="k", linewidth=1.2)
    ax.annotate(f"{name}\n[{out}]", (x, y), textcoords="offset points",
                xytext=(12, 10), fontsize=10, fontweight="bold", color=c)
# a NEW candidate env, placed by its cheap measurement -> read off forecast
ax.scatter([5.0], [0.5], s=320, marker="*", color="k", zorder=6)
ax.annotate("NEW env ?\n(measure -> drop here\n-> forecast: WIN)", (5.0, 0.5),
            textcoords="offset points", xytext=(14, -46), fontsize=9.5,
            color="k", arrowprops=dict(arrowstyle="->", color="k"))

ax.set_xlim(0.5, 11); ax.set_ylim(0.05, 0.78)
ax.set_xlabel("SIGNAL-TO-NOISE  (action effect / env+opponent noise)  -->", fontsize=11)
ax.set_ylabel("CONCENTRATION  (Gini of per-state action-spread)  -->", fontsize=11)
ax.set_title("CCE suitability map — drop any env on it, read off the forecast",
             fontsize=12, fontweight="bold")
ax.axvline(3.0, color="gray", ls=":", lw=1)
stamp(fig); fig.tight_layout(); fig.savefig(f"{OUT}/1_quadrant_map.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# PLOT 2 — FL-det deep dive: 4x4 grid heatmaps (the calibration anchor)
# ----------------------------------------------------------------------------
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
holes = [(1,1),(1,3),(2,3),(3,0)]; goal=(3,3); start=(0,0)

def base_grid(vals, title, ax, cmap="viridis"):
    g = np.array(vals, float)
    im = ax.imshow(g, cmap=cmap, vmin=0, vmax=1)
    for (r,c) in holes:
        ax.add_patch(Rectangle((c-0.5,r-0.5),1,1, color="black"))
        ax.text(c,r,"H",ha="center",va="center",color="white",fontweight="bold")
    ax.text(goal[1],goal[0],"G",ha="center",va="center",color="white",fontweight="bold")
    ax.text(start[1],start[0],"S",ha="center",va="center",color="red",fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return im

# fabricate: true stakes high on cells next to holes / on the path
true_stakes = [
    [0.2,0.3,0.2,0.1],
    [0.7,0.0,0.8,0.0],
    [0.6,0.55,0.9,0.0],
    [0.0,0.5,0.95,0.0],
]
# CCE finds nearly the same cells (slightly noisy)
cce_map = [
    [0.15,0.35,0.25,0.1],
    [0.75,0.0,0.7,0.0],
    [0.5,0.6,0.85,0.0],
    [0.0,0.45,0.9,0.0],
]
# replay counts: CCE concentrates on cliffs; PER spreads / chases early surprise
replay_cce = [
    [0.1,0.2,0.15,0.05],
    [0.8,0.0,0.75,0.0],
    [0.6,0.55,0.95,0.0],
    [0.0,0.5,0.9,0.0],
]
replay_per = [
    [0.55,0.6,0.5,0.45],
    [0.4,0.0,0.5,0.0],
    [0.45,0.5,0.55,0.0],
    [0.0,0.55,0.6,0.0],
]
fig, axes = plt.subplots(1,4, figsize=(16,4.6))
im0=base_grid(true_stakes,
    "(a) WHICH CELLS DECIDE WIN/LOSE\nthe real answer  [EXACT, from env.P]\nbright = cliff",axes[0],"viridis")
im1=base_grid(cce_map,
    "(b) WHICH CELLS CCE THINKS MATTER\nCCE's guess  [from rollouts]\nshould match (a)",axes[1],"viridis")
im2=base_grid(replay_cce,
    "(c) HOW OFTEN CCE STUDIED EACH CELL\ntally of buffer grabs  [CCE run]\nbright = studied a lot",axes[2],"magma")
im3=base_grid(replay_per,
    "(d) HOW OFTEN PER STUDIED EACH CELL\ntally of buffer grabs  [PER run]\nspread thin = no focus",axes[3],"magma")
cb1=fig.colorbar(im0, ax=axes[:2], shrink=0.7, label="how much the cell matters (0 - 1)")
cb2=fig.colorbar(im2, ax=axes[2:], shrink=0.7, label="times the cell was studied (0 - 1)")
fig.suptitle("FL-det 4x4   |   (a) the cliffs  ->  (b) CCE FINDS them  ->  (c) CCE STUDIES them   vs   (d) PER studies everything evenly",
             fontsize=11.5, fontweight="bold")
stamp(fig); fig.savefig(f"{OUT}/2_fldet_grids.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# PLOT 3 — calibration + SNR + per-cell convergence (the supporting evidence)
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(15,4.4))
gs = gridspec.GridSpec(1,3, figure=fig, wspace=0.32)

# (a) calibration: cheap predictor vs exact truth (should hug the diagonal)
ax=fig.add_subplot(gs[0])
rng=np.random.default_rng(0)
exact=rng.uniform(0,1,40); cheap=exact+rng.normal(0,0.08,40)
ax.scatter(exact,cheap,s=40,color=CCE_C,edgecolor="k",alpha=0.8)
ax.plot([0,1],[0,1],"--",color="gray")
ax.set_xlabel("EXACT stakes (env.P)"); ax.set_ylabel("CHEAP predictor (rollouts)")
ax.set_title("(a) calibration: cheap ≈ exact\nrho=0.94  -> trust it elsewhere", fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# (b) SNR bars: between-action vs within-action spread, det vs stoch
ax=fig.add_subplot(gs[1])
labels=["FL-det","FL-stoch"]
between=[0.85,0.80]; within=[0.10,0.65]
x=np.arange(2); w=0.35
ax.bar(x-w/2,between,w,label="between-action (signal)",color=CCE_C)
ax.bar(x+w/2,within,w,label="within-action (noise)",color="#9e9e9e")
for i,(b,wi) in enumerate(zip(between,within)):
    ax.text(i,0.92,f"SNR={b/wi:.1f}",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.05)
ax.set_ylabel("return spread"); ax.legend(fontsize=8)
ax.set_title("(b) SNR: noise drowns the signal\nin stochastic FL", fontsize=10)

# (c) per-cell convergence: pivotal cells learn sooner under CCE
ax=fig.add_subplot(gs[2])
steps=np.linspace(0,1,100)
def curve(speed): return 1-np.exp(-speed*steps*6)
ax.plot(steps, curve(2.2), color=CCE_C, lw=2, label="CCE — pivotal cells")
ax.plot(steps, curve(0.9), color=TD_C, lw=2, label="PER — pivotal cells")
ax.plot(steps, curve(1.4), color=CCE_C, lw=1, ls=":", label="CCE — easy cells")
ax.plot(steps, curve(1.3), color=TD_C, lw=1, ls=":", label="PER — easy cells")
ax.set_xlabel("training (norm)"); ax.set_ylabel("Q correctness vs Q* (norm)")
ax.set_title("(c) pivotal cells converge SOONER\nunder CCE -> faster solve", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
fig.suptitle("Supporting evidence — calibrate (a), why stoch dies (b), the payoff (c)",
             fontsize=12, fontweight="bold")
stamp(fig); fig.savefig(f"{OUT}/3_evidence.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote 3 mock plots to", OUT)
