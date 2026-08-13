"""MOCK / FAKE-DATA — the visuals BEHIND the two ruler numbers.
A) what the Gini number is made of   (sorted bar-pile + Lorenz curve)
B) what the SNR number is made of    (within-move vs between-move wiggle)
All numbers fabricated to show FORMAT, not real results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
CCE_C = "#4CAF50"; NOISE_C = "#9e9e9e"; FAKE = "ILLUSTRATIVE — FAKE DATA"
rng = np.random.default_rng(1)

def stamp(fig):
    fig.text(0.99, 0.01, FAKE, ha="right", va="bottom",
             fontsize=8, color="#b00", style="italic", alpha=0.85)

def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    c = np.cumsum(x)
    return float((n + 1 - 2*np.sum(c)/c[-1]) / n)

def lorenz_xy(x):
    x = np.sort(np.asarray(x, float))
    c = np.insert(np.cumsum(x), 0, 0)
    return np.linspace(0, 1, c.size), c / c[-1]

# ============================================================================
# A — BEHIND THE GINI NUMBER
#   concentrated env (few cliffs) vs flat env (all equal)
# ============================================================================
# fabricated per-state action-spreads for two envs
conc = np.concatenate([rng.uniform(0.0,0.08,80), rng.uniform(0.6,0.95,12)])  # lumpy
flat = rng.uniform(0.35,0.55,92)                                             # even
g_conc, g_flat = gini(conc), gini(flat)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# row 0: sorted bar-piles
for ax, data, title, gv in [(axes[0,0], conc, "CONCENTRATED env", g_conc),
                            (axes[0,1], flat, "FLAT env", g_flat)]:
    s = np.sort(data)[::-1]
    ax.bar(range(len(s)), s, color=CCE_C, width=1.0)
    ax.set_title(f"{title}\nsorted per-state stakes   (Gini = {gv:.2f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("states, sorted tall -> short"); ax.set_ylabel("stakes (action-spread)")
    ax.set_ylim(0, 1)
axes[0,0].annotate("a few cliffs\ncarry it all", (8, 0.8),
                   xytext=(30, 0.7), fontsize=9,
                   arrowprops=dict(arrowstyle="->"))
axes[0,1].text(46, 0.7, "every state\nabout equal\n-> nothing to chase",
               ha="center", fontsize=9, color="#8e0000")

# row 1: Lorenz curves (both on each axis for contrast)
for ax in (axes[1,0], axes[1,1]):
    ax.plot([0,1],[0,1], "--", color="gray", label="perfectly equal (Gini 0)")
for ax, data, gv, c in [(axes[1,0], conc, g_conc, CCE_C),
                        (axes[1,1], flat, g_flat, "#1976d2")]:
    lx, ly = lorenz_xy(data)
    ax.plot(lx, ly, color=c, lw=2.5, label=f"this env (Gini {gv:.2f})")
    ax.fill_between(lx, ly, lx, color=c, alpha=0.18)
    ax.set_xlabel("share of states (low -> high stakes)")
    ax.set_ylabel("share of TOTAL stakes")
    ax.set_title(f"Lorenz curve — area = Gini = {gv:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0,1); ax.set_ylim(0,1)
axes[1,0].annotate("big belly\n= lumpy\n= CCE has a target", (0.62,0.18),
                   fontsize=9, color=CCE_C)
axes[1,1].annotate("hugs the line\n= flat\n= no target", (0.55,0.3),
                   fontsize=9, color="#8e0000")

fig.suptitle("BEHIND THE GINI NUMBER — sorted pile (top) and Lorenz curve (bottom)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0.02,1,0.96]); stamp(fig)
fig.savefig(f"{OUT}/4_behind_gini.png", dpi=130); plt.close(fig)

# ============================================================================
# B — BEHIND THE SNR NUMBER
#   per-action rollout outcomes: within-move wiggle (noise) vs between (signal)
# ============================================================================
acts = ["UP","DOWN","LEFT","RIGHT"]
true_means = np.array([0.05, 0.80, 0.10, 1.00])   # the underlying signal
n_roll = 18

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

def strip(ax, within_noise, title):
    means_obs = []
    for i, m in enumerate(true_means):
        pts = np.clip(m + rng.normal(0, within_noise, n_roll), 0, 1)
        ax.scatter(pts, np.full(n_roll, i) + rng.normal(0,0.05,n_roll),
                   s=22, color=NOISE_C, alpha=0.7, zorder=2)
        ax.scatter([pts.mean()], [i], s=160, color=CCE_C, edgecolor="k",
                   zorder=3, marker="D")
        means_obs.append(pts.mean())
    means_obs = np.array(means_obs)
    signal = means_obs.std(); noise = within_noise
    snr = signal / noise if noise > 1e-6 else 99
    ax.set_yticks(range(4)); ax.set_yticklabels(acts)
    ax.set_xlim(-0.05, 1.05); ax.set_xlabel("rollout score (0 lose .. 1 win)")
    ax.set_title(f"{title}\nsignal(between)={signal:.2f}  noise(within)={noise:.2f}  "
                 f"SNR={'HUGE' if snr>50 else f'{snr:.1f}'}",
                 fontsize=10, fontweight="bold")
    return signal, noise

# panel 1: deterministic
strip(axes[0], 0.01, "DETERMINISTIC ice")
axes[0].text(0.5, 3.4, "dots stacked = same move\nsame result = NO noise\nsignal stands clear",
             ha="center", fontsize=8.5, color=CCE_C)
# panel 2: slippery
strip(axes[1], 0.30, "SLIPPERY ice")
axes[1].text(0.5, 3.4, "dots spread = same move\ndifferent luck = NOISE\nsignal buried",
             ha="center", fontsize=8.5, color="#8e0000")

# panel 3: variance decomposition bars for both
ax = axes[2]
labels = ["FL-det","FL-stoch"]; between=[0.43,0.40]; within=[0.01,0.30]
x = np.arange(2); w=0.35
ax.bar(x-w/2, between, w, color=CCE_C, label="signal (between moves)")
ax.bar(x+w/2, within, w, color=NOISE_C, label="noise (within a move)")
for i,(b,wi) in enumerate(zip(between,within)):
    snr = b/wi if wi>1e-6 else 99
    ax.text(i, max(b,wi)+0.02, f"SNR={'HUGE' if snr>50 else f'{snr:.1f}'}",
            ha="center", fontweight="bold", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,0.55)
ax.set_ylabel("return spread"); ax.legend(fontsize=8.5)
ax.set_title("the ratio = SNR\n(bottom-axis of the map)", fontsize=10, fontweight="bold")

fig.suptitle("BEHIND THE SNR NUMBER — each gray dot = one rollout, green diamond = move's average",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0,0.02,1,0.95]); stamp(fig)
fig.savefig(f"{OUT}/5_behind_snr.png", dpi=130); plt.close(fig)
print("wrote 4_behind_gini.png and 5_behind_snr.png to", OUT)
