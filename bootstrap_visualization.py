"""
Bootstrap sample efficiency visualization.

Simulates what the analysis would look like with N=10 seeds per method,
using win rate trajectories calibrated to your actual 3m SMAX results.

Run with:  python bootstrap_visualization.py
Produces:  bootstrap_visualization.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Simulated win-rate trajectories
# Each method has a "true" learning curve + per-seed noise.
# Calibrated so that:
#   - Consequence-DQN learns faster early (sample efficiency)
#   - All methods converge to ~70% by 25k episodes
#   - Variance matches what you see in your 3m data (~15pp std)
# ---------------------------------------------------------------------------

CHECKPOINTS = np.array([5_000, 10_000, 15_000, 20_000, 25_000])
N_SEEDS = 10
N_BOOTSTRAP = 10_000

def sigmoid_curve(x, x_mid, steepness, floor, ceiling):
    return floor + (ceiling - floor) / (1 + np.exp(-steepness * (x - x_mid)))

def simulate_runs(mean_curve, noise_std, n_seeds, rng):
    """Each seed gets a slightly shifted version of the mean curve + noise."""
    runs = []
    for _ in range(n_seeds):
        seed_offset = rng.normal(0, 0.04)           # per-seed bias
        noise = rng.normal(0, noise_std, len(mean_curve))
        run = np.clip(mean_curve + seed_offset + noise, 0.05, 0.98)
        runs.append(run)
    return np.array(runs)   # shape: (n_seeds, n_checkpoints)

x = CHECKPOINTS / 1000  # in thousands for the sigmoid

# Consequence-DQN: fast early ramp, converges ~72%
consequence_mean = sigmoid_curve(x, x_mid=6, steepness=0.55, floor=0.30, ceiling=0.74)

# PER baseline: slower ramp, converges ~70%
per_mean = sigmoid_curve(x, x_mid=10, steepness=0.45, floor=0.25, ceiling=0.71)

# Uniform DQN: slowest, most variance, converges ~65%
uniform_mean = sigmoid_curve(x, x_mid=12, steepness=0.40, floor=0.20, ceiling=0.67)

consequence_runs = simulate_runs(consequence_mean, noise_std=0.06, n_seeds=N_SEEDS, rng=rng)
per_runs         = simulate_runs(per_mean,         noise_std=0.09, n_seeds=N_SEEDS, rng=rng)
uniform_runs     = simulate_runs(uniform_mean,     noise_std=0.10, n_seeds=N_SEEDS, rng=rng)

# ---------------------------------------------------------------------------
# Bootstrap P(A > B) at each checkpoint
# ---------------------------------------------------------------------------

def bootstrap_prob_a_beats_b(a_data, b_data, n_bootstrap=N_BOOTSTRAP, rng=rng):
    """
    a_data, b_data: arrays of shape (n_seeds,)
    Returns P(mean of resample from A > mean of resample from B)
    """
    n = len(a_data)
    count = 0
    for _ in range(n_bootstrap):
        a_sample = rng.choice(a_data, size=n, replace=True).mean()
        b_sample = rng.choice(b_data, size=n, replace=True).mean()
        if a_sample > b_sample:
            count += 1
    return count / n_bootstrap

def bootstrap_ci_on_prob(a_data, b_data, n_outer=200, n_bootstrap=500, rng=rng):
    """
    Compute a 90% CI on the P(A>B) estimate itself via an outer bootstrap.
    This shows how uncertain the probability estimate is.
    """
    n = len(a_data)
    probs = []
    for _ in range(n_outer):
        a_resample = rng.choice(a_data, size=n, replace=True)
        b_resample = rng.choice(b_data, size=n, replace=True)
        p = bootstrap_prob_a_beats_b(a_resample, b_resample, n_bootstrap=n_bootstrap, rng=rng)
        probs.append(p)
    return np.percentile(probs, 5), np.percentile(probs, 95)

print("Computing bootstrap probabilities... (this takes ~30s)")
prob_vs_per     = []
prob_vs_uniform = []
ci_vs_per       = []
ci_vs_uniform   = []

for i in range(len(CHECKPOINTS)):
    c = consequence_runs[:, i]
    p = per_runs[:, i]
    u = uniform_runs[:, i]

    prob_vs_per.append(bootstrap_prob_a_beats_b(c, p))
    prob_vs_uniform.append(bootstrap_prob_a_beats_b(c, u))
    ci_vs_per.append(bootstrap_ci_on_prob(c, p))
    ci_vs_uniform.append(bootstrap_ci_on_prob(c, u))

prob_vs_per     = np.array(prob_vs_per)
prob_vs_uniform = np.array(prob_vs_uniform)
ci_vs_per       = np.array(ci_vs_per)       # (n_checkpoints, 2)
ci_vs_uniform   = np.array(ci_vs_uniform)

# Bootstrap distributions at 10k checkpoint for the distribution panel
FOCUS_IDX = 1   # 10k episodes
def get_bootstrap_dist(data, n_bootstrap=N_BOOTSTRAP, rng=rng):
    n = len(data)
    return np.array([rng.choice(data, size=n, replace=True).mean()
                     for _ in range(n_bootstrap)])

boot_consequence_10k = get_bootstrap_dist(consequence_runs[:, FOCUS_IDX])
boot_per_10k         = get_bootstrap_dist(per_runs[:, FOCUS_IDX])
boot_uniform_10k     = get_bootstrap_dist(uniform_runs[:, FOCUS_IDX])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

COLORS = {
    "consequence": "#2196F3",   # blue
    "per":         "#FF5722",   # orange-red
    "uniform":     "#9E9E9E",   # grey
}

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0f0f0f")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       top=0.90, bottom=0.07, left=0.07, right=0.97)

ax_curves = fig.add_subplot(gs[0, 0])
ax_dist   = fig.add_subplot(gs[0, 1])
ax_prob   = fig.add_subplot(gs[1, :])

for ax in [ax_curves, ax_dist, ax_prob]:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

fig.suptitle("Bootstrap Sample Efficiency Analysis — SMAX 3m (Simulated, N=10 seeds)",
             color="white", fontsize=14, fontweight="bold")

# ── Panel 1: Raw learning curves ──────────────────────────────────────────
cp_k = CHECKPOINTS / 1000

for i, seed_run in enumerate(consequence_runs):
    ax_curves.plot(cp_k, seed_run, color=COLORS["consequence"], alpha=0.18, linewidth=1)
for i, seed_run in enumerate(per_runs):
    ax_curves.plot(cp_k, seed_run, color=COLORS["per"], alpha=0.18, linewidth=1)
for i, seed_run in enumerate(uniform_runs):
    ax_curves.plot(cp_k, seed_run, color=COLORS["uniform"], alpha=0.18, linewidth=1)

ax_curves.plot(cp_k, consequence_runs.mean(0), color=COLORS["consequence"],
               linewidth=2.5, label="Consequence-DQN")
ax_curves.plot(cp_k, per_runs.mean(0), color=COLORS["per"],
               linewidth=2.5, label="DQN + PER")
ax_curves.plot(cp_k, uniform_runs.mean(0), color=COLORS["uniform"],
               linewidth=2.5, label="DQN Uniform")

ax_curves.axvline(x=10, color="yellow", linestyle="--", alpha=0.6, linewidth=1)
ax_curves.text(10.3, 0.28, "10k\n(zoom →)", color="yellow", fontsize=8, va="bottom")

ax_curves.set_xlabel("Training episodes (thousands)")
ax_curves.set_ylabel("Win rate")
ax_curves.set_title("Panel 1 — Learning curves (all seeds)")
ax_curves.legend(facecolor="#111", labelcolor="white", framealpha=0.8, fontsize=9)
ax_curves.set_ylim(0.1, 1.0)
ax_curves.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# ── Panel 2: Bootstrap distributions at 10k ───────────────────────────────
bins = np.linspace(0.25, 0.90, 55)

ax_dist.hist(boot_consequence_10k, bins=bins, color=COLORS["consequence"],
             alpha=0.7, label="Consequence-DQN", density=True)
ax_dist.hist(boot_per_10k, bins=bins, color=COLORS["per"],
             alpha=0.7, label="DQN + PER", density=True)
ax_dist.hist(boot_uniform_10k, bins=bins, color=COLORS["uniform"],
             alpha=0.7, label="DQN Uniform", density=True)

# Shade the overlap region between consequence and PER
overlap_lo = max(boot_consequence_10k.min(), boot_per_10k.min())
overlap_hi = min(boot_consequence_10k.max(), boot_per_10k.max())
ax_dist.axvspan(overlap_lo, overlap_hi, color="white", alpha=0.05)

p_c_vs_p = (boot_consequence_10k > boot_per_10k).mean()
ax_dist.text(0.97, 0.95,
             f"P(consequence > PER)\n= {p_c_vs_p:.2f}",
             transform=ax_dist.transAxes, ha="right", va="top",
             color="white", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", edgecolor="#555"))

ax_dist.set_xlabel("Bootstrap mean win rate @ 10k episodes")
ax_dist.set_ylabel("Density")
ax_dist.set_title("Panel 2 — Bootstrap distributions at 10k episodes")
ax_dist.legend(facecolor="#111", labelcolor="white", framealpha=0.8, fontsize=9)
ax_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# ── Panel 3: P(consequence > baseline) over training ──────────────────────
ax_prob.axhline(y=0.5, color="white", linestyle="--", alpha=0.4, linewidth=1,
                label="Chance (P = 0.5)")
ax_prob.axhline(y=0.8, color="yellow", linestyle=":", alpha=0.3, linewidth=1)
ax_prob.text(cp_k[-1] + 0.2, 0.80, "P=0.8\n(strong)", color="yellow",
             fontsize=8, va="center")

# Consequence vs PER
ax_prob.fill_between(cp_k, ci_vs_per[:, 0], ci_vs_per[:, 1],
                     color=COLORS["per"], alpha=0.2)
ax_prob.plot(cp_k, prob_vs_per, color=COLORS["per"], linewidth=2.5,
             marker="o", markersize=7, label="P(consequence > PER)")

# Consequence vs Uniform
ax_prob.fill_between(cp_k, ci_vs_uniform[:, 0], ci_vs_uniform[:, 1],
                     color=COLORS["uniform"], alpha=0.2)
ax_prob.plot(cp_k, prob_vs_uniform, color=COLORS["uniform"], linewidth=2.5,
             marker="s", markersize=7, label="P(consequence > Uniform)")

# Annotate widest CI to illustrate uncertainty from N=10
widest_idx = np.argmax(ci_vs_per[:, 1] - ci_vs_per[:, 0])
lo, hi = ci_vs_per[widest_idx]
ax_prob.annotate(
    f"90% CI on P itself:\n[{lo:.2f}, {hi:.2f}]\n← uncertainty from N=10",
    xy=(cp_k[widest_idx], prob_vs_per[widest_idx]),
    xytext=(cp_k[widest_idx] + 1.5, prob_vs_per[widest_idx] - 0.12),
    color="white", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="white", lw=1),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", edgecolor="#888")
)

ax_prob.set_xlabel("Training episodes (thousands)")
ax_prob.set_ylabel("P(Consequence-DQN wins)")
ax_prob.set_title("Panel 3 — Sample efficiency: probability consequence-DQN beats baseline at each checkpoint\n"
                  "(shaded band = 90% CI on the probability estimate itself, from outer bootstrap)")
ax_prob.legend(facecolor="#111", labelcolor="white", framealpha=0.8, fontsize=10)
ax_prob.set_ylim(0.0, 1.05)
ax_prob.set_xlim(cp_k[0] - 0.5, cp_k[-1] + 0.5)
ax_prob.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

plt.savefig("bootstrap_visualization.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: bootstrap_visualization.png")
plt.show()
