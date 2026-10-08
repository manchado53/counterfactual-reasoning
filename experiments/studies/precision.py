import numpy as np
import matplotlib.pyplot as plt

def precision_at_k(oracle_vals, cce_vals, k):
    n = len(oracle_vals)
    top_n = max(1, int(n * k))
    top_oracle = set(np.argsort(oracle_vals)[-top_n:])
    top_cce    = set(np.argsort(cce_vals)[-top_n:])
    return len(top_oracle & top_cce) / top_n


def plot_c4_precision(oracle_vals, cce_by_seed, out_path, ks=(0.05, 0.10, 0.20)):
    """Bar chart: CCE Precision@K vs random baseline, trained policy, all seeds."""
    k_labels = [f'{int(k*100)}%' for k in ks]
    cce_means, cce_stds = [], []
    for k in ks:
        vals = [precision_at_k(oracle_vals, cce_by_seed[s], k) for s in cce_by_seed]
        cce_means.append(np.mean(vals))
        cce_stds.append(np.std(vals))

    x = np.arange(len(ks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x - w/2, cce_means, w, yerr=cce_stds, capsize=4,
                  color='#4CAF50', label='CCE (trained)')
    ax.bar(x + w/2, ks, w, color='#9E9E9E', label='Random baseline')

    for i, (mean, k) in enumerate(zip(cce_means, ks)):
        lift = mean / k
        ax.text(x[i] - w/2, mean + cce_stds[i] + 0.01, f'{lift:.1f}×',
                ha='center', va='bottom', fontsize=9, color='#2E7D32')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top {l}' for l in k_labels])
    ax.set_ylabel('Precision@K')
    ax.set_ylim(0, min(1.0, max(cce_means) + max(cce_stds) + 0.15))
    ax.set_title('Claim 1 — Precision@K: CCE vs Random Baseline\n(FrozenLake 8×8, fully trained)')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Fig C4 saved → {out_path}')
