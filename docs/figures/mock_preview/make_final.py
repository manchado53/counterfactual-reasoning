"""MOCK / FAKE-DATA — the FINAL metric set we will track.
One forecast map + one 6-panel dashboard covering all 5 metrics.
All numbers fabricated to show FORMAT, not real results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

OUT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
GREEN, AMBER, RED, BLUE, GREY = "#2e7d32", "#f9a825", "#c62828", "#1976d2", "#9e9e9e"
FAKE = "ILLUSTRATIVE — FAKE DATA"
rng = np.random.default_rng(7)

def stamp(fig):
    fig.text(0.995, 0.005, FAKE, ha="right", va="bottom",
             fontsize=8, color="#b00", style="italic", alpha=0.85)

# fabricated per-env summary (consistent with known outcomes)
ENVS = {
    "FL-det":   dict(c=GREEN, gini=0.55, snr=8.5,  need=0.85, rho=0.78, ess=0.62, verdict="WIN"),
    "FL-stoch": dict(c=RED,   gini=0.46, snr=1.2,  need=0.80, rho=0.30, ess=0.60, verdict="NULL"),
    "C4-mcts":  dict(c=AMBER, gini=0.40, snr=2.5,  need=0.62, rho=0.36, ess=0.55, verdict="weak"),
}

def lorenz(x):
    x = np.sort(np.asarray(x,float)); c = np.insert(np.cumsum(x),0,0)
    return np.linspace(0,1,c.size), c/c[-1]

def gini_sample(g, n=120):
    # build a fake spread-distribution with roughly the target gini
    if g > 0.5:   d = np.concatenate([rng.uniform(0,.08,int(n*.85)), rng.uniform(.5,1,int(n*.15))])
    elif g > 0.42:d = np.concatenate([rng.uniform(0,.2,int(n*.7)),  rng.uniform(.4,.9,int(n*.3))])
    else:         d = rng.uniform(.2,.7,n)
    return d

# ============================================================================
# FIGURE 1 — the forecast map (hero). 2 axes + NEED as a gate note.
# ============================================================================
fig, ax = plt.subplots(figsize=(8.6,6.8))
ax.add_patch(Rectangle((3.0,0.40),8,0.35,color="#c8e6c9",alpha=0.55,zorder=0))
ax.text(7.0,0.725,"likely CCE WIN\nforks exist + visible",ha="center",va="top",
        fontsize=11,color=GREEN,fontweight="bold")
ax.text(1.4,0.12,"no gain\nfoggy or flat",ha="center",fontsize=10,color="#8e0000")
for name,d in ENVS.items():
    x,y = d["snr"], d["gini"]
    ax.scatter([x],[y],s=300,color=d["c"],edgecolor="k",lw=1.2,zorder=5)
    ax.annotate(f"{name}\n[{d['verdict']}]",(x,y),textcoords="offset points",
                xytext=(12,10),fontsize=10,fontweight="bold",color=d["c"])
ax.scatter([5.0],[0.5],s=340,marker="*",color="k",zorder=6)
ax.annotate("NEW env?\nmeasure -> drop here",(5.0,0.5),textcoords="offset points",
            xytext=(14,-40),fontsize=9.5,arrowprops=dict(arrowstyle="->"))
ax.axvline(3.0,color="gray",ls=":",lw=1)
ax.set_xlim(0.5,11); ax.set_ylim(0.05,0.78)
ax.set_xlabel("SNR  — can we SEE the forks? (clear ->)",fontsize=11)
ax.set_ylabel("CONCENTRATION — are there forks? (more ->)",fontsize=11)
ax.set_title("FORECAST MAP   (3rd gate: NEED — are the forks actually visited?)",
             fontsize=12,fontweight="bold")
stamp(fig); fig.tight_layout(); fig.savefig(f"{OUT}/final_1_map.png",dpi=130); plt.close(fig)

# ============================================================================
# FIGURE 2 — the tracking dashboard, 5 metrics
# ============================================================================
fig = plt.figure(figsize=(16,9))
gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.28,
                       left=0.06,right=0.97,top=0.88,bottom=0.08)

# ---- (1) Concentration : Lorenz curves ----
ax = fig.add_subplot(gs[0,0])
ax.plot([0,1],[0,1],"--",color=GREY,lw=1)
for name,d in ENVS.items():
    lx,ly = lorenz(gini_sample(d["gini"]))
    ax.plot(lx,ly,color=d["c"],lw=2.3,label=f"{name}  G={d['gini']:.2f}")
ax.set_title("1 · CONCENTRATION\nare there forks?  (Lorenz, area=Gini)",fontsize=10.5,fontweight="bold")
ax.set_xlabel("share of states"); ax.set_ylabel("share of total stakes")
ax.legend(fontsize=8,loc="upper left"); ax.set_xlim(0,1); ax.set_ylim(0,1)

# ---- (2) SNR : between vs within bars ----
ax = fig.add_subplot(gs[0,1])
names = list(ENVS); x = np.arange(len(names)); w=0.36
between = [0.85,0.80,0.70]; within=[b/ENVS[n]["snr"] for b,n in zip(between,names)]
ax.bar(x-w/2,between,w,color=GREEN,label="signal (between actions)")
ax.bar(x+w/2,within,w,color=GREY,label="noise (within an action)")
for i,n in enumerate(names):
    ax.text(i,0.9,f"SNR\n{ENVS[n]['snr']:.1f}",ha="center",fontsize=9,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names,fontsize=9); ax.set_ylim(0,1.05)
ax.set_ylabel("return spread")
ax.set_title("2 · SNR\ncan we SEE the forks?",fontsize=10.5,fontweight="bold")
ax.legend(fontsize=7.5,loc="upper right")

# ---- (3) NEED-overlap : stakes vs visits scatter ----
ax = fig.add_subplot(gs[0,2])
# trap zone = high stakes, low visits
ax.add_patch(Rectangle((0,0.6),0.32,0.4,color="#ffcdd2",alpha=0.7,zorder=0))
ax.text(0.16,0.93,"TRAP\nhigh stakes\nnever visited",ha="center",va="top",
        fontsize=8.5,color="#8e0000")
# FL-det: stakes line up with visits (good) -> dots avoid trap
vis = rng.uniform(0,1,60); stk = np.clip(0.2+0.7*vis+rng.normal(0,0.12,60),0,1)
ax.scatter(vis,stk,s=30,color=GREEN,edgecolor="k",alpha=0.75,label="FL-det states")
# a few trap states for contrast
tv = rng.uniform(0,0.25,7); ts = rng.uniform(0.65,0.95,7)
ax.scatter(tv,ts,s=42,color=RED,marker="x",label="trap states")
ax.set_xlabel("visit frequency  (NEED)"); ax.set_ylabel("stakes  C(s)")
ax.set_title("3 · NEED-OVERLAP\nare the forks actually driven?",fontsize=10.5,fontweight="bold")
ax.legend(fontsize=7.5,loc="lower right"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.97,0.04,"overlap=0.85",ha="right",fontsize=9,color=GREEN,transform=ax.transAxes)

# ---- (4) Score<->Stakes : scatter + rho ----
ax = fig.add_subplot(gs[1,0])
truth = rng.uniform(0,1,80); cce = np.clip(truth*0.8+rng.normal(0,0.13,80),0,1)
ax.scatter(truth,cce,s=26,color=BLUE,edgecolor="k",alpha=0.7)
ax.plot([0,1],[0,1],"--",color=GREY)
ax.set_xlabel("true stakes C(s)"); ax.set_ylabel("CCE priority")
ax.set_title("4 · SCORE <-> STAKES\ndoes CCE point at the forks?  (rho=0.78)",
             fontsize=10.5,fontweight="bold")
ax.set_xlim(0,1); ax.set_ylim(0,1)

# ---- (5) ESS over training : spread / don't-obsess ----
ax = fig.add_subplot(gs[1,1])
t = np.linspace(0,1,100)
ess_ok  = 0.6+0.05*np.sin(t*9)
ess_bad = 0.6*np.exp(-t*3)+0.05
ax.axhspan(0,0.15,color="#ffcdd2",alpha=0.6); ax.text(0.5,0.07,"danger: too spiky (forgets)",
        ha="center",fontsize=8.5,color="#8e0000")
ax.plot(t,ess_ok,color=GREEN,lw=2,label="CCE (smooth lean) — healthy")
ax.plot(t,ess_bad,color=RED,lw=2,ls="--",label="too peaked — collapses")
ax.set_xlabel("training (norm)"); ax.set_ylabel("ESS  (1 / Σ pᵢ²,  norm)")
ax.set_title("5 · ESS / OVERLAP\nleaning, not obsessing?",fontsize=10.5,fontweight="bold")
ax.legend(fontsize=7.5,loc="upper right"); ax.set_ylim(0,1)

# ---- (6) scorecard table ----
ax = fig.add_subplot(gs[1,2]); ax.axis("off")
ax.set_title("VERDICT SCORECARD",fontsize=11,fontweight="bold")
rows = ["Concentration","SNR","NEED-overlap","Score↔Stakes","ESS ok?","→ FORECAST"]
cols = list(ENVS)
def cellcolor(metric,env):
    d = ENVS[env]
    good = {"Concentration":d["gini"]>0.45,"SNR":d["snr"]>3,"NEED-overlap":d["need"]>0.7,
            "Score↔Stakes":d["rho"]>0.5,"ESS ok?":d["ess"]>0.3}
    if metric=="→ FORECAST": return {"WIN":"#c8e6c9","NULL":"#ffcdd2","weak":"#ffe0b2"}[d["verdict"]]
    return "#c8e6c9" if good[metric] else "#ffcdd2"
def celltext(metric,env):
    d = ENVS[env]
    return {"Concentration":f"{d['gini']:.2f}","SNR":f"{d['snr']:.1f}","NEED-overlap":f"{d['need']:.2f}",
            "Score↔Stakes":f"{d['rho']:.2f}","ESS ok?":"yes" if d['ess']>0.3 else "no",
            "→ FORECAST":d["verdict"]}[metric]
nr,nc = len(rows),len(cols); x0,y0,cw,ch = 0.30,0.0,0.225,1.0/(nr+1)
for j,cenv in enumerate(cols):
    ax.text(x0+cw*(j+0.5),1.0,cenv,ha="center",va="center",fontsize=9,fontweight="bold")
for i,m in enumerate(rows):
    yy = 1.0-ch*(i+1)
    ax.text(x0-0.02,yy,m,ha="right",va="center",fontsize=9,
            fontweight="bold" if m=="→ FORECAST" else "normal")
    for j,cenv in enumerate(cols):
        xx = x0+cw*j
        ax.add_patch(Rectangle((xx,yy-ch/2),cw*0.95,ch*0.85,
                     color=cellcolor(m,cenv),ec="white"))
        ax.text(xx+cw*0.47,yy,celltext(m,cenv),ha="center",va="center",fontsize=9,
                fontweight="bold")
ax.set_xlim(0,1); ax.set_ylim(0,1.05)

fig.suptitle("CCE METRIC DASHBOARD  —  3 forecast metrics (top)  +  2 algorithm checks (bottom)",
             fontsize=14,fontweight="bold")
stamp(fig); fig.savefig(f"{OUT}/final_2_dashboard.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("wrote final_1_map.png and final_2_dashboard.png")
