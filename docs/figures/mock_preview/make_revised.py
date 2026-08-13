"""MOCK / FAKE-DATA — the REVISED metric set after adversarial review.
final_3_framework.png : EVB = Gain x Need tree + scorecard (adds DISTINCT-TD)
final_4_slipsweep.png : the hero experiment #1 (slip -> SNR -> CCE benefit)
All numbers fabricated to show FORMAT, not real results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.gridspec as gridspec

OUT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
GREEN,AMBER,RED,BLUE,GREY = "#2e7d32","#f9a825","#c62828","#1976d2","#9e9e9e"
FAKE = "ILLUSTRATIVE — FAKE DATA"; rng = np.random.default_rng(3)
def stamp(fig):
    fig.text(0.995,0.005,FAKE,ha="right",va="bottom",fontsize=8,color="#b00",style="italic",alpha=.85)

# ============================================================================
# FIGURE 3 — EVB = Gain x Need framework tree + revised scorecard
# ============================================================================
fig = plt.figure(figsize=(16,8.6))
gs = gridspec.GridSpec(1,2,width_ratios=[1.25,1],wspace=0.16,left=0.03,right=0.985,top=0.9,bottom=0.06)

# ---- left: the tree ----
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,text,fc,tc="white",fs=10,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                 fc=fc,ec="white",lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=tc,fontsize=fs,
            fontweight="bold" if bold else "normal")
def line(x1,y1,x2,y2,c=GREY):
    ax.plot([x1,x2],[y1,y2],color=c,lw=2,zorder=0)

box(3.4,8.7,3.2,1.0,"worth replaying\nEVB = GAIN × NEED","#37474f",fs=12)
# two halves
line(4.2,8.7,2.2,7.6); line(5.8,8.7,7.8,7.6)
box(0.7,6.8,3.0,0.9,"GAIN  (CCE estimates)\nhow much action matters",GREEN,fs=10)
box(6.3,6.8,3.0,0.9,"NEED  (we add)\nwill we go back?",AMBER,"black",fs=10)

# gain conditions
gconds = [
  ("Is GAIN real & visible?","SNR (greedy) · Concentration",BLUE,5.6),
  ("Does GAIN beat TD?","DISTINCT-TD  ★ NEW","#6a1b9a",4.3),
  ("Is GAIN reachable?","HORIZON-FIT (cf vs eff. horizon)",BLUE,3.0),
  ("Is GAIN faithful?","GAIN-FIDELITY vs Q*/EVB  (not vs C(s)!)",BLUE,1.7),
]
for q,m,c,yy in gconds:
    line(2.2,6.8,1.0,yy+0.45)
    box(0.5,yy,4.6,0.9,f"{q}\n{m}",c,fs=9.5)
# need + sampler
box(6.0,5.6,3.7,0.9,"NEED = successor-rep occupancy\n(visits of high-stakes states)",AMBER,"black",fs=9.5)
line(7.8,6.8,7.85,6.5)
box(6.0,4.0,3.7,0.9,"SAMPLER SANITY\nESS = collapse alarm only\nprecision@k / recall@k",GREY,fs=9)
ax.text(5,0.4,"★ DISTINCT-TD was MISSING — it is the only one that\n"
        "predicts CCE beats PER, not just 'prioritizing helps'.",
        ha="center",fontsize=10,color="#6a1b9a",fontweight="bold")
ax.set_title("Reframed: 5 ad-hoc numbers  →  conditions on GAIN × NEED",fontsize=13,fontweight="bold")

# ---- right: revised scorecard ----
ax = fig.add_subplot(gs[1]); ax.axis("off")
ax.set_title("REVISED SCORECARD",fontsize=12,fontweight="bold")
ENVS = {
 "FL-det":   dict(snr=8.5,conc=0.72,dtd=0.78,need=0.85,fid=0.80,verdict="WIN"),
 "FL-stoch": dict(snr=1.2,conc=0.55,dtd=0.74,need=0.80,fid=0.30,verdict="NULL"),
 "C4-mcts":  dict(snr=2.5,conc=0.48,dtd=0.40,need=0.62,fid=None,verdict="weak"),
}
rows = ["SNR (greedy)","Concentration","DISTINCT-TD ★","NEED","GAIN-fidelity","→ FORECAST"]
keys = ["snr","conc","dtd","need","fid","verdict"]
good = {"snr":lambda v:v>3,"conc":lambda v:v>0.5,"dtd":lambda v:v>0.5,
        "need":lambda v:v>0.7,"fid":lambda v:(v is not None and v>0.5)}
cols = list(ENVS); x0,cw=0.34,0.215; nr=len(rows); ch=1.0/(nr+1)
for j,c in enumerate(cols):
    ax.text(x0+cw*(j+0.5),1.0,c,ha="center",fontsize=10,fontweight="bold")
for i,(r,k) in enumerate(zip(rows,keys)):
    yy=1.0-ch*(i+1)
    ax.text(x0-0.02,yy,r,ha="right",va="center",fontsize=10,
            fontweight="bold" if k=="verdict" else "normal",
            color="#6a1b9a" if "DISTINCT" in r else "black")
    for j,c in enumerate(cols):
        d=ENVS[c]; xx=x0+cw*j
        if k=="verdict":
            fc={"WIN":"#c8e6c9","NULL":"#ffcdd2","weak":"#ffe0b2"}[d[k]]; txt=d[k]
        else:
            v=d[k]
            if v is None: fc="#eeeeee"; txt="n/a"
            else: fc="#c8e6c9" if good[k](v) else "#ffcdd2"; txt=f"{v:.2f}" if k!="snr" else f"{v:.1f}"
        ax.add_patch(Rectangle((xx,yy-ch/2),cw*0.95,ch*0.82,color=fc,ec="white"))
        ax.text(xx+cw*0.47,yy,txt,ha="center",va="center",fontsize=10,fontweight="bold")
ax.set_xlim(0,1); ax.set_ylim(0,1.05)
ax.text(0.5,-0.02,"C4 loses on DISTINCT-TD (0.40): CCE ≈ TD there → can't beat PER",
        ha="center",fontsize=9,color="#6a1b9a",transform=ax.transAxes)
fig.suptitle("REVISED METRIC SET — organized under EVB, with the missing DISTINCT-TD added",
             fontsize=14,fontweight="bold")
stamp(fig); fig.savefig(f"{OUT}/final_3_framework.png",dpi=130,bbox_inches="tight"); plt.close(fig)

# ============================================================================
# FIGURE 4 — the hero experiment #1: slip sweep (dose-response)
# ============================================================================
fig,axes = plt.subplots(1,2,figsize=(15,5.2))
slip = np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.33])
snr  = np.array([8.5,5.5,3.6,2.4,1.7,1.3,1.1])
benefit = np.array([0.85,0.62,0.40,0.22,0.10,0.03,-0.02])
ci = np.array([0.10,0.10,0.11,0.11,0.10,0.09,0.09])

# panel A: both vs slip
ax=axes[0]
l1=ax.plot(slip,benefit,"o-",color=GREEN,lw=2.4,label="CCE benefit (vs best baseline)")
ax.fill_between(slip,benefit-ci,benefit+ci,color=GREEN,alpha=0.15)
ax.axhline(0,color=GREY,ls=":")
ax.set_xlabel("slip probability  (more noise ->)"); ax.set_ylabel("CCE benefit (std. effect)",color=GREEN)
ax.tick_params(axis="y",labelcolor=GREEN)
ax2=ax.twinx(); l2=ax2.plot(slip,snr,"s--",color=BLUE,lw=2,label="SNR (forecast metric)")
ax2.set_ylabel("SNR",color=BLUE); ax2.tick_params(axis="y",labelcolor=BLUE)
ax.set_title("Experiment #1 — SLIP SWEEP\nnoise up -> SNR down -> CCE benefit down (together)",
             fontsize=11,fontweight="bold")
ax.annotate("det WIN\n(anchor)",(0.0,0.85),xytext=(0.04,0.6),fontsize=9,color=GREEN,
            arrowprops=dict(arrowstyle="->",color=GREEN))
ax.annotate("stoch NULL\n(anchor)",(0.25,0.03),xytext=(0.26,0.35),fontsize=9,color=RED,
            arrowprops=dict(arrowstyle="->",color=RED))
lines=l1+l2; ax.legend(lines,[l.get_label() for l in lines],fontsize=8.5,loc="upper right")

# panel B: benefit vs measured SNR (the real claim)
ax=axes[1]
ax.errorbar(snr,benefit,yerr=ci,fmt="o",color=GREEN,ecolor=GREY,capsize=3,ms=8)
# fit line
z=np.polyfit(snr,benefit,1); xs=np.linspace(1,9,50)
ax.plot(xs,np.polyval(z,xs),"--",color="#333")
ax.axhline(0,color=GREY,ls=":")
for s,b in zip(slip,benefit): pass
ax.set_xlabel("measured SNR"); ax.set_ylabel("CCE benefit (std. effect)")
ax.set_title("the claim: benefit tracks measured SNR\n(monotone = CONFIRM · flat = FALSIFY)",
             fontsize=11,fontweight="bold")
ax.text(0.97,0.05,"CONFIRM if monotone\n& crosses 0",ha="right",transform=ax.transAxes,
        fontsize=9,color=GREEN)
fig.suptitle("HERO EXPERIMENT — one env, one knob (slip). Turns N=4 anecdotes into a dose-response curve.",
             fontsize=13,fontweight="bold")
stamp(fig); fig.tight_layout(rect=[0,0.02,1,0.94]); fig.savefig(f"{OUT}/final_4_slipsweep.png",dpi=130); plt.close(fig)
print("wrote final_3_framework.png and final_4_slipsweep.png")
