"""MOCK / FAKE-DATA — dashboard whose LAYOUT IS THE TREE.
Rows grouped under GAIN / NEED / SANITY. Each metric: the question, a mini-plot
of what we measure, and the per-env reading (green/red). final_5_treedash.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT = "/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
GREEN,AMBER,RED,BLUE,GREY,PURP = "#2e7d32","#f9a825","#c62828","#1976d2","#9e9e9e","#6a1b9a"
GOODC,BADC,NAC = "#c8e6c9","#ffcdd2","#eeeeee"
rng = np.random.default_rng(11)

# per-env readings + good thresholds
ENVS = ["FL-det","FL-stoch","C4-mcts"]
DATA = {
 "conc": (dict(zip(ENVS,[0.72,0.55,0.48])), lambda v:v>0.5,  "{:.2f}"),
 "snr":  (dict(zip(ENVS,[8.5,1.2,2.5])),    lambda v:v>3,    "{:.1f}"),
 "dtd":  (dict(zip(ENVS,[0.78,0.74,0.40])), lambda v:v>0.5,  "{:.2f}"),
 "fid":  (dict(zip(ENVS,[0.80,0.30,None])), lambda v:(v is not None and v>0.5), "{:.2f}"),
 "hor":  (dict(zip(ENVS,[0.90,0.90,0.70])), lambda v:v>0.8,  "{:.2f}"),
 "need": (dict(zip(ENVS,[0.85,0.80,0.62])), lambda v:v>0.7,  "{:.2f}"),
 "ess":  (dict(zip(ENVS,[1,1,1])),          lambda v:v>0,    "ok"),
}

fig = plt.figure(figsize=(15,12))
bg = fig.add_axes([0,0,1,1]); bg.axis("off"); bg.set_xlim(0,1); bg.set_ylim(0,1)

# layout
items = [
 ("h","GAIN  —  is CCE's guess any good?",GREEN),
 ("m","Concentration","are there forks at all?","conc","lorenz"),
 ("m","SNR (greedy)","fork louder than the noise?","snr","snrbars"),
 ("m","DISTINCT-TD  ★","different from PER/TD?  (else just slow PER)","dtd","scat_lo"),
 ("m","GAIN-fidelity","is the guess correct?  (FrozenLake only)","fid","scat_di"),
 ("m","HORIZON-fit","does the rollout reach the payoff?","hor","horbars"),
 ("h","NEED  —  the half CCE forgets",AMBER),
 ("m","NEED","do the forks actually get visited?","need","needsc"),
 ("h","SANITY",GREY),
 ("m","ESS / precision@k","did the draws collapse?  (alarm only)","ess","essbars"),
]
top, bot = 0.935, 0.035
unit = (top-bot) / (3*0.46 + 7*1.0 + 1.05)   # +1.05 reserves the forecast strip
HH, MH = 0.46*unit, 1.0*unit
xL = 0.015                 # label left
xV0, xVW = 0.34, 0.14      # mini-viz left/width
envX = {"FL-det":0.575,"FL-stoch":0.725,"C4-mcts":0.875}; cw=0.13

# env column headers
for e,xc in envX.items():
    bg.text(xc, top+0.012, e, ha="center", va="bottom", fontsize=12, fontweight="bold")
bg.text(0.015, top+0.012, "metric", fontsize=11, color=GREY)
bg.text(0.34, top+0.012, "what we measure", fontsize=11, color=GREY)
fig.suptitle("CCE DASHBOARD  —  the LAYOUT is the tree:  GAIN checks  +  NEED  +  SANITY",
             fontsize=15, fontweight="bold", y=0.985)

def miniax(ytop, h):
    return fig.add_axes([xV0, ytop-h, xVW, h*0.92])

def draw_viz(kind, ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
    if kind=="lorenz":
        x=np.linspace(0,1,30); y=x**2.6
        ax.plot([0,1],[0,1],"--",color=GREY,lw=1); ax.plot(x,y,color=GREEN,lw=2)
        ax.fill_between(x,y,x,color=GREEN,alpha=.15)
    elif kind=="snrbars":
        ax.bar([0,1],[0.85,0.12],color=[GREEN,GREY],width=.7); ax.set_ylim(0,1)
        ax.text(0.5,0.9,"sig vs noise",ha="center",fontsize=7,transform=ax.transAxes)
    elif kind=="scat_lo":
        x=rng.uniform(0,1,40); y=rng.uniform(0,1,40)   # uncorrelated = distinct
        ax.scatter(x,y,s=10,color=PURP,alpha=.7); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.5,1.06,"CCE vs TD (want NO line)",ha="center",fontsize=6.5,transform=ax.transAxes)
    elif kind=="scat_di":
        x=rng.uniform(0,1,40); y=np.clip(x+rng.normal(0,.1,40),0,1)
        ax.plot([0,1],[0,1],"--",color=GREY,lw=1); ax.scatter(x,y,s=10,color=BLUE,alpha=.7)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.5,1.06,"CCE vs true Q* (want a line)",ha="center",fontsize=6.5,transform=ax.transAxes)
    elif kind=="horbars":
        ax.bar([0,1],[0.9,1.0],color=[BLUE,GREY],width=.7); ax.set_ylim(0,1.1)
        ax.text(0.5,0.95,"cf-horizon vs needed",ha="center",fontsize=6.5,transform=ax.transAxes)
    elif kind=="needsc":
        ax.add_patch(Rectangle((0,0.6),0.3,0.4,color=BADC,alpha=.7))
        vis=rng.uniform(0,1,30); stk=np.clip(0.2+0.7*vis+rng.normal(0,.1,30),0,1)
        ax.scatter(vis,stk,s=10,color=AMBER,edgecolor="k",lw=.3)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.5,1.06,"stakes vs visits (red=trap)",ha="center",fontsize=6.5,transform=ax.transAxes)
    elif kind=="essbars":
        ax.bar(range(8),[.5,.4,.45,.38,.5,.42,.47,.4],color=GREY,width=.8); ax.set_ylim(0,1)
        ax.axhline(.1,color=RED,ls=":"); ax.text(0.5,0.9,"flat=healthy",ha="center",fontsize=7,transform=ax.transAxes)

y = top
for it in items:
    if it[0]=="h":
        _,txt,c = it
        bg.add_patch(FancyBboxPatch((0.008,y-HH*0.86),0.985,HH*0.8,
                     boxstyle="round,pad=0.004,rounding_size=0.01",fc=c,ec="none",alpha=0.9))
        bg.text(0.018,y-HH*0.46,txt,va="center",fontsize=12.5,fontweight="bold",
                color="black" if c==AMBER else "white")
        y -= HH
    else:
        _,name,q,key,viz = it
        yc = y-MH*0.5
        bg.text(xL,yc+0.013,name,va="center",fontsize=11.5,fontweight="bold",
                color=PURP if "DISTINCT" in name else "black")
        bg.text(xL,yc-0.016,q,va="center",fontsize=9,color="#555")
        ax=miniax(y,MH); draw_viz(viz,ax)
        d,good,fmt = DATA[key]
        for e,xc in envX.items():
            v=d[e]
            if v is None: fc,txt=NAC,"n/a"
            else: fc,txt=(GOODC if good(v) else BADC),(fmt.format(v) if fmt!="ok" else "ok")
            bg.add_patch(Rectangle((xc-cw/2,yc-MH*0.34),cw,MH*0.6,color=fc,ec="white"))
            bg.text(xc,yc,txt,ha="center",va="center",fontsize=12,fontweight="bold")
        bg.plot([0.008,0.992],[y-MH,y-MH],color="#eaeaea",lw=.8)
        y -= MH

# forecast strip (drawn just below the last row, using running y)
fy = y
bg.add_patch(Rectangle((0.008,fy-unit*0.92),0.985,unit*0.9,fc="#263238",ec="none"))
verd={"FL-det":("WIN",GOODC),"FL-stoch":("NULL",BADC),"C4-mcts":("weak","#ffe0b2")}
bg.text(0.018,fy-unit*0.47,"→ FORECAST",va="center",fontsize=12.5,fontweight="bold",color="white")
for e,xc in envX.items():
    t,c=verd[e]
    bg.add_patch(Rectangle((xc-cw/2,fy-unit*0.78),cw,unit*0.62,color=c,ec="white"))
    bg.text(xc,fy-unit*0.47,t,ha="center",va="center",fontsize=12.5,fontweight="bold")
bg.text(0.5,0.012,"ILLUSTRATIVE — FAKE DATA",ha="center",fontsize=8,color="#b00",style="italic")
fig.savefig(f"{OUT}/final_5_treedash.png",dpi=125); plt.close(fig)
print("wrote final_5_treedash.png")
