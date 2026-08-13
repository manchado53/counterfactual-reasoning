"""Dead-simple Lorenz explainer with a 5-state worked example."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning/docs/figures/mock_preview"
GREEN,GREY,RED="#2e7d32","#9e9e9e","#c62828"
names=["A","B","C","D","E"]

def lorenz(stakes):
    s=np.sort(stakes); c=np.insert(np.cumsum(s),0,0)
    x=np.linspace(0,1,len(c)); y=c/c[-1]; return x,y

fig,ax=plt.subplots(2,2,figsize=(12,8))
for row,(stakes,title,col) in enumerate([
    ([2,2,2,2,2],"EVEN world — every state matters the same",GREEN),
    ([0,0,0,0,10],"PILED world — one state holds ALL the stakes",RED)]):
    # bars
    a=ax[row,0]
    a.bar(names,stakes,color=col,width=.7)
    a.set_title(title,fontsize=12,fontweight="bold")
    a.set_ylabel("stakes  C(s)\n(how much the move matters)")
    a.set_xlabel("the 5 states"); a.set_ylim(0,10.5)
    for i,v in enumerate(stakes): a.text(i,v+0.2,str(v),ha="center",fontweight="bold")
    # lorenz
    b=ax[row,1]
    x,y=lorenz(stakes)
    b.plot([0,1],[0,1],"--",color=GREY,lw=1.5,label="perfectly even")
    b.plot(x,y,"o-",color=col,lw=2.5,label="this world")
    b.fill_between(x,y,x,color=col,alpha=0.15)
    b.set_xlabel("share of states  (counted low → high stakes)")
    b.set_ylabel("share of TOTAL stakes")
    b.set_xlim(0,1); b.set_ylim(0,1); b.legend(fontsize=9,loc="upper left")
    g=1-2*np.trapz(y,x)
    if row==0:
        b.set_title("→ straight line, NO belly  =  Gini ≈ 0",fontsize=12,fontweight="bold")
        b.annotate("line sits ON the diagonal\n= even = nothing to prioritize",
                   (0.55,0.45),fontsize=9.5,color=col)
    else:
        b.set_title(f"→ big BELLY  =  Gini high (≈{g:.2f})",fontsize=12,fontweight="bold")
        b.annotate("crawls along bottom,\nthen jumps at the end\n(E dumps in all 10)",
                   (0.18,0.55),fontsize=9.5,color=col)
        b.annotate("BELLY = the gap\n= the Gini number",(0.6,0.18),fontsize=9.5,color=col,
                   fontweight="bold")
fig.suptitle("The first graph (Lorenz): are stakes EVEN (diagonal) or PILED on a few (big belly)?",
             fontsize=13.5,fontweight="bold")
fig.text(0.99,0.01,"ILLUSTRATIVE — FAKE DATA",ha="right",fontsize=8,color="#b00",style="italic")
fig.tight_layout(rect=[0,0.02,1,0.96])
fig.savefig(f"{OUT}/lorenz_explained.png",dpi=130); print("wrote lorenz_explained.png")
