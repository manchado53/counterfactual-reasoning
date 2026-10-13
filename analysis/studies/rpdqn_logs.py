import os
import numpy as np
R = None

def load(j):
    m = f'{R}/{j}/metrics.log'
    if not os.path.exists(m): return None
    cfg, ep, wr, ret = {}, [], [], []
    for l in open(m):
        if l.startswith('#'):
            k = l[1:].split(':', 1)
            if len(k) == 2: cfg[k[0].strip()] = k[1].strip()
            continue
        p = l.split()
        if not p or p[0] == 'episode': continue
        ep.append(int(p[0])); wr.append(float(p[3].rstrip('%'))/100); ret.append(float(p[5]))
    if len(ep) < 3: return None
    n = int(cfg.get('n_steps_per_update', 512))
    return dict(cfg=cfg, ep=np.array(ep), gs=np.array(ep)*24/n,
                wr=np.array(wr), ret=np.array(ret), nspu=n)


def smooth(y, w=7):
    """Centred moving average with EDGE-SAFE ends.

    np.convolve(..., mode='same') zero-pads, which drags the first and last few
    points toward zero and draws a cliff at both ends of every curve. Reflect the
    signal instead so the ends are averaged against real data.
    """
    if len(y) < w: return y
    pad = w // 2
    yp = np.pad(y, pad, mode='reflect')
    return np.convolve(yp, np.ones(w)/w, mode='valid')
