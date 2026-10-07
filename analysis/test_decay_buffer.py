import numpy as np
import pytest
from decay_buffer import DecayBuffer, blend_weight
from counterfactual_rl.agents.shared.consequence_buffers import ConsequenceReplayBuffer

@pytest.mark.parametrize('mode,mu',[('multiplicative',.5),('additive',.5),('additive',1.)])
def test_parity_endpoints_updates_and_wrap(mode,mu):
    kw=dict(capacity=7,normalise_priorities='rank',priority_mixing=mode,mu=mu)
    old,new=ConsequenceReplayBuffer(**kw),DecayBuffer(**kw)
    for i in range(23):
        for b in (old,new):
            b.add({'r':float(i%3)})
            ids=np.arange(len(b))
            b.update_consequence_scores(ids,np.sin(ids+i)**2)
            b.update_priorities(ids,np.cos(ids+i)*10)
        new.blend=0
        np.testing.assert_array_equal(new._compute_priorities(),old._compute_priorities())
        p=old._compute_priorities().copy();td=new.ranked_td()
        for w in (0.,.2,.5,1.,.5,0.):
            new.blend=w
            actual=new._compute_priorities()
            np.testing.assert_allclose(actual,(1-w)*p+w*td,rtol=1e-14)
            assert np.isfinite(actual).all() and (actual>0).all()
            np.testing.assert_allclose(actual.sum(),1.)
        batch=min(4,len(new))
        np.random.seed(123);a=old.sample(batch)
        np.random.seed(123);b=new.sample(batch)
        np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        new.blend=.5
        _,ids,weights=new.sample(batch)
        np.testing.assert_allclose(weights,1/(len(new)*new._compute_priorities()[ids]))

@pytest.mark.parametrize('episode,w',[(0,0),(99999,0),(100000,0),(125000,.5),(150000,1),(200000,1)])
def test_clock(episode,w):
    assert blend_weight(episode)==w
    assert blend_weight(episode,False)==0

def test_invalid_interval():
    with pytest.raises(ValueError):blend_weight(0,start=1,end=1)
