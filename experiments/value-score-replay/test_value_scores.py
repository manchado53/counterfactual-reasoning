import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from value_scores import action_gaps, ValueBase, FrozenLakeValue, JaxNavValue

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]


def test_gaps_negative_ties_and_shift_invariance():
    q = np.array([[5,5,5],[-4,1,7],[0,5,9]])
    actions = [1,0,1]
    np.testing.assert_allclose(action_gaps(q,actions), [0,11,5])
    np.testing.assert_allclose(action_gaps(q+100,actions), [0,11,5])


def test_weights_exclude_taken_and_fallback():
    q = [[0,5,9],[0,5,9]]
    p = [[.25,.5,.25],[0,1,0]]
    np.testing.assert_allclose(action_gaps(q,[1,1],p),[4.5,4.5])


def test_target_network_and_parent_key_advancement():
    class Fake(ValueBase):
        network = SimpleNamespace(apply=lambda params, x: x+params)
        target_params = 7.
        params = 100.
        _key = jax.random.PRNGKey(42)
    a = Fake()
    expected_key, discarded = jax.random.split(a._key)
    jax.random.split(discarded,100)  # old child splits cannot affect parent key
    np.testing.assert_allclose(a._values(jnp.array([[1.,2.]])), [[8.,9.]])
    np.testing.assert_array_equal(a._key, expected_key)
    with pytest.raises(AssertionError):
        a._build_rollout_fn()


def test_paired_configs_exact_and_historical_stop_kept():
    m = json.loads((BUNDLE/'manifest.json').read_text())
    assert len(m['rows']) == 12
    for a,b in zip(m['rows'][::2],m['rows'][1::2]):
        assert a['arm']=='cce' and b['arm']=='value'
        assert a['config']==b['config']
        if a['environment']=='frozenlake':
            assert a['config']['early_stop_win_rate']==.95
            assert a['config']['priority_mixing']=='multiplicative'
            assert a['config']['n_envs']==256 and a['config']['collect_steps']==128
        else:
            assert a['config']['collector_variant']=='fixed'
            assert a['config']['cf_drift_probe_n']==256
            assert a['config']['decay_start']==100000 and a['config']['decay_end']==150000
    for name,expected in m['source_hashes'].items():
        assert hashlib.sha256((ROOT/'work/value-score-replay'/name).read_bytes()).hexdigest()==expected


def test_frozenlake_refresh_selection_and_random_stream():
    class Buffer:
        def __len__(self):return 3
        def sample_uniform(self,n):
            indices=np.random.choice(3,n,replace=False)
            return [{'a':int(i%2)} for i in indices], indices
        def get_jax_state(self,i):return None if i==2 else int(i)
        def update_consequence_scores(self,indices,scores):self.result=(indices,scores)
    class Fake(FrozenLakeValue):
        buffer=Buffer()
        n_score_sample=3
        network=SimpleNamespace(apply=lambda p,x:jnp.array([x, x+p]))
        target_params=3.
        _key=jax.random.PRNGKey(42)
    a=Fake()
    np.random.seed(8); expected_indices=np.random.choice(3,3,replace=False)
    expected_rng=np.random.get_state()
    np.random.seed(8); a._score_buffer_transitions()
    np.testing.assert_array_equal(a.buffer.result[0],expected_indices[expected_indices!=2])
    np.testing.assert_allclose(a.buffer.result[1],3)
    np.testing.assert_array_equal(np.random.get_state()[1],expected_rng[1])
    assert np.random.get_state()[2:]==expected_rng[2:]
    np.testing.assert_array_equal(a._key,jax.random.split(jax.random.PRNGKey(42))[0])


def test_jaxnav_weights_use_online_but_gaps_use_target():
    class Fake(JaxNavValue):
        network=SimpleNamespace(apply=lambda p,x:p*x)
        target_params=2.
        params=1.
        _key=jax.random.PRNGKey(2)
        config={'cf_action_prob_temperature':.5}
    a=Fake()
    obs=np.array([[0.,1.,2.]])
    score,q=a._score_states(None,obs,[0])
    p=np.asarray(jax.nn.softmax(jnp.array(obs)/.5,axis=-1))
    np.testing.assert_allclose(score,action_gaps(obs*2,[0],p))
    np.testing.assert_allclose(q,obs)
