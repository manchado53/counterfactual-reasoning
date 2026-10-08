import hashlib
import numpy as np
from experiments.parity.numeric import compare_traces


def trace(path, scores, *, weight=.5):
    path.mkdir()
    a = np.asarray(scores, dtype=np.float64)
    np.save(path/'scores-chunk-1.npy', a)
    return dict(events=[dict(kind='insert', scores=dict(paths=['()'], leaves=[dict(
        shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest())])),
        dict(kind='sample', weights=[weight])])


def test_allow_only_small_raw_score_drift_with_identical_ranks(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    x, y = trace(a, [.1,.2]), trace(b, [.10000001,.2])
    result = compare_traces(x, y, a, b, 'score-numeric-v1')
    assert result['difference'] is None
    assert result['max_score_abs_difference'] > 0
    assert compare_traces(x,y,a,b,'exact')['difference'] is not None


def test_rank_change_fails_even_inside_numeric_tolerance(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    x,y = trace(a,[.1,.1]), trace(b,[.1,.10000001])
    assert 'rank' in compare_traces(x,y,a,b,'score-numeric-v1')['difference']


def test_other_training_fields_remain_exact(tmp_path):
    a,b = tmp_path/'a', tmp_path/'b'
    x,y = trace(a,[.1,.2]), trace(b,[.1,.2],weight=.50000000001)
    assert 'weights' in compare_traces(x,y,a,b,'score-numeric-v1')['difference']


def test_score_drift_outside_fixed_limit_fails(tmp_path):
    a,b = tmp_path/'a', tmp_path/'b'
    x,y = trace(a,[.1,.2]), trace(b,[.10001,.2])
    assert 'limit' in compare_traces(x,y,a,b,'score-numeric-v1')['difference']


def test_tampered_score_snapshot_fails(tmp_path):
    a,b = tmp_path/'a', tmp_path/'b'
    x,y = trace(a,[.1,.2]), trace(b,[.1,.2])
    np.save(b/'scores-chunk-1.npy', np.array([.1,.3]))
    assert 'hash' in compare_traces(x,y,a,b,'score-numeric-v1')['difference']
