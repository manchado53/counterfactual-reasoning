"""Separate, predeclared raw-score tolerance; all other trace fields stay exact.

Approved after old/old T4 measurement, before any numeric old/new comparison.
Never relabel the original failed bit-exact gate as passed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.parity.run import first_difference

SCORE_ATOL = 1e-7
CONTRACT = dict(name='score-numeric-v1', raw_score_atol=SCORE_ATOL, raw_score_rtol=0,
                ranks='exact', all_other_fields='exact')


def compare_traces(left, right, left_dir, right_dir, mode='exact'):
    result = dict(difference=None, max_score_abs_difference=0., score_snapshots=0)
    if mode == 'exact':
        result['difference'] = first_difference(left, right)
        return result
    if mode != CONTRACT['name']:
        raise ValueError('Unknown parity comparison contract')
    a, b = deepcopy(left), deepcopy(right)
    try:
        aa = [e for e in a['events'] if e['kind'] == 'insert']
        bb = [e for e in b['events'] if e['kind'] == 'insert']
        if not aa or len(aa) != len(bb):
            raise ValueError('Missing or unequal insert events')
        for index, (x,y) in enumerate(zip(aa,bb), 1):
            arrays = []
            for event, directory in ((x,left_dir),(y,right_dir)):
                path = Path(directory)/f'scores-chunk-{index}.npy'
                values = np.load(path, allow_pickle=False)
                meta = event['scores']['leaves']
                if len(meta) != 1 or list(values.shape) != meta[0]['shape'] or values.dtype.str != meta[0]['dtype']:
                    raise ValueError('Score snapshot shape/dtype mismatch')
                if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() != meta[0]['sha256']:
                    raise ValueError('Score snapshot hash mismatch')
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite scores')
                arrays.append(values)
            if arrays[0].shape != arrays[1].shape:
                raise ValueError('Score snapshot shape mismatch')
            delta = float(np.max(np.abs(arrays[0]-arrays[1]), initial=0))
            result['max_score_abs_difference'] = max(result['max_score_abs_difference'], delta)
            result['score_snapshots'] += 1
            if delta > SCORE_ATOL:
                raise ValueError('Raw-score difference exceeds fixed limit')
            if not np.array_equal(rankdata(arrays[0], method='average'), rankdata(arrays[1], method='average')):
                raise ValueError('Raw-score rank/tie order differs')
            # Mask only this checked hash, not shape, dtype, paths, or other fields.
            x['scores']['leaves'][0]['sha256'] = 'checked-score-numeric-v1'
            y['scores']['leaves'][0]['sha256'] = 'checked-score-numeric-v1'
        result['difference'] = first_difference(a, b)
    except (ValueError, OSError, KeyError) as exc:
        result['difference'] = str(exc)
    return result
