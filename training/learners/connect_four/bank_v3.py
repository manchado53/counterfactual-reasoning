"""Frozen 500 empty + 500 paired-opening cases, with portable board identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .bank_v2 import canonical, digest, play
from .minimax_eval import SPEC

BANK_PATH = Path(__file__).resolve().parents[3]/'experiments/connect_four/fixed-eval-v3/evaluation-bank.json'
SUITES = ('empty', 'opening')


def generate():
    rng = np.random.default_rng(2026101403)
    openings, seen = [], set()
    while len(openings) < 50:
        board = np.zeros((6,7), np.int8)
        moves = []
        for ply in range(4):
            col = int(rng.choice(np.flatnonzero(board[0] == 0)))
            moves.append(col)
            assert not play(board, col, 1+ply%2)
        identity = tuple(board.ravel())
        if identity not in seen:
            seen.add(identity)
            openings.append((moves, board.tolist()))
    cases = []
    for suite in SUITES:
        for depth in range(1,6):
            for role in (0,1):
                for index in range(50):
                    moves, board = openings[index] if suite == 'opening' else ([], np.zeros((6,7),int).tolist())
                    cases.append(dict(id=len(cases), suite=suite, depth=depth, role=role,
                        opening_id=index if suite == 'opening' else None,
                        opening=moves, board=board, seed=int(rng.integers(1,2**31))))
    return dict(version=3, opponent=SPEC, cases=cases)


def verify(bank):
    if bank.get('version') != 3 or bank.get('opponent') != SPEC or len(bank['cases']) != 1000:
        raise ValueError('Wrong v3 bank schema')
    counts = np.zeros((2,5,2),int)
    openings, seeds, case_ids = {}, set(), set()
    for i, c in enumerate(bank['cases']):
        assert c['id'] == i and c['role'] in (0,1) and c['depth'] in range(1,6)
        suite = SUITES.index(c['suite'])
        assert len(c['opening']) == (4 if suite else 0)
        assert 0 < c['seed'] < 2**31 and c['seed'] not in seeds
        seeds.add(c['seed'])
        board = np.zeros((6,7),np.int8)
        for ply, col in enumerate(c['opening']):
            assert type(col) is int and 0 <= col < 7
            assert not play(board,col,1+ply%2)
        np.testing.assert_array_equal(board,c['board'])
        if suite:
            index = c['opening_id']
            assert index in range(50)
            identity = (tuple(c['opening']), tuple(board.ravel()))
            assert openings.setdefault(index,identity) == identity
        else:
            assert c['opening_id'] is None
        identity = (suite,c['depth'],c['role'],c['opening_id'],c['seed'])
        assert identity not in case_ids
        case_ids.add(identity)
        counts[suite,c['depth']-1,c['role']] += 1
    assert (counts == 50).all()
    assert len(openings) == len({b for _,b in openings.values()}) == 50
    # Each opening must appear once in every role/depth cell.
    for depth in range(1,6):
        for role in (0,1):
            assert sorted(c['opening_id'] for c in bank['cases'] if c['suite']=='opening'
                          and c['depth']==depth and c['role']==role) == list(range(50))
    return digest(bank)


def load(expected=None, path=BANK_PATH):
    raw = Path(path).read_bytes()
    bank = json.loads(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if raw != canonical(bank) or (expected is not None and actual != expected):
        raise ValueError('Evaluation bank hash mismatch')
    verify(bank)
    return bank, actual


if __name__ == '__main__':
    bank = generate()
    print(verify(bank))
    BANK_PATH.write_bytes(canonical(bank))
