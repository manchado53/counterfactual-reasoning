"""Versioned evaluation cases. Generation uses no training randomness."""
import hashlib
import json
from pathlib import Path
import numpy as np

BANK_PATH = Path(__file__).resolve().parents[3] / 'experiments/connect_four/jaxnav-transfer-v2/evaluation-bank.json'
POLICIES = ('random', 'center', 'win', 'win-block-center', 'win-block-edge')
TRAIN_BANK = dict(policy='win-block-center-v1', ties=[3, 2, 4, 1, 5, 0, 6], roles=[0, 1])


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def play(board, column, player):
    empty = np.flatnonzero(board[:, column] == 0)
    if not len(empty):
        raise ValueError('Illegal opening move')
    board[empty[-1], column] = player
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(6):
            for c in range(7):
                if 0 <= r + 3*dr < 6 and 0 <= c + 3*dc < 7:
                    if all(board[r+k*dr, c+k*dc] == player for k in range(4)):
                        return True
    return False


def generate():
    rng = np.random.default_rng(2026101402)
    cases, seen = [], set()
    for role in (0, 1):
        for policy in range(5):
            for index in range(100):
                length = (4 + 2*(index % 5)) if role == 0 else (5 + 2*(index % 4))
                while True:
                    board = np.zeros((6, 7), dtype=np.int8)
                    moves = []
                    for ply in range(length):
                        col = int(rng.choice(np.flatnonzero(board[0] == 0)))
                        moves.append(col)
                        if play(board, col, 1 + ply % 2):
                            break
                    else:
                        identity = tuple(board.ravel()) + (role,)
                        if identity not in seen:
                            seen.add(identity)
                            break
                cases.append(dict(id=len(cases), role=role, policy=policy,
                                  seed=int(rng.integers(1, 2**31)), opening=moves,
                                  board=board.tolist()))
    return dict(version=2, policies=list(POLICIES), cases=cases)


def verify(bank):
    if bank['version'] != 2 or bank['policies'] != list(POLICIES) or len(bank['cases']) != 1000:
        raise ValueError('Wrong evaluation bank schema')
    seen, counts = set(), np.zeros((2, 5), int)
    for i, case in enumerate(bank['cases']):
        role, policy = case['role'], case['policy']
        assert case['id'] == i and role in (0, 1) and policy in range(5)
        assert 4 <= len(case['opening']) <= 12 and len(case['opening']) % 2 == role
        board = np.zeros((6, 7), dtype=np.int8)
        for ply, col in enumerate(case['opening']):
            assert 0 <= col < 7
            assert not play(board, col, 1 + ply % 2), 'Terminal opening'
        np.testing.assert_array_equal(board, case['board'])
        key = tuple(board.ravel()) + (role,)
        assert key not in seen, 'Duplicate board'
        seen.add(key)
        counts[role, policy] += 1
    assert (counts == 100).all()
    return digest(bank)


def load(expected=None, path=BANK_PATH):
    raw = Path(path).read_bytes()
    bank = json.loads(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if raw != canonical(bank) or (expected is not None and actual != expected):
        raise ValueError('Evaluation bank hash mismatch')
    return bank, actual


if __name__ == '__main__':
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    bank = generate()
    print(verify(bank))
    BANK_PATH.write_bytes(canonical(bank))
