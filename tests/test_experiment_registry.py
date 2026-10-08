"""The recipe/run interface remains lazy and preserves each study's clock."""
import subprocess
import sys


def test_registry_validation_needs_no_simulator():
    code = '''
import sys
from experiments.registry import get_environment
from experiments.recipes import load_recipe
row = load_recipe('configs/experiment/jaxnav_decay_20261007.json', 1)
assert get_environment('jaxnav').evaluation_axis(row['config']) == 'episode'
row = load_recipe('configs/experiment/frozenlake-stochastic-transfer.json', 0)
assert get_environment('frozenlake').evaluation_axis(row['config']) == 'env_steps'
assert not {'jax', 'jaxmarl', 'pgx', 'torch'} & set(sys.modules)
try:
    get_environment('typo')
except ValueError:
    pass
else:
    raise AssertionError('Unknown environments must fail closed')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
