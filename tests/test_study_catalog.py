from pathlib import Path
from training.catalog import catalog


def test_all_discovered_runnable_recipes_validate():
    root = Path(__file__).resolve().parents[1]
    result = catalog(root)
    assert not list((root / 'experiments').rglob('*.py'))
    assert {'jaxnav','frozenlake','doorkey','connect_four','smax','chess','cvrp',
            'rpdqn','cvrp_budget'} <= {
        row['environment'] for row in result['training']}
    assert result['analyses']
    assert result['history']['runs'] >= 5000
    assert result['history']['experiments'] >= 40
    for row in result['training'] + result['analyses']:
        assert not row['invalid'], row
        assert (root / row['path']).with_name('README.md').is_file(), row
