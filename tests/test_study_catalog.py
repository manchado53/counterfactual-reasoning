from pathlib import Path
from experiments.catalog import catalog


def test_all_discovered_runnable_recipes_validate():
    result = catalog(Path(__file__).resolve().parents[1])
    assert {'jaxnav','frozenlake','doorkey','connect_four','smax','chess','cvrp'} <= {
        row['environment'] for row in result['training']}
    assert result['analyses']
    for row in result['training'] + result['analyses']:
        assert not row['invalid'], row
