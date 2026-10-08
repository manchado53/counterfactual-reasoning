from experiments.discovery import scan_tree


def test_discovery_includes_ignored_work_and_declares_caps(tmp_path):
    (tmp_path / 'work').mkdir()
    (tmp_path / 'work' / 'uncommitted.py').write_text('original = 1\n')
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs' / 'huge.json').write_text('not source')
    found = scan_tree(tmp_path)
    assert [e['path'] for e in found['entries']] == ['work/uncommitted.py']
    assert scan_tree(tmp_path, max_files=0)['truncated'] == 'file-count cap'
    limited = scan_tree(tmp_path, max_bytes=1)
    assert not limited['entries'] and limited['skipped'][0]['reason'] == 'byte cap'
