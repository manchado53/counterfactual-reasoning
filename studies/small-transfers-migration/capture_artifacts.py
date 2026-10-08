"""Inventory retained results; snapshot bounded text records without moving originals."""
from pathlib import Path
import hashlib
import json
from capture import DEST, SOURCES, add

records = []
copied = 0
cap = 8 * 1024**2
roots = [('andon-doorkey', SOURCES['andon']/'src/counterfactual_rl/agents/doorkey'),
         ('root-connect-four', SOURCES['root']/'src/counterfactual_rl/agents/connect_four'),
         ('fixed-doorkey-cpu-20261001',Path('/tmp/doorkey-transfer-cpu-cli-20261001'))]
for label,root in roots:
    files = []
    if label.startswith('fixed'):
        files = sorted(root.glob('*'))
    else:
        for sub in ('experiments','logs','runs'):
            files.extend(sorted((root/sub).rglob('*')))
    for path in files:
        if not path.is_file():
            continue
        size = path.stat().st_size
        record = dict(path=str(path),size=size,mtime_ns=path.stat().st_mtime_ns,preserved_in_place=True)
        # Never read multi-GB checkpoint trees. Bound total copied result bytes.
        if path.suffix in ('.json','.jsonl','.log','.out') and size <= 256*1024 and copied+size <= cap:
            content = path.read_bytes()
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                record['copy_status'] = 'not-utf8'
            else:
                rel = 'studies/small-transfers-migration/retained/'+label+'/'+str(path.relative_to(root))
                if not (DEST/rel).exists():
                    add(rel,text)
                else:
                    assert (DEST/rel).read_text() == '\n'.join(text.splitlines())+'\n'
                record.update(snapshot=rel,sha256=hashlib.sha256(content).hexdigest(),
                    snapshot_sha256=hashlib.sha256((DEST/rel).read_bytes()).hexdigest())
                copied += size
        else:
            record['copy_status'] = 'retained-in-place; binary, per-file or total cap'
        records.append(record)
add('studies/small-transfers-migration/artifact-inventory.json',json.dumps(dict(
    copied_bytes=copied,copy_cap_bytes=cap,files=records,
    scope='Named retained local roots only; no scheduler or remote run audit'),indent=2)+'\n')
print(len(records),'files inventoried;',copied,'text bytes copied')
