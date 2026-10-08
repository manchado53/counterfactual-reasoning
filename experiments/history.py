"""Lossless historical manifests. Never fill old settings from today's defaults.

Import retains the original bytes and row metadata. Export returns those bytes,
not a runnable recipe. No training modules or simulator dependencies are loaded.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_metrics_config(path):
    """Read the explicit config header; never import a trainer or load a pickle."""
    config = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("#"):
            break
        match = re.fullmatch(r"# ([a-z_][a-z_0-9]*|PER_parameters): (.*)", line)
        if not match:
            continue
        key, raw = match.groups()
        if key in config:
            raise ValueError(f"Duplicate config field: {key}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Historical string enums were emitted without quotes.
            if not re.fullmatch(r"[A-Za-z_0-9][A-Za-z_0-9.-]*", raw):
                raise ValueError(f"Unsupported historical value for {key}") from None
            value = raw
        config[key] = value
    if not config:
        raise ValueError("No explicit config header; refusing to infer defaults")
    json.dumps(config, allow_nan=False)
    return config


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_number(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_number)


def _rows(raw, encoding):
    if encoding == "base64-lines":
        values = [_json(base64.b64decode(line, validate=True))
                  for line in raw.splitlines() if line.strip()]
        pairs = [(str(i), value) for i, value in enumerate(values)]
    elif encoding == "json":
        value = _json(raw)
        if isinstance(value, dict) and "rows" in value:
            values = value["rows"]
            if not isinstance(values, list):
                raise ValueError("rows must be a list")
            pairs = [(str(row.get("task", i)), row) for i, row in enumerate(values)]
        elif isinstance(value, dict):
            pairs = list(value.items())
        elif isinstance(value, list):
            pairs = [(str(i), row) for i, row in enumerate(value)]
        else:
            raise ValueError("Expected a manifest object or list")
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    result = []
    seen = set()
    for key, row in pairs:
        if key in seen:
            raise ValueError(f"Duplicate row identity: {key}")
        seen.add(key)
        if not isinstance(row, dict) or not isinstance(row.get("config", row), dict):
            raise ValueError(f"Row {key} must contain a config object")
        result.append({"id": key, "raw": row})
    if not result:
        raise ValueError("Empty manifest")
    return result


def import_manifest(path, *, encoding="json", provenance=None):
    """Capture exact bytes and parsed rows without executing source or defaults."""
    path = Path(path)
    raw = path.read_bytes()
    return {"schema_version": 1, "kind": "historical-raw-manifest",
            "source": {"path": str(path.resolve()), "sha256": sha256(raw),
                       "encoding": encoding, "bytes_base64": base64.b64encode(raw).decode("ascii"),
                       "provenance": provenance or {}},
            "rows": _rows(raw, encoding)}


def validate_manifest(record):
    """Check both the retained bytes and their parsed view; return original bytes."""
    if record.get("schema_version") != 1 or record.get("kind") != "historical-raw-manifest":
        raise ValueError("Unsupported historical manifest schema")
    source = record["source"]
    raw = base64.b64decode(source["bytes_base64"], validate=True)
    if sha256(raw) != source["sha256"]:
        raise ValueError("Source hash mismatch")
    expected = json.dumps(_rows(raw, source["encoding"]), sort_keys=True, allow_nan=False)
    actual = json.dumps(record["rows"], sort_keys=True, allow_nan=False)
    if expected != actual:
        raise ValueError("Parsed rows differ from preserved source")
    return raw


def export_manifest(record, destination):
    """Restore original bytes to a NEW file; never overwrite an existing file."""
    raw = validate_manifest(record)
    with Path(destination).open("xb") as stream:
        stream.write(raw)


def _bundle_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe bundle member: {name}")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError(f"Symlink in bundle member: {name}")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaped bundle member: {name}")
    return path


def export_legacy_bundle(entry, record, destination, *, max_bytes=64 * 1024**2):
    """Export pinned bytes plus recovery gates. Never execute or change a trainer.

    Caller must supply the trusted ledger entry and its archived raw manifest.
    All bytes are checked before creating the archive. The cap bounds memory.
    Extra smoke and lock records are inventoried separately, not called sealed.
    """
    if entry["id"] not in {"temporal_replay_v2", "episode_carry_pilot"}:
        raise ValueError("Only audited hold/carry recovery is supported")
    if entry["code_status"] != "pinned_legacy":
        raise ValueError("Requires a pinned legacy entry")
    bundle = entry["source_bundle"]
    root = Path(bundle["root"])
    hash_bytes = _bundle_file(root, "hashes.json").read_bytes()
    if sha256(hash_bytes) != bundle["hash_manifest"]["sha256"]:
        raise ValueError("Sealed hash manifest drift")
    expected = _json(hash_bytes)
    if not isinstance(expected, dict) or not {"manifest.json", "versions.json"} <= set(expected):
        raise ValueError("Incomplete sealed hash manifest")
    manifest_bytes = validate_manifest(record)
    if expected["manifest.json"] != sha256(manifest_bytes):
        raise ValueError("Recipe does not match sealed manifest")
    names = set(expected) | {"hashes.json"}
    extras = [p.name for p in root.glob("smoke-*.json") if re.fullmatch(r"smoke-\d+\.json", p.name)]
    extras += [name for name in ("pip-freeze.txt", "conda-explicit.txt") if (root / name).is_file()]
    names.update(extras)
    paths = {name: _bundle_file(root, name) for name in names}
    if sum(path.stat().st_size for path in paths.values()) > max_bytes:
        raise ValueError("Bundle exceeds export byte cap")
    contents = {}
    total = 0
    for name, path in sorted(paths.items()):
        data = path.read_bytes()
        total += len(data)
        if total > max_bytes:
            raise ValueError("Bundle exceeds export byte cap")
        if name in expected and sha256(data) != expected[name]:
            raise ValueError(f"Sealed source drift: {name}")
        contents[name] = data
    if contents["hashes.json"] != hash_bytes or contents["manifest.json"] != manifest_bytes:
        raise ValueError("Bundle changed during export")
    # These are required by the carry trainer's own train-mode gate.
    if entry["id"] == "episode_carry_pilot":
        rows = _json(manifest_bytes)["rows"]
        reports = [_json(contents[f"smoke-{r['task']}.json"]) for r in rows if r["seed"] == 0]
        if len(reports) != 10 or not all(r["passed"] and r["env_steps"] == 65536 for r in reports):
            raise ValueError("Carry smoke gate is incomplete")
        if len({r["eval_bank_hash"] for r in reports}) != 1:
            raise ValueError("Carry smoke banks differ")
    report = {"schema_version": 1, "id": entry["id"], "sealed_files": len(expected),
              "source_bytes": total, "member_sha256": {name: sha256(data) for name, data in contents.items()},
              "unsealed_recovery_records": sorted(set(extras) - set(expected)),
              "scope": "Exact legacy bytes; no trainer rewrite, package install, or runtime smoke."}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("recovery-info.json", json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("import")
    capture.add_argument("source", type=Path)
    capture.add_argument("destination", type=Path)
    capture.add_argument("--encoding", choices=("json", "base64-lines"), default="json")
    for name in ("export", "verify"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        if name == "export":
            command.add_argument("destination", type=Path)
    bundle = commands.add_parser("export-bundle")
    bundle.add_argument("ledger", type=Path)
    bundle.add_argument("entry_id", choices=("temporal_replay_v2", "episode_carry_pilot"))
    bundle.add_argument("destination", type=Path)
    bundle.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export-bundle":
        ledger = _json(args.ledger.read_bytes())
        entries = [e for e in ledger["entries"] if e["id"] == args.entry_id]
        if len(entries) != 1:
            raise ValueError("No unique legacy entry")
        record_path = _bundle_file(args.repo, entries[0]["recipe"])
        report = export_legacy_bundle(entries[0], _json(record_path.read_bytes()), args.destination)
        print(json.dumps(report, indent=2))
    elif args.command == "import":
        record = import_manifest(args.source, encoding=args.encoding)
        with args.destination.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        record = _json(args.source.read_bytes())
        if args.command == "export":
            export_manifest(record, args.destination)
        else:
            validate_manifest(record)


if __name__ == "__main__":
    main()
