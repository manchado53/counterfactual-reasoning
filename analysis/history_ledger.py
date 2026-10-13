"""Build and audit the compact index of historical experiments and runs.

The large evidence stays in its original manifest or in the immutable migration
audit tag.  The committed JSONL index stores one small pointer per recovered run.
It never upgrades a recipe, chat statement, or Slurm completion into a result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


HISTORY_DIR = Path("experiments/history")
VALID_RUN_KINDS = {
    "observed_log",
    "submitted_manifest",
    "recovered_manifest",
    "scheduler_record",
}
VALID_EXECUTION = {"completed", "partial", "failed", "cancelled", "unknown"}
VALID_RESULT = {"included", "excluded", "partial", "not_evaluated", "unknown"}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_hash(config: dict) -> str:
    return _sha256_bytes(_canonical(config).encode())


def _read_source(root: Path, location: dict) -> tuple[bytes, str]:
    kind = location["kind"]
    if kind == "repo":
        path = root / location["path"]
        return path.read_bytes(), location["path"]
    if kind == "git":
        spec = f'{location["ref"]}:{location["path"]}'
        proc = subprocess.run(
            ["git", "show", spec], cwd=root, check=True, capture_output=True
        )
        return proc.stdout, spec
    raise ValueError(f"Unknown source location kind: {kind}")


def _arm(config: dict, fallback=None):
    if fallback is not None:
        return str(fallback)
    algorithm = config.get("algorithm")
    if algorithm == "dqn-uniform":
        return "uniform"
    if algorithm == "dqn":
        return "per"
    if algorithm != "consequence-dqn":
        return algorithm or "unknown"
    mixing = config.get("priority_mixing", "additive")
    if mixing == "multiplicative":
        return "cce_mul"
    if float(config.get("mu", 0.25)) >= 1.0:
        return "cce_only"
    return "cce_add"


def _base_record(source: dict, experiment_id: str, record_id: str, config: dict, **fields):
    record = {
        "run_id": f'{source["id"]}:{record_id}',
        "experiment_id": experiment_id,
        "source_id": source["id"],
        "source_record_id": str(record_id),
        "environment": source["environment"],
        "run_kind": source["run_kind"],
        "execution_state": source["execution_state"],
        "result_state": source["result_state"],
        "config_sha256": _config_hash(config),
        "seed": config.get("seed"),
        "arm": _arm(config),
    }
    record.update({k: v for k, v in fields.items() if v is not None})
    override = source.get("record_overrides", {}).get(str(record_id), {})
    record.update(override)
    return record


def _parse_json_map(source: dict, payload: dict):
    for record_id, config in payload.items():
        if not isinstance(config, dict):
            continue
        yield _base_record(
            source,
            source["experiment_id"],
            str(record_id),
            config,
            scheduler_job_id=str(record_id) if str(record_id).isdigit() else None,
        )


def _parse_recipe_rows(source: dict, payload: dict):
    for row in payload["rows"]:
        config = row["config"]
        record_id = row.get("original_job") or row.get("source_run") or row["task"]
        record = _base_record(
            source,
            source["experiment_id"],
            str(record_id),
            config,
            scheduler_job_id=str(record_id) if str(record_id).isdigit() else None,
            source_run=row.get("source_run"),
            source_log_sha256=row.get("source_header_sha256")
            or row.get("provenance", {}).get("log_sha256"),
        )
        record["seed"] = row.get("seed", record["seed"])
        record["arm"] = _arm(config, row.get("arm"))
        yield record


def _parse_jaxnav_ledger(source: dict, payload: dict):
    for entry in payload["entries"]:
        if entry["id"] in source.get("exclude_entries", []):
            continue
        experiment_id = source["experiment_prefix"] + entry["id"].replace("_", "-")
        for cell_index, cell in enumerate(entry.get("cells") or []):
            config = cell["settings"]
            for row in cell.get("rows", []):
                record_id = f'{entry["id"]}:{row.get("id", row.get("task"))}'
                merged = dict(config)
                if row.get("seed") is not None:
                    merged["seed"] = row["seed"]
                record = _base_record(
                    source,
                    experiment_id,
                    record_id,
                    merged,
                    source_task=row.get("task"),
                    collector=row.get("collector"),
                    mode=row.get("mode"),
                    gpu=row.get("gpu"),
                    cell=cell_index,
                )
                record["arm"] = _arm(merged, row.get("arm"))
                if entry.get("code_status") == "pinned_legacy":
                    record["run_kind"] = "observed_log"
                yield record


def _parse_manifest_catalog(source: dict, payload: dict, root: Path):
    """Collapse repeated archived manifest copies into one row per env/job."""
    unique = {}
    aliases = defaultdict(list)
    for manifest in payload["manifests"]:
        environment = manifest["environment"]
        experiment_id = source["experiment_ids"][environment]
        archive_path = manifest["archive"]
        wrapper, locator = _read_source(
            root,
            {"kind": "git", "ref": source["location"]["ref"], "path": archive_path},
        )
        data = base64.b64decode(json.loads(wrapper)["base64"])
        if _sha256_bytes(data) != manifest["sha256"]:
            raise ValueError(f"Archived manifest hash mismatch: {archive_path}")
        for job_id, config in json.loads(data).items():
            key = (environment, str(job_id))
            config_hash = _config_hash(config)
            if key in unique and unique[key][1] != config_hash:
                raise ValueError(f"Conflicting archived configs for {environment} job {job_id}")
            unique[key] = (config, config_hash, experiment_id)
            aliases[key].append(locator)
    for (environment, job_id), (config, _, experiment_id) in sorted(unique.items()):
        nested_source = dict(source, environment=environment)
        yield _base_record(
            nested_source,
            experiment_id,
            f"{environment}:{job_id}",
            config,
            scheduler_job_id=job_id,
            manifest_aliases=sorted(set(aliases[(environment, job_id)])),
        )


def _parse_manifest_list(source: dict, payload: dict, root: Path):
    for manifest in payload["manifests"]:
        data, locator = _read_source(
            root,
            {"kind": "git", "ref": payload["git_ref"], "path": manifest["path"]},
        )
        if _sha256_bytes(data) != manifest["sha256"]:
            raise ValueError(f'Manifest hash mismatch: {manifest["path"]}')
        nested_source = dict(source, environment=manifest["environment"])
        for key, config in json.loads(data).items():
            job_id = config.get("job_id")
            if job_id is None and str(key).isdigit():
                job_id = str(key)
            yield _base_record(
                nested_source,
                manifest["experiment_id"],
                f'{manifest["id"]}:{key}',
                config,
                scheduler_job_id=str(job_id) if job_id is not None else None,
                manifest_locator=locator,
            )


def _parse_scheduler_snapshot(source: dict, payload: dict, root: Path):
    groups_doc = json.loads((root / HISTORY_DIR / source["groups_path"]).read_text())
    jobs = {}
    for group in groups_doc["groups"]:
        for job_id in group.get("job_ids", []):
            jobs[str(job_id)] = group
        for first, last in group.get("job_ranges", []):
            for job_id in range(first, last + 1):
                jobs[str(job_id)] = group
    state_map = {
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "TIMEOUT": "failed",
        "RUNNING": "partial",
        "PENDING": "partial",
    }
    for row in payload["records"]:
        display_id = row["JobID"]
        group = jobs.get(display_id.split("_", 1)[0])
        if group is None:
            continue
        nested_source = dict(
            source,
            environment=group["environment"],
            execution_state=state_map.get(row["State"], "unknown"),
        )
        config = {
            "scheduler_job_id": display_id,
            "scheduler_state": row["State"],
            "scheduler_exit_code": row["ExitCode"],
        }
        record = _base_record(
            nested_source,
            group["experiment_id"],
            display_id,
            config,
            scheduler_job_id=display_id,
            scheduler_raw_job_id=row["JobIDRaw"],
            scheduler_state=row["State"],
            scheduler_exit_code=row["ExitCode"],
            elapsed=row["Elapsed"],
            started_at=row["Start"],
            ended_at=row["End"],
        )
        record["config_sha256"] = None
        yield record


PARSERS = {
    "json_map": _parse_json_map,
    "recipe_rows": _parse_recipe_rows,
    "jaxnav_ledger": _parse_jaxnav_ledger,
}


def _deduplicate_scheduler_records(records):
    """One physical scheduler job, with every weaker source retained as an alias."""
    strength = {
        "observed_log": 3,
        "submitted_manifest": 2,
        "recovered_manifest": 1,
        "scheduler_record": 0,
    }
    by_key = {}
    without_job = []
    for record in records:
        job_id = record.get("scheduler_job_id")
        if not job_id:
            without_job.append(record)
            continue
        key = (record["environment"], job_id)
        if key not in by_key:
            record["evidence_aliases"] = []
            by_key[key] = record
            continue
        old = by_key[key]
        if strength[record["run_kind"]] > strength[old["run_kind"]]:
            primary, alias = record, old
            primary["evidence_aliases"] = old.pop("evidence_aliases", [])
            by_key[key] = primary
        else:
            primary, alias = old, record
        primary.setdefault("evidence_aliases", []).append(
            {
                "source_id": alias["source_id"],
                "source_record_id": alias["source_record_id"],
                "run_kind": alias["run_kind"],
                "config_sha256": alias["config_sha256"],
            }
        )
        scheduler = record if record["run_kind"] == "scheduler_record" else (
            old if old["run_kind"] == "scheduler_record" else None
        )
        if scheduler is not None:
            primary["execution_state"] = scheduler["execution_state"]
            primary["scheduler_state"] = scheduler.get("scheduler_state")
            primary["scheduler_exit_code"] = scheduler.get("scheduler_exit_code")
        primary["evidence_aliases"].sort(key=lambda x: (x["source_id"], x["source_record_id"]))
    return without_job + list(by_key.values())


def load_definitions(root: Path):
    return json.loads((root / HISTORY_DIR / "experiments.json").read_text())


def load_sources(root: Path):
    return json.loads((root / HISTORY_DIR / "sources.json").read_text())


def build_records(root: Path):
    sources_doc = load_sources(root)
    records = []
    source_audit = []
    for source in sources_doc["sources"]:
        data, locator = _read_source(root, source["location"])
        actual_hash = _sha256_bytes(data)
        if actual_hash != source["sha256"]:
            raise ValueError(
                f'Source hash mismatch for {source["id"]}: '
                f'{actual_hash} != {source["sha256"]}'
            )
        payload = json.loads(data)
        if source["parser"] == "manifest_catalog":
            parsed = list(_parse_manifest_catalog(source, payload, root))
        elif source["parser"] == "manifest_list":
            parsed = list(_parse_manifest_list(source, payload, root))
        elif source["parser"] == "scheduler_snapshot":
            parsed = list(_parse_scheduler_snapshot(source, payload, root))
        else:
            parsed = list(PARSERS[source["parser"]](source, payload))
        if len(parsed) != source["expected_records"]:
            raise ValueError(
                f'Record count mismatch for {source["id"]}: '
                f'{len(parsed)} != {source["expected_records"]}'
            )
        records.extend(parsed)
        source_audit.append(
            {
                "source_id": source["id"],
                "locator": locator,
                "sha256": actual_hash,
                "records": len(parsed),
            }
        )
    records = _deduplicate_scheduler_records(records)
    records.sort(key=lambda x: x["run_id"])
    return records, source_audit


def load_committed_records(root: Path):
    path = root / HISTORY_DIR / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(root: Path, compare_generated=True):
    definitions = load_definitions(root)
    sources_doc = load_sources(root)
    experiment_ids = [item["id"] for item in definitions["experiments"]]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Duplicate experiment ids")
    source_ids = [item["id"] for item in sources_doc["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source ids")

    committed = load_committed_records(root)
    run_ids = [item["run_id"] for item in committed]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run ids")
    known_experiments = set(experiment_ids)
    known_sources = set(source_ids)
    for record in committed:
        if record["experiment_id"] not in known_experiments:
            raise ValueError(f'Unknown experiment: {record["experiment_id"]}')
        if record["source_id"] not in known_sources:
            raise ValueError(f'Unknown source: {record["source_id"]}')
        if record["run_kind"] not in VALID_RUN_KINDS:
            raise ValueError(f'Bad run kind: {record["run_kind"]}')
        if record["execution_state"] not in VALID_EXECUTION:
            raise ValueError(f'Bad execution state: {record["execution_state"]}')
        if record["result_state"] not in VALID_RESULT:
            raise ValueError(f'Bad result state: {record["result_state"]}')

    source_audit = []
    if compare_generated:
        generated, source_audit = build_records(root)
        if committed != generated:
            raise ValueError("runs.jsonl is stale; run history_ledger.py rebuild")

    by_env = Counter(r["environment"] for r in committed)
    by_execution = Counter(r["execution_state"] for r in committed)
    by_kind = Counter(r["run_kind"] for r in committed)
    scheduler_groups = defaultdict(list)
    for record in committed:
        if record.get("scheduler_job_id"):
            scheduler_groups[(record["experiment_id"], record["scheduler_job_id"])].append(
                record["run_id"]
            )
    duplicates = {f"{k[0]}:{k[1]}": v for k, v in scheduler_groups.items() if len(v) > 1}
    return {
        "schema_version": 1,
        "experiments": len(experiment_ids),
        "runs": len(committed),
        "sources": len(source_ids),
        "runs_by_environment": dict(sorted(by_env.items())),
        "runs_by_execution_state": dict(sorted(by_execution.items())),
        "runs_by_evidence_kind": dict(sorted(by_kind.items())),
        "duplicate_scheduler_keys": duplicates,
        "source_audit": source_audit,
        "warning": "An indexed run is not automatically a valid scientific result.",
    }


def rebuild(root: Path):
    records, _ = build_records(root)
    path = root / HISTORY_DIR / "runs.jsonl"
    path.write_text("".join(_canonical(record) + "\n" for record in records))
    return path, len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "rebuild", "summary"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "rebuild":
        path, count = rebuild(root)
        print(json.dumps({"path": str(path), "runs": count}, indent=2))
        return
    print(json.dumps(validate(root, compare_generated=args.command == "audit"), indent=2))


if __name__ == "__main__":
    main()
