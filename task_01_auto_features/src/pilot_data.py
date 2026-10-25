"""Prepare a reproducible, complete-history 64-user Yambda pilot without model training."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .pilot_storage import ensure_run_writable
else:
    from pilot_storage import ensure_run_writable

TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TASK_ROOT / "data" / "raw" / "yambda"
EVENT_SOURCE = DATA_ROOT / "flat" / "50m" / "multi_event.parquet"
ARTIST_SOURCE = DATA_ROOT / "artist_item_mapping.parquet"
ALBUM_SOURCE = DATA_ROOT / "album_item_mapping.parquet"
MANIFEST = TASK_ROOT / "data" / "download_manifest.json"
BATCH_SIZE = 65536


def _manifest_hashes() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(Path(entry["local_path"]).as_posix()): entry.get("sha256", entry.get("expected_sha256"))
            for entry in manifest["files"] if entry.get("dataset") == "yambda"}


def _quantile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def _mapping(source: Path, output: Path, item_col: str, entity_col: str,
             selected_items: set[int], batch_size: int) -> dict:
    pf = pq.ParquetFile(source)
    item_type = pf.schema_arrow.field(item_col).type
    value_set = pa.array(sorted(selected_items), type=item_type)
    links: set[tuple[int, int]] = set()
    rows_scanned = rows_selected = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[item_col, entity_col], use_threads=False):
        rows_scanned += batch.num_rows
        mask = pc.is_in(batch.column(0), value_set=value_set)
        filtered = batch.filter(mask)
        rows_selected += filtered.num_rows
        if filtered.num_rows:
            links.update((int(a), int(e)) for a, e in zip(filtered.column(0).to_pylist(), filtered.column(1).to_pylist()))
    if links:
        out_schema = pa.schema([(item_col, item_type), (entity_col, pf.schema_arrow.field(entity_col).type)])
        writer = pq.ParquetWriter(output, out_schema, compression="zstd")
        ordered_links = sorted(links)
        chunk_size = 100_000
        for start in range(0, len(ordered_links), chunk_size):
            part = ordered_links[start:start + chunk_size]
            writer.write_table(pa.table({item_col: [x[0] for x in part], entity_col: [x[1] for x in part]}, schema=out_schema))
        writer.close()
    else:
        schema = pa.schema([pf.schema_arrow.field(item_col), pf.schema_arrow.field(entity_col)])
        pq.write_table(pa.Table.from_batches([], schema=schema), output, compression="zstd")
    item_entities: dict[int, set[int]] = defaultdict(set)
    for item, entity in links: item_entities[item].add(entity)
    mapped_items = set(item_entities)
    return {"source_file": source.name, "source_rows": pf.metadata.num_rows, "rows_scanned": rows_scanned,
            "selected_rows_before_deduplication": rows_selected, "unique_links": len(links),
            "duplicate_links_removed": rows_selected - len(links), "unique_mapped_items": len(mapped_items),
            "selected_items_covered": len(mapped_items & selected_items),
            "selected_item_coverage_pct": 100.0 * len(mapped_items & selected_items) / len(selected_items) if selected_items else None,
            "items_with_multiple_entities": sum(len(v) > 1 for v in item_entities.values()),
            "max_entities_per_item": max(map(len, item_entities.values()), default=0),
            "unique_entities": len({e for _, e in links}), "output_rows": pq.ParquetFile(output).metadata.num_rows}


def _prepare(output: Path, users: int, seed: int, event_source: Path, artist_source: Path,
             album_source: Path, source_hashes: dict[str, str], batch_size: int = BATCH_SIZE) -> dict:
    ensure_run_writable(output.parent)
    ensure_run_writable(output)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("events.parquet", "artist_mapping.parquet", "album_mapping.parquet", "metadata.json"):
        stale_path = output / stale
        if stale_path.exists(): stale_path.unlink()
    source = pq.ParquetFile(event_source)

    # Uniform sample from the observed UID population, independently of activity.
    population: set[int] = set()
    uid_rows_scanned = 0
    for batch in source.iter_batches(batch_size=batch_size, columns=["uid"], use_threads=False):
        uid_rows_scanned += batch.num_rows
        population.update(map(int, pc.unique(batch.column(0)).to_pylist()))
        if uid_rows_scanned % 1_000_000 < batch.num_rows:
            print(f"UID pass: scanned {uid_rows_scanned:,}/{source.metadata.num_rows:,}; unique users {len(population):,}", flush=True)
    if users < 1 or users > len(population):
        raise ValueError(f"users must be in [1, {len(population)}], got {users}")
    selected = sorted(random.Random(seed).sample(sorted(population), users))
    selected_set = set(selected)

    event_out = output / "events.parquet"
    writer = None
    rows = 0
    event_counts = Counter()
    missing = Counter()
    timestamps: list[int] = []
    selected_items: set[int] = set()
    history_rows: Counter[int] = Counter()
    chronology_violations = 0
    same_uid_timestamp_extra_rows = 0
    same_uid_timestamp_groups = 0
    previous: tuple[int, int] | None = None
    active_group_rows = 0
    previous_user: int | None = None
    selected_item_repeats = 0
    user_items: set[tuple[int, int]] = set()
    source_rows_scanned = 0

    for batch in source.iter_batches(batch_size=batch_size, use_threads=False):
        source_rows_scanned += batch.num_rows
        mask = pc.is_in(batch.column(batch.schema.get_field_index("uid")), value_set=pa.array(selected, type=batch.schema.field("uid").type))
        chosen = batch.filter(mask)
        if chosen.num_rows:
            if writer is None: writer = pq.ParquetWriter(event_out, chosen.schema, compression="zstd")
            writer.write_batch(chosen)
        cols = {name: chosen.column(name).to_pylist() for name in chosen.schema.names} if chosen.num_rows else {}
        for i, uid0 in enumerate(cols.get("uid", [])):
            uid, ts, item = int(uid0), int(cols["timestamp"][i]), int(cols["item_id"][i])
            if previous is not None:
                pu, pt = previous
                if uid < pu or (uid == pu and ts < pt): chronology_violations += 1
                if uid == pu and ts == pt:
                    same_uid_timestamp_extra_rows += 1
                    active_group_rows += 1
                else:
                    if active_group_rows > 1: same_uid_timestamp_groups += 1
                    active_group_rows = 1
            else:
                active_group_rows = 1
            previous = (uid, ts)
            if previous_user != uid:
                user_items = set()
                previous_user = uid
            if (uid, item) in user_items: selected_item_repeats += 1
            else: user_items.add((uid, item))
            rows += 1
            history_rows[uid] += 1
            timestamps.append(ts)
            selected_items.add(item)
            event_counts[str(cols["event_type"][i])] += 1
            for name, values in cols.items():
                if values[i] is None: missing[name] += 1
        if source_rows_scanned % 1_000_000 < batch.num_rows:
            print(f"event pass: scanned {source_rows_scanned:,}/{source.metadata.num_rows:,}; selected {rows:,}", flush=True)
    if writer is not None: writer.close()
    if active_group_rows > 1: same_uid_timestamp_groups += 1

    if not rows: raise ValueError("Selected users have no events")
    item_ids = pa.array(sorted(selected_items), type=source.schema_arrow.field("item_id").type)
    artist_meta = _mapping(artist_source, output / "artist_mapping.parquet", "item_id", "artist_id", set(map(int, item_ids.to_pylist())), batch_size)
    album_meta = _mapping(album_source, output / "album_mapping.parquet", "item_id", "album_id", set(map(int, item_ids.to_pylist())), batch_size)
    for name, meta, path in (("artist", artist_meta, output / "artist_mapping.parquet"), ("album", album_meta, output / "album_mapping.parquet")):
        saved_rows = pq.ParquetFile(path).metadata.num_rows
        if saved_rows != meta["unique_links"]:
            raise RuntimeError(f"{name} mapping output has {saved_rows} rows but metadata counts {meta['unique_links']} unique links")

    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": seed, "requested_users": users,
        "selected_users": selected, "source": {"events": str(event_source), "source_file_hash_sha256": source_hashes.get("yambda/flat/50m/multi_event.parquet"),
                    "published_50m_rows": source.metadata.num_rows,
                    "completeness_note": "This pilot retains all published 50M events for selected UIDs; it does not establish completeness relative to Yambda-5B."},
        "rows": rows, "unique_users": len(history_rows), "unique_items": len(selected_items),
        "events_per_user": {"min": min(history_rows.values()), "median": _quantile(list(history_rows.values()), .5), "max": max(history_rows.values())},
        "event_type_counts": dict(event_counts), "timestamp_raw": {"min": min(timestamps), "max": max(timestamps), "units": "raw source integer; no calendar conversion"},
        "chronology": {"violations_in_selected_rows": chronology_violations, "same_uid_timestamp_extra_rows": same_uid_timestamp_extra_rows,
                       "same_uid_timestamp_groups": same_uid_timestamp_groups, "sort_key": ["uid", "timestamp"], "checked_across_batch_boundaries": True},
        "selected_user_item_repeated_events": selected_item_repeats,
        "missing_value_counts": dict(missing),
        "proposed_global_time_boundaries": {"method": "event-weighted empirical timestamp quantiles over selected events; identical raw timestamps remain indivisible",
            "train_query_start_p30_raw": _quantile(timestamps, .30), "train_end_p70_raw": _quantile(timestamps, .70),
            "validation_end_p85_raw": _quantile(timestamps, .85),
            "intervals": "train_query_start_p30 <= timestamp < train_end_p70; train_end_p70 <= timestamp < validation_end_p85 for validation; timestamp >= validation_end_p85 for test. Equal timestamps are indivisible. Proposed offline semantics: logged event known at timestamp."},
        "mappings": {"artist": artist_meta, "album": album_meta},
        "resources": {"duration_seconds": round(time.perf_counter() - started, 3)},
        "versions": {"python": sys.version, "pyarrow": pa.__version__},
        "commands": ["python src/pilot_data.py --users 64 --seed 20261025 --output artifacts/pilot_001/data", "python -m unittest tests.test_pilot_data -v"],
        "source_scan_rows": source_rows_scanned, "source_rows_for_selection": source.metadata.num_rows,
        "outputs": {"events": "events.parquet", "artists": "artist_mapping.parquet", "albums": "album_mapping.parquet"}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def prepare_data(output: Path, users: int = 64, seed: int = 20261025) -> dict:
    output = Path(output)
    return _prepare(output, users, seed, EVENT_SOURCE, ARTIST_SOURCE, ALBUM_SOURCE, _manifest_hashes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot_001/data"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else TASK_ROOT / args.output
    result = prepare_data(output, args.users, args.seed)
    print(f"Pilot saved: {result['rows']:,} events, {result['unique_users']} users, {result['unique_items']:,} items")


if __name__ == "__main__": main()
