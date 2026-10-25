"""Prepare a 256-user scale-data snapshot without materializing reserved labels in Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .pilot_data import _mapping, _manifest_hashes
else:
    from pilot_data import _mapping, _manifest_hashes

TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TASK_ROOT / "data" / "raw" / "yambda"
EVENT_SOURCE = DATA_ROOT / "flat" / "50m" / "multi_event.parquet"
ARTIST_SOURCE = DATA_ROOT / "artist_item_mapping.parquet"
ALBUM_SOURCE = DATA_ROOT / "album_item_mapping.parquet"
BATCH_SIZE = 65536
STATS_CHUNK_SIZE = 8192
DEV_COLUMNS = ["uid", "item_id", "timestamp", "event_type"]


def _check_monitor(monitor) -> None:
    if monitor is not None:
        monitor.check()


def _selection_hash(selected: list[int]) -> str:
    payload = json.dumps(selected, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mapping_with_monitor(source: Path, output: Path, item_col: str, entity_col: str,
                          selected_items: set[int], batch_size: int, monitor) -> dict:
    """Bounded, batch-monitored variant of the shared pilot mapping filter."""
    pf = pq.ParquetFile(source)
    item_type = pf.schema_arrow.field(item_col).type
    entity_type = pf.schema_arrow.field(entity_col).type
    value_set = pa.array(sorted(selected_items), type=item_type)
    links: set[tuple[int, int]] = set()
    scanned = kept = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[item_col, entity_col], use_threads=False):
        _check_monitor(monitor)
        scanned += batch.num_rows
        filtered = batch.filter(pc.is_in(batch.column(0), value_set=value_set))
        kept += filtered.num_rows
        links.update((int(item), int(entity)) for item, entity in
                     zip(filtered.column(0).to_pylist(), filtered.column(1).to_pylist()))
    schema = pa.schema([pf.schema_arrow.field(item_col), pf.schema_arrow.field(entity_col)])
    if links:
        writer = pq.ParquetWriter(output, schema, compression="zstd")
        try:
            ordered = sorted(links)
            for start in range(0, len(ordered), 100_000):
                _check_monitor(monitor)
                chunk = ordered[start:start + 100_000]
                writer.write_table(pa.table({item_col: [x[0] for x in chunk], entity_col: [x[1] for x in chunk]}, schema=schema))
        finally:
            writer.close()
    else:
        pq.write_table(pa.Table.from_batches([], schema=schema), output, compression="zstd")
    item_entities: dict[int, set[int]] = {}
    entity_ids = set()
    for item, entity in links:
        item_entities.setdefault(item, set()).add(entity)
        entity_ids.add(entity)
    mapped_items = set(item_entities)
    return {"source_file": source.name, "source_rows": pf.metadata.num_rows, "rows_scanned": scanned,
            "selected_rows_before_deduplication": kept, "unique_links": len(links),
            "duplicate_links_removed": kept - len(links), "unique_mapped_items": len(mapped_items),
            "selected_items_covered": len(mapped_items & selected_items),
            "selected_item_coverage_pct": 100.0 * len(mapped_items & selected_items) / len(selected_items) if selected_items else None,
            "items_with_multiple_entities": sum(len(v) > 1 for v in item_entities.values()),
            "max_entities_per_item": max(map(len, item_entities.values()), default=0),
            "unique_entities": len(entity_ids), "output_rows": pq.ParquetFile(output).metadata.num_rows}


def _write_failed(output: Path, error: BaseException) -> None:
    failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
               "note": "Partial outputs, if present, are not valid run results; output path is intentionally non-reusable."}
    try:
        (output / "FAILED.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


def _prepare_inner(output: Path, users: int, seed: int, cutoff: int, event_source: Path,
                   artist_source: Path, album_source: Path, source_hashes: dict[str, str],
                   batch_size: int = BATCH_SIZE, monitor=None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Scale output must be a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = pq.ParquetFile(event_source)

    # Read only UID values while constructing the population; activity counts do not affect selection.
    population: set[int] = set()
    uid_rows_scanned = 0
    for batch in source.iter_batches(batch_size=batch_size, columns=["uid"], use_threads=False):
        _check_monitor(monitor)
        uid_rows_scanned += batch.num_rows
        population.update(map(int, pc.unique(batch.column(0)).to_pylist()))
        if uid_rows_scanned % 1_000_000 < batch.num_rows:
            print(f"UID population pass: {uid_rows_scanned:,}/{source.metadata.num_rows:,}; users={len(population):,}", flush=True)
    if users < 1 or users > len(population):
        raise ValueError(f"users must be in [1, {len(population)}], got {users}")
    selected = sorted(random.Random(seed).sample(sorted(population), users))
    uid_type = source.schema_arrow.field("uid").type
    selected_values = pa.array(selected, type=uid_type)

    full_path = output / "events_full.parquet"
    dev_path = output / "events.parquet"
    full_writer = dev_writer = None
    full_rows = dev_rows = source_rows_scanned = 0
    dev_items: set[int] = set()
    event_counts: Counter[str] = Counter()
    user_counts: Counter[int] = Counter()
    null_counts: Counter[str] = Counter()
    chronology_violations = 0
    timestamp_tie_extra_rows = 0
    timestamp_tie_groups = 0
    previous: tuple[int, int] | None = None
    active_tie_rows = 0

    event_error = None
    try:
        for batch in source.iter_batches(batch_size=batch_size, use_threads=False):
            _check_monitor(monitor)
            source_rows_scanned += batch.num_rows
            if source_rows_scanned % 1_000_000 < batch.num_rows:
                print(f"selected-event pass: {source_rows_scanned:,}/{source.metadata.num_rows:,}; full={full_rows:,}; dev={dev_rows:,}", flush=True)
            uid_index = batch.schema.get_field_index("uid")
            uid_mask = pc.is_in(batch.column(uid_index), value_set=selected_values)
            selected_full = batch.filter(uid_mask)
            if selected_full.num_rows:
                if full_writer is None:
                    full_writer = pq.ParquetWriter(full_path, selected_full.schema, compression="zstd")
                full_writer.write_batch(selected_full)
                full_rows += selected_full.num_rows

            # Filter reserved/post-cutoff rows in Arrow before any value conversion/statistics.
            timestamp_index = selected_full.schema.get_field_index("timestamp")
            if not selected_full.num_rows:
                continue
            before_cutoff = pc.less(selected_full.column(timestamp_index), pa.scalar(cutoff, type=selected_full.schema.field("timestamp").type))
            selected_dev = selected_full.filter(before_cutoff)
            if not selected_dev.num_rows:
                continue
            dev_batch = selected_dev.select(DEV_COLUMNS)
            if dev_writer is None:
                dev_writer = pq.ParquetWriter(dev_path, dev_batch.schema, compression="zstd")
            dev_writer.write_batch(dev_batch)
            dev_rows += dev_batch.num_rows

            for start in range(0, dev_batch.num_rows, STATS_CHUNK_SIZE):
                _check_monitor(monitor)
                piece = dev_batch.slice(start, STATS_CHUNK_SIZE)
                vals = {name: piece.column(name).to_pylist() for name in DEV_COLUMNS}
                for i, uid_raw in enumerate(vals["uid"]):
                    uid, timestamp, item = int(uid_raw), int(vals["timestamp"][i]), int(vals["item_id"][i])
                    if previous is not None:
                        old_uid, old_timestamp = previous
                        if uid < old_uid or (uid == old_uid and timestamp < old_timestamp):
                            chronology_violations += 1
                        if uid == old_uid and timestamp == old_timestamp:
                            timestamp_tie_extra_rows += 1
                            active_tie_rows += 1
                        else:
                            if active_tie_rows > 1:
                                timestamp_tie_groups += 1
                            active_tie_rows = 1
                    else:
                        active_tie_rows = 1
                    previous = (uid, timestamp)
                    user_counts[uid] += 1
                    dev_items.add(item)
                    event_counts[str(vals["event_type"][i])] += 1
                for name in DEV_COLUMNS:
                    null_counts[name] += sum(value is None for value in vals[name])
    except BaseException as exc:
        event_error = exc
    finally:
        for writer in (full_writer, dev_writer):
            if writer is not None:
                try:
                    writer.close()
                except BaseException as close_error:
                    if event_error is None:
                        event_error = close_error
    if event_error is not None:
        raise event_error
    if active_tie_rows > 1: timestamp_tie_groups += 1
    if full_rows == 0: raise ValueError("Selected users have no source events")
    if dev_rows == 0:
        schema = pa.schema([source.schema_arrow.field(name) for name in DEV_COLUMNS])
        pq.write_table(pa.Table.from_batches([], schema=schema), dev_path, compression="zstd")

    _check_monitor(monitor)
    artist_meta = (_mapping_with_monitor if monitor is not None else _mapping)(
        artist_source, output / "artist_mapping.parquet", "item_id", "artist_id", dev_items, batch_size,
        **({"monitor": monitor} if monitor is not None else {}))
    _check_monitor(monitor)
    album_meta = (_mapping_with_monitor if monitor is not None else _mapping)(
        album_source, output / "album_mapping.parquet", "item_id", "album_id", dev_items, batch_size,
        **({"monitor": monitor} if monitor is not None else {}))
    _check_monitor(monitor)
    for name, meta, path in (("artist", artist_meta, output / "artist_mapping.parquet"),
                             ("album", album_meta, output / "album_mapping.parquet")):
        if pq.ParquetFile(path).metadata.num_rows != meta["unique_links"]:
            raise RuntimeError(f"{name} mapping output row count differs from unique link count")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "users_requested": users, "seed": seed,
        "selected_users": selected, "selected_uid_count": len(selected), "selected_uid_list_sha256": _selection_hash(selected),
        "selection_population": {"method": "uniform random.sample over sorted unique UID population", "uids_scanned": len(population),
                                "uid_rows_scanned": uid_rows_scanned, "source_event_rows": source.metadata.num_rows},
        "source": {"events_path": str(event_source), "events_sha256_from_manifest": source_hashes.get("yambda/flat/50m/multi_event.parquet"),
                   "dataset_revision": "dd6f3a19eef5866e346c3270e098baa641a44948",
                   "history_scope": "complete within published flat/50m file for selected UIDs; not a completeness claim relative to 5B"},
        "cutoff": {"timestamp_raw": cutoff, "comparison": "development rows satisfy timestamp < cutoff; rows at or after cutoff are not interpreted or summarized"},
        "full_history_retention": {"all_published_events_for_selected_uids_written": True, "stored_row_count_arrow": full_rows,
                                   "file": "events_full.parquet", "post_cutoff_payload_interpreted": False},
        "development": {"file": "events.parquet", "columns": DEV_COLUMNS, "rows": dev_rows, "unique_users": len(user_counts),
                        "unique_items": len(dev_items), "items": "unique item IDs from development rows only",
                        "event_type_counts": dict(event_counts), "events_per_user": {"min": min(user_counts.values(), default=0),
                            "median": statistics.median(user_counts.values()) if user_counts else 0,
                            "max": max(user_counts.values(), default=0)},
                        "missing_value_counts": dict(null_counts),
                        "chronology": {"sort_key": ["uid", "timestamp"], "violations": chronology_violations,
                                       "equal_uid_timestamp_extra_rows": timestamp_tie_extra_rows,
                                       "equal_uid_timestamp_groups": timestamp_tie_groups,
                                       "checked_across_batch_boundaries": True}},
        "mappings_snapshot": {"temporal_availability": "not established; fixed published snapshot", "artist": artist_meta, "album": album_meta},
        "resources": {"duration_seconds": round(time.perf_counter() - started, 3)},
        "versions": {"python": sys.version, "pyarrow": pa.__version__},
        "source_rows_scanned_in_selected_event_pass": source_rows_scanned,
        "outputs": {"full_history": "events_full.parquet", "development": "events.parquet",
                    "artist_mapping": "artist_mapping.parquet", "album_mapping": "album_mapping.parquet"},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _prepare(output: Path, users: int, seed: int, cutoff: int, event_source: Path,
             artist_source: Path, album_source: Path, source_hashes: dict[str, str],
             batch_size: int = BATCH_SIZE, monitor=None) -> dict:
    """Run preparation and persist a failure marker for errors at any stage."""
    output = Path(output)
    is_new_output = not output.exists()
    try:
        return _prepare_inner(output, users, seed, cutoff, event_source, artist_source,
                              album_source, source_hashes, batch_size, monitor)
    except BaseException as exc:
        if is_new_output and output.exists():
            _write_failed(output, exc)
        raise


def prepare_scale(output: Path, users: int = 256, seed: int = 20261025, cutoff: int = 23380165,
                  event_source: Path = EVENT_SOURCE, artist_source: Path = ARTIST_SOURCE,
                  album_source: Path = ALBUM_SOURCE, monitor=None) -> dict:
    return _prepare(Path(output), users, seed, cutoff, Path(event_source), Path(artist_source), Path(album_source),
                    _manifest_hashes(), monitor=monitor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scale data only; does not train or evaluate models.")
    parser.add_argument("--users", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--cutoff", type=int, default=23380165)
    parser.add_argument("--output", type=Path, default=Path("artifacts/p003_scale256/data"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else TASK_ROOT / args.output
    result = prepare_scale(output, args.users, args.seed, args.cutoff)
    print(f"Prepared {result['development']['rows']:,} development rows and "
          f"{result['full_history_retention']['stored_row_count_arrow']:,} stored full-history rows for "
          f"{result['selected_uid_count']} users.")


if __name__ == "__main__": main()
