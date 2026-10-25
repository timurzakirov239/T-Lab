"""Freeze disjoint uniform user cohorts and prepare cohort data for P005."""
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
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .scale_data import (
        ALBUM_SOURCE, ARTIST_SOURCE, BATCH_SIZE, DATA_ROOT, DEV_COLUMNS,
        EVENT_SOURCE, _mapping_with_monitor,
    )
else:
    from scale_data import (
        ALBUM_SOURCE, ARTIST_SOURCE, BATCH_SIZE, DATA_ROOT, DEV_COLUMNS,
        EVENT_SOURCE, _mapping_with_monitor,
    )


def _hash_uid_list(values: Sequence[int]) -> str:
    payload = json.dumps([int(x) for x in values], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _check(monitor) -> None:
    if monitor is not None:
        monitor.check()


def _failed(output: Path, error: BaseException) -> None:
    try:
        (output / "FAILED.json").write_text(json.dumps({
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "note": "Partial files, if any, are diagnostic only; this output path is not reusable.",
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


def _new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"P005 output must be a new directory: {output}")
    output.mkdir(parents=True)


def prepare_memberships(output: Path, seeds: Sequence[int] = (20261026, 20261027), users: int = 64,
                        exclude_users: Sequence[int] = (), event_source: Path = EVENT_SOURCE,
                        monitor=None, batch_size: int = BATCH_SIZE) -> dict:
    """Select reproducible, sequentially disjoint cohorts from UID-only stream."""
    output = Path(output)
    _new_output(output)
    try:
        source = pq.ParquetFile(event_source)
        population: set[int] = set()
        rows_scanned = 0
        for batch in source.iter_batches(batch_size=batch_size, columns=["uid"], use_threads=False):
            _check(monitor)
            rows_scanned += batch.num_rows
            population.update(map(int, pc.unique(batch.column(0)).to_pylist()))
            if rows_scanned % 1_000_000 < batch.num_rows:
                print(f"P005 UID pass: {rows_scanned:,}/{source.metadata.num_rows:,}; users={len(population):,}", flush=True)
        ordered_population = sorted(population)
        excluded = set(map(int, exclude_users))
        if not excluded.issubset(population):
            raise ValueError("exclude_users contains UID(s) absent from source population")
        cohorts = []
        chosen_all: set[int] = set()
        for seed_value in seeds:
            seed = int(seed_value)
            available = [uid for uid in ordered_population if uid not in excluded and uid not in chosen_all]
            if users < 1 or users > len(available):
                raise ValueError(f"cannot sample {users} users for seed {seed}; {len(available)} remain")
            chosen = sorted(random.Random(seed).sample(available, users))
            chosen_all.update(chosen)
            cohorts.append({"seed": seed, "selected_users": chosen, "selected_user_count": len(chosen),
                            "selected_users_sha256": _hash_uid_list(chosen),
                            "available_population_before_sample": len(available)})
        membership = {
            "status": "frozen", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "UID-only population; random.sample(sorted(available UID population), users); no activity weighting",
            "seed_source": "Python random.Random(seed) per cohort; population sorted before sampling",
            "source_events": str(Path(event_source)), "source_rows_scanned": rows_scanned,
            "source_unique_uid_count": len(ordered_population), "source_uid_population_sha256": _hash_uid_list(ordered_population),
            "excluded_users": sorted(excluded), "excluded_user_count": len(excluded),
            "excluded_users_sha256": _hash_uid_list(sorted(excluded)),
            "seeds": [int(s) for s in seeds], "users_per_cohort": users, "cohorts": cohorts,
            "disjoint_cohorts": True,
        }
        (output / "memberships.json").write_text(json.dumps(membership, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return membership
    except BaseException as exc:
        _failed(output, exc)
        raise


def _prepare_cohort_inner(output: Path, selected_users: Sequence[int], seed: int, cutoff: int,
                          event_source: Path, artist_source: Path, album_source: Path,
                          monitor=None, batch_size: int = BATCH_SIZE) -> dict:
    _new_output(output)
    started = time.perf_counter()
    selected = sorted(set(map(int, selected_users)))
    if not selected or len(selected) != len(selected_users):
        raise ValueError("selected_users must be nonempty and unique")
    source = pq.ParquetFile(event_source)
    selected_values = pa.array(selected, type=source.schema_arrow.field("uid").type)

    full_path = output / "events_full.parquet"
    dev_path = output / "events.parquet"
    full_writer = dev_writer = None
    full_rows = dev_rows = source_rows_scanned = 0
    dev_items: set[int] = set()
    event_counts: Counter[str] = Counter()
    per_user: Counter[int] = Counter()
    chronology_violations = ties_extra = ties_groups = 0
    previous: tuple[int, int] | None = None
    tie_group_rows = 0
    event_error = None
    try:
        for batch in source.iter_batches(batch_size=batch_size, use_threads=False):
            _check(monitor)
            source_rows_scanned += batch.num_rows
            mask = pc.is_in(batch.column(batch.schema.get_field_index("uid")), value_set=selected_values)
            selected_full = batch.filter(mask)
            if selected_full.num_rows:
                if full_writer is None:
                    full_writer = pq.ParquetWriter(full_path, selected_full.schema, compression="zstd")
                full_writer.write_batch(selected_full)
                full_rows += selected_full.num_rows
            if not selected_full.num_rows:
                continue
            # Reserved/post-cutoff payload stays Arrow-only and is written intact above.
            timestamp_col = selected_full.column(selected_full.schema.get_field_index("timestamp"))
            dev = selected_full.filter(pc.less(timestamp_col, pa.scalar(cutoff, type=timestamp_col.type)))
            if not dev.num_rows:
                continue
            dev = dev.select(DEV_COLUMNS)
            if dev_writer is None:
                dev_writer = pq.ParquetWriter(dev_path, dev.schema, compression="zstd")
            dev_writer.write_batch(dev)
            dev_rows += dev.num_rows
            for start in range(0, dev.num_rows, 8192):
                _check(monitor)
                part = dev.slice(start, 8192)
                columns = {name: part.column(name).to_pylist() for name in DEV_COLUMNS}
                for i, raw_uid in enumerate(columns["uid"]):
                    uid, ts, item = int(raw_uid), int(columns["timestamp"][i]), int(columns["item_id"][i])
                    if previous is not None:
                        old_uid, old_ts = previous
                        if uid < old_uid or (uid == old_uid and ts < old_ts): chronology_violations += 1
                        if uid == old_uid and ts == old_ts:
                            ties_extra += 1
                            tie_group_rows += 1
                        else:
                            if tie_group_rows > 1: ties_groups += 1
                            tie_group_rows = 1
                    else:
                        tie_group_rows = 1
                    previous = (uid, ts)
                    per_user[uid] += 1
                    dev_items.add(item)
                    event_counts[str(columns["event_type"][i])] += 1

    except BaseException as exc:
        event_error = exc
    finally:
        for writer in (full_writer, dev_writer):
            if writer is not None:
                try:
                    writer.close()
                except BaseException as close_error:
                    if event_error is None: event_error = close_error
    if event_error is not None: raise event_error
    if tie_group_rows > 1: ties_groups += 1
    if full_rows == 0: raise ValueError("selected users have no events")
    if dev_rows == 0:
        schema = pa.schema([source.schema_arrow.field(name) for name in DEV_COLUMNS])
        pq.write_table(pa.Table.from_batches([], schema=schema), dev_path, compression="zstd")

    _check(monitor)
    artist = _mapping_with_monitor(artist_source, output / "artist_mapping.parquet", "item_id", "artist_id", dev_items, batch_size, monitor)
    album = _mapping_with_monitor(album_source, output / "album_mapping.parquet", "item_id", "album_id", dev_items, batch_size, monitor)
    _check(monitor)
    for name, meta in (("artist", artist), ("album", album)):
        output_path = output / f"{name}_mapping.parquet"
        if pq.ParquetFile(output_path).metadata.num_rows != meta["unique_links"]:
            raise RuntimeError(f"{name} mapping output rows differ from unique links")
    metadata = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed), "selected_users": selected, "selected_user_count": len(selected),
        "selected_users_sha256": _hash_uid_list(selected),
        "cutoff_timestamp_raw": int(cutoff), "dev_filter": "timestamp < cutoff; no calendar conversion",
        "source": {"events_path": str(event_source), "source_row_count": source.metadata.num_rows,
                   "history_scope": "all published events for selected UIDs in the source file; no claim of completeness relative to 5B"},
        "full_history_retention": {"file": "events_full.parquet", "all_selected_uid_events_written": True,
                                   "stored_full_rows_arrow_count": full_rows, "post_cutoff_payload_decoded": False},
        "development": {"file": "events.parquet", "columns": DEV_COLUMNS, "rows": dev_rows,
                        "unique_users": len(per_user), "unique_items": len(dev_items), "event_type_counts": dict(event_counts),
                        "events_per_user": {"min": min(per_user.values(), default=0),
                                            "median": statistics.median(per_user.values()) if per_user else 0,
                                            "max": max(per_user.values(), default=0)},
                        "chronology": {"sort_key": ["uid", "timestamp"], "violations": chronology_violations,
                                       "equal_uid_timestamp_extra_rows": ties_extra,
                                       "equal_uid_timestamp_groups": ties_groups, "checked_across_batch_boundaries": True}},
        "mappings_snapshot": {"temporal_availability": "unknown; fixed snapshot assumption", "artist": artist, "album": album},
        "source_rows_scanned": source_rows_scanned, "duration_seconds": round(time.perf_counter() - started, 3),
        "versions": {"python": sys.version, "pyarrow": pa.__version__},
        "outputs": {"full_history": "events_full.parquet", "development": "events.parquet",
                    "artist_mapping": "artist_mapping.parquet", "album_mapping": "album_mapping.parquet"},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def prepare_cohort(output: Path, selected_users: Sequence[int], seed: int, cutoff: int = 23380165,
                   event_source: Path = EVENT_SOURCE, artist_source: Path = ARTIST_SOURCE,
                   album_source: Path = ALBUM_SOURCE, monitor=None,
                   batch_size: int = BATCH_SIZE) -> dict:
    """Write one frozen cohort's full history and its development-only table."""
    output = Path(output)
    is_new_output = not output.exists()
    try:
        return _prepare_cohort_inner(output, selected_users, seed, cutoff, Path(event_source),
                                     Path(artist_source), Path(album_source), monitor, batch_size)
    except BaseException as exc:
        if is_new_output and output.exists(): _failed(output, exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P005 cohort membership and one cohort's data.")
    parser.add_argument("--memberships", type=Path, required=True)
    parser.add_argument("--cohort-index", type=int, default=0)
    parser.add_argument("--cohort-output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=23380165)
    args = parser.parse_args()
    memberships_path = args.memberships
    memberships = json.loads(memberships_path.read_text(encoding="utf-8"))
    cohort = memberships["cohorts"][args.cohort_index]
    prepare_cohort(args.cohort_output, cohort["selected_users"], cohort["seed"], args.cutoff)


if __name__ == "__main__": main()
