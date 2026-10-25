"""Load published Yambda history and select frozen-protocol final-test queries.

The loader deliberately requests only the four public event columns required by
the evaluator. Payload added to later Parquet versions is never materialized.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_BATCH_SIZE = 65_536
_ARROW_COLUMNS = ("uid", "item_id", "timestamp", "event_type")
_UINT32_MAX = np.iinfo(np.uint32).max


def _check(monitor: Any) -> None:
    if monitor is not None:
        check = getattr(monitor, "check", None)
        if check is None or not callable(check):
            raise TypeError("monitor must expose a callable check() method")
        check()


def _uint32_column(batch: pa.RecordBatch, name: str) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    if not pa.types.is_integer(column.type):
        raise TypeError(f"{name} must be an Arrow integer column")
    if column.null_count:
        raise ValueError(f"{name} contains nulls")
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} did not convert to an integer array")
    if values.dtype.kind == "i" and values.size and int(values.min()) < 0:
        raise ValueError(f"{name} contains negative values")
    if values.size and int(values.max()) > _UINT32_MAX:
        raise ValueError(f"{name} exceeds uint32 range")
    return values.astype(np.uint32, copy=False)


def load_published_history(path: str | Path, monitor: Any = None) -> dict[str, Any]:
    """Load all published event rows, sorted by timestamp, UID, then item ID.

    ``event_type`` is encoded using sorted, observed event names. All four
    columns are read in batches; timestamps are not filtered at the former
    development/test boundary.
    """
    path = Path(path)
    parquet = pq.ParquetFile(path)
    missing = [name for name in _ARROW_COLUMNS if name not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"published event source lacks required columns: {missing}")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ("uid", "item_id", "timestamp", "event_code")}
    insertion_names: list[str] = []
    name_to_code: dict[str, int] = {}
    for batch in parquet.iter_batches(batch_size=_BATCH_SIZE, columns=list(_ARROW_COLUMNS), use_threads=False):
        if batch.num_rows:
            chunks["uid"].append(_uint32_column(batch, "uid"))
            chunks["item_id"].append(_uint32_column(batch, "item_id"))
            chunks["timestamp"].append(_uint32_column(batch, "timestamp"))
            event_column = batch.column(batch.schema.get_field_index("event_type"))
            if event_column.null_count:
                raise ValueError("event_type contains nulls")
            if not (pa.types.is_string(event_column.type) or pa.types.is_large_string(event_column.type)
                    or pa.types.is_dictionary(event_column.type)):
                raise TypeError("event_type must be a string or dictionary-encoded string column")
            names = event_column.to_pylist()
            codes = np.empty(len(names), dtype=np.uint8)
            for index, name in enumerate(names):
                if not isinstance(name, str) or not name:
                    raise ValueError("event_type names must be nonempty strings")
                code = name_to_code.get(name)
                if code is None:
                    if len(insertion_names) >= 256:
                        raise ValueError("event_type has more than 256 distinct names")
                    code = len(insertion_names)
                    name_to_code[name] = code
                    insertion_names.append(name)
                codes[index] = code
            chunks["event_code"].append(codes)
        _check(monitor)

    arrays = {name: (np.concatenate(parts) if parts else np.empty(0, dtype=dtype))
              for name, parts, dtype in (
                  ("uid", chunks["uid"], np.uint32),
                  ("item_id", chunks["item_id"], np.uint32),
                  ("timestamp", chunks["timestamp"], np.uint32),
                  ("event_code", chunks["event_code"], np.uint8))}
    event_names = sorted(insertion_names)
    remap = np.asarray([event_names.index(name) for name in insertion_names], dtype=np.uint8)
    if arrays["event_code"].size:
        arrays["event_code"] = remap[arrays["event_code"]]
    order = np.lexsort((arrays["item_id"], arrays["uid"], arrays["timestamp"]))
    return {**{name: values[order] for name, values in arrays.items()}, "event_names": event_names}


def select_test_queries(events: dict[str, Any], seed: int, start: int = 23_380_165,
                        limit: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select at most ``limit`` unambiguous test listens per user.

    A group is eligible when exactly one listen exists for a ``(uid,timestamp)``
    pair at or after ``start``. Every UID in the supplied frozen history is
    considered; sampling is per UID, not a user-population subsample. Queries
    retain the evaluator's ``validation`` split label and carry the explicit
    ``evaluation_period='test'`` marker. Candidate generation can later add
    candidates and causal target annotations; this function does not mutate the
    event arrays or omit a target based on candidate availability.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)) or start < 0:
        raise ValueError("start must be a non-negative integer timestamp")
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("limit must be a non-negative integer per UID")
    required = ("uid", "item_id", "timestamp", "event_code", "event_names")
    if any(name not in events for name in required):
        raise ValueError("events must contain uid, item_id, timestamp, event_code, and event_names")
    uid = np.asarray(events["uid"])
    item = np.asarray(events["item_id"])
    times = np.asarray(events["timestamp"])
    codes = np.asarray(events["event_code"])
    if any(array.ndim != 1 for array in (uid, item, times, codes)) or not (len(uid) == len(item) == len(times) == len(codes)):
        raise ValueError("event arrays must be one-dimensional and have equal lengths")
    names = list(events["event_names"])
    if "listen" not in names:
        raise ValueError("events do not contain a listen event type")
    listen_code = names.index("listen")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero((codes == listen_code) & (times >= int(start))):
        groups[(int(uid[index]), int(times[index]))].append(int(index))
    per_user: dict[int, list[int]] = defaultdict(list)
    ambiguous_groups = 0
    for (user_id, _timestamp), indices in groups.items():
        if len(indices) == 1:
            per_user[user_id].append(indices[0])
        else:
            ambiguous_groups += 1

    all_users = set(map(int, np.unique(uid)))
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    eligible_query_count = 0
    for user_id in sorted(all_users):
        eligible = sorted(per_user.get(user_id, []), key=lambda i: (int(times[i]), int(item[i])))
        eligible_query_count += len(eligible)
        if not eligible or limit == 0:
            continue
        chosen = rng.choice(np.asarray(eligible, dtype=np.int64), size=min(int(limit), len(eligible)), replace=False)
        for index in chosen:
            selected.append({"user_id": user_id, "timestamp": int(times[index]),
                             "target_item": int(item[index]), "split": "validation",
                             "evaluation_period": "test"})
    selected.sort(key=lambda query: (query["timestamp"], query["user_id"], query["target_item"]))
    for query_id, query in enumerate(selected):
        query["query_id"] = query_id
    eligible_users = sum(bool(per_user.get(user_id)) for user_id in all_users)
    sampled_users = len({query["user_id"] for query in selected})
    info = {"total_users": len(all_users), "eligible_users": eligible_users,
            "sampled_users": sampled_users, "users_without_eligible_queries": len(all_users) - eligible_users,
            "eligible_queries_before_sampling": eligible_query_count, "sampled_queries": len(selected),
            "ambiguous_uid_timestamp_listen_groups_excluded": ambiguous_groups,
            "queries_per_user_limit": int(limit), "start_timestamp": int(start), "seed": int(seed)}
    return selected, info
