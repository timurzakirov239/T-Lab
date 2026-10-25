"""Run the bounded development pilot; never evaluate the reserved final period."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import heapq
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import pyarrow.parquet as pq

if __package__:
    from .pilot_evaluation import evaluate_ranking
    from .pilot_features import FeatureEngine
    from .pilot_storage import ensure_run_writable
else:
    from pilot_evaluation import evaluate_ranking
    from pilot_features import FeatureEngine
    from pilot_storage import ensure_run_writable

TASK = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ResourceMonitor:
    def __init__(self, limits: dict):
        self.limits = limits
        self.process = psutil.Process()
        self.peak_rss_mb = 0.0
        self.min_available_mb = math.inf
        self.exceeded = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _sample(self):
        info = self.process.memory_info()
        rss = max(info.rss, getattr(info, "peak_wset", 0)) / 2**20
        available = psutil.virtual_memory().available / 2**20
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        self.min_available_mb = min(self.min_available_mb, available)
        if rss > self.limits["runner_peak_rss_mb"]:
            self.exceeded = f"Process RSS limit exceeded: {rss:.0f} MiB"
        if available < self.limits["minimum_available_mb"]:
            self.exceeded = f"Available memory below reserve: {available:.0f} MiB"

    def _watch(self):
        while not self.stop.is_set():
            self._sample()
            self.stop.wait(0.25)

    def start(self):
        self._sample()
        self.thread.start()

    def check(self):
        self._sample()
        if self.exceeded:
            raise MemoryError(self.exceeded)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self._sample()
        return {"peak_process_rss_mb": round(self.peak_rss_mb, 2), "minimum_available_system_mb": round(self.min_available_mb, 2), "sampling_interval_seconds": 0.25, "limits": self.limits, "limit_exceeded": self.exceeded}


def load_mapping(path: Path, entity_name: str):
    columns = pq.read_table(path, columns=["item_id", entity_name], use_threads=False).to_pydict()
    values = defaultdict(set)
    for item, entity in zip(columns["item_id"], columns[entity_name]):
        values[int(item)].add(int(entity))
    return {item: tuple(sorted(entities)) for item, entities in values.items()}


def load_events(path: Path):
    data = pq.read_table(path, columns=["uid", "item_id", "timestamp", "event_type"], use_threads=False)
    columns = data.to_pydict()
    types = sorted(set(columns["event_type"]))
    mapping = {name: i for i, name in enumerate(types)}
    uid = np.asarray(columns["uid"], dtype=np.uint32)
    item = np.asarray(columns["item_id"], dtype=np.uint32)
    timestamp = np.asarray(columns["timestamp"], dtype=np.uint32)
    event_type = np.asarray([mapping[t] for t in columns["event_type"]], dtype=np.uint8)
    del columns, data
    # Stable deterministic order; simultaneous events are only observed after queries.
    order = np.lexsort((item, uid, timestamp))
    return {"uid": uid[order], "item_id": item[order], "timestamp": timestamp[order], "event_code": event_type[order], "event_names": types}


def select_queries(events: dict, config: dict):
    times = events["timestamp"]
    boundaries = {"train_query_start": int(np.quantile(times, .30, method="lower")), "train_end": int(np.quantile(times, .70, method="lower")), "validation_end": int(np.quantile(times, .85, method="lower"))}
    listen_code = events["event_names"].index("listen")
    # Count all listen rows in a (user,timestamp) group before selecting one.
    indices = np.flatnonzero(events["event_code"] == listen_code)
    grouped = defaultdict(list)
    for index in indices:
        t = int(times[index])
        if boundaries["train_query_start"] <= t < boundaries["validation_end"]:
            grouped[(int(events["uid"][index]), t)].append(int(index))
    per_user = defaultdict(list)
    ambiguous = Counter()
    for (uid, t), matches in grouped.items():
        split = "train" if t < boundaries["train_end"] else "validation"
        if len(matches) != 1:
            ambiguous[split] += 1
            continue
        per_user[(uid, split)].append(matches[0])
    rng = np.random.default_rng(config["seed"])
    selected = []
    available = Counter()
    for (uid, split), candidates in sorted(per_user.items()):
        available[split] += len(candidates)
        limit = config[f"{split}_queries_per_user"]
        chosen = rng.choice(candidates, size=min(limit, len(candidates)), replace=False)
        for index in chosen:
            selected.append({"user_id": uid, "timestamp": int(times[index]), "target_item": int(events["item_id"][index]), "split": split})
    selected.sort(key=lambda q: (q["timestamp"], q["user_id"], q["target_item"]))
    for qid, query in enumerate(selected):
        query["query_id"] = qid
    return selected, boundaries, {"ambiguous_uid_timestamp_listen_groups_excluded": dict(ambiguous), "eligible_unambiguous_queries_before_sampling": dict(available), "sampled_queries": dict(Counter(q["split"] for q in selected)), "reserved_period_queries_generated": 0}


class CandidateGenerator:
    def __init__(self, artist_map, album_map, config):
        self.artists, self.albums = artist_map, album_map
        self.config = config
        self.counts = Counter()
        self.last_listen = defaultdict(dict)
        self.user_artists = defaultdict(Counter)
        self.user_albums = defaultdict(Counter)
        self.artist_items, self.album_items = defaultdict(set), defaultdict(set)
        for item, entities in artist_map.items():
            for entity in entities:
                self.artist_items[entity].add(item)
        for item, entities in album_map.items():
            for entity in entities:
                self.album_items[entity].add(item)
        self.catalog = set()
        self.previous_t = None
        self._popular_cache = []
        self._popular_cache_time = None

    def observe(self, event):
        uid, item, t = event["uid"], event["item_id"], event["timestamp"]
        self.catalog.add(item)
        if event["event_type"] == "listen":
            self.counts[item] += 1
            self.last_listen[uid][item] = t
            for entities, profile in ((self.artists.get(item, ()), self.user_artists[uid]), (self.albums.get(item, ()), self.user_albums[uid])):
                for entity in entities:
                    profile[entity] += 1.0 / len(entities)
        self.previous_t = t

    @staticmethod
    def top(counter, limit):
        return [key for key, _ in heapq.nsmallest(limit, counter.items(), key=lambda kv: (-kv[1], kv[0]))]

    def _related(self, profile, entity_items, limit):
        scores = defaultdict(float)
        for entity in self.top(profile, 5):
            for item in entity_items.get(entity, ()):
                if item in self.catalog and self.counts[item] > 0:
                    scores[item] += profile[entity] * math.log1p(self.counts[item])
        return self.top(scores, limit)

    def candidates(self, uid, timestamp):
        if self.previous_t is not None and self.previous_t >= timestamp:
            raise ValueError("Candidate history must be strictly before query")
        cap = self.config["max_candidates"]
        quotas = self.config["candidate_quotas"]
        if self._popular_cache_time != timestamp:
            self._popular_cache = self.top(self.counts, cap)
            self._popular_cache_time = timestamp
        sources = [
            ("popular", self._popular_cache),
            ("repeat", self.top(self.last_listen.get(uid, {}), cap)),
            ("artist", self._related(self.user_artists.get(uid, {}), self.artist_items, cap)),
            ("album", self._related(self.user_albums.get(uid, {}), self.album_items, cap)),
        ]
        selected, seen = [], set()
        for name, items in sources:
            added = 0
            for item in items:
                if item not in seen and item in self.catalog:
                    selected.append(item)
                    seen.add(item)
                    added += 1
                    if added >= quotas[name] or len(selected) >= cap:
                        break
            if len(selected) >= cap:
                break
        for item in self._popular_cache:
            if len(selected) >= cap:
                break
            if item not in seen:
                selected.append(item)
                seen.add(item)
        return selected


def replay_features(events, queries, artist_map, album_map, config, short_window, half_life, monitor, out):
    started = time.perf_counter()
    engine = FeatureEngine(artist_map, album_map, short_window=short_window, half_life=half_life)
    generator = CandidateGenerator(artist_map, album_map, config)
    rows = []
    matrix_parts, offsets, sizes = [], [], []
    index = 0
    offset = 0
    next_log = time.perf_counter() + 15
    for original in queries:
        query = dict(original)
        t = query["timestamp"]
        while index < len(events["timestamp"]) and int(events["timestamp"][index]) < t:
            event = {"uid": int(events["uid"][index]), "item_id": int(events["item_id"][index]), "timestamp": int(events["timestamp"][index]), "event_type": events["event_names"][int(events["event_code"][index])]}
            engine.observe(event)
            generator.observe(event)
            index += 1
            if index % 5000 == 0:
                monitor.check()
        candidates = generator.candidates(query["user_id"], t)
        target = query["target_item"]
        query.update(candidate_items=candidates, target_is_repeat=target in generator.last_listen.get(query["user_id"], {}), target_known_before=target in generator.catalog)
        query["popularity_scores"] = [float(generator.counts[item]) for item in candidates]
        if any(item not in generator.catalog for item in candidates):
            raise AssertionError("Future item leaked into candidate list")
        features = engine.transform(query["user_id"], candidates, t)
        if not np.isfinite(features).all():
            raise ValueError("Feature matrix contains nonfinite values")
        offsets.append(offset)
        sizes.append(len(candidates))
        offset += len(candidates)
        matrix_parts.append(features)
        rows.append(query)
        if time.perf_counter() >= next_log:
            print(f"FEATURES window={short_window}: queries {len(rows)}/{len(queries)}, events {index:,}, RSS={psutil.Process().memory_info().rss/2**20:.0f} MiB", flush=True)
            next_log = time.perf_counter() + 15
    matrix = np.concatenate(matrix_parts, axis=0) if matrix_parts else np.empty((0, len(engine.feature_names)), dtype=np.float32)
    names, groups = list(engine.feature_names), dict(engine.feature_groups)
    details = {"short_window_raw": short_window, "half_life_raw": half_life, "rows": int(matrix.shape[0]), "features": int(matrix.shape[1]), "matrix_mb": round(matrix.nbytes / 2**20, 3), "events_observed": index, "elapsed_seconds": round(time.perf_counter()-started, 3), "feature_names": names, "feature_groups": groups}
    write_json(out / f"feature_build_{short_window}.json", details)
    del matrix_parts, engine, generator
    gc.collect()
    monitor.check()
    return matrix, rows, np.asarray(offsets), np.asarray(sizes), details


def train_and_score(label, allowed_groups, matrix, queries, offsets, sizes, feature_info, config, out, monitor):
    from catboost import CatBoostClassifier
    columns = [i for i, name in enumerate(feature_info["feature_names"]) if feature_info["feature_groups"][name] in allowed_groups]
    train_indices, y = [], []
    validation_positions = []
    missing_train = 0
    for index, query in enumerate(queries):
        if query["split"] == "train":
            if query["target_item"] not in query["candidate_items"]:
                missing_train += 1
                continue
            train_indices.extend(range(int(offsets[index]), int(offsets[index]+sizes[index])))
            y.extend(int(item == query["target_item"]) for item in query["candidate_items"])
        else:
            validation_positions.append(index)
    if len(set(y)) != 2:
        raise ValueError("Pilot training set needs retrieved positives and unmarked candidates")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    x_train = matrix[np.ix_(train_indices, columns)]
    model = CatBoostClassifier(**config["model"], random_seed=config["seed"])
    monitor.check()
    started = time.perf_counter()
    model.fit(x_train, np.asarray(y, dtype=np.uint8))
    fit_seconds = time.perf_counter() - started
    del x_train
    gc.collect()
    predict_started = time.perf_counter()
    all_scores = model.predict_proba(matrix[:, columns])[:, 1]
    scores = [all_scores[int(offsets[i]):int(offsets[i]+sizes[i])] for i in validation_positions]
    validation = [queries[i] for i in validation_positions]
    metrics = evaluate_ranking(validation, scores, ks=tuple(config["metrics_k"]))
    prediction_seconds = time.perf_counter() - predict_started
    model.save_model(str(out / "models" / f"{label}.cbm"))
    np.savez_compressed(out / "predictions" / f"{label}.npz", query_ids=np.asarray([q["query_id"] for q in validation]), offsets=np.cumsum([0] + [len(s) for s in scores]), scores=np.concatenate(scores) if scores else np.empty(0))
    result = {"model": label, "groups": sorted(allowed_groups), "feature_count": len(columns), "feature_names": [feature_info["feature_names"][i] for i in columns], "train_rows": int(len(y)), "train_positives": int(sum(y)), "train_queries_missing_target": missing_train, "fit_seconds": round(fit_seconds, 3), "predict_and_evaluate_seconds": round(prediction_seconds, 3), "short_window_raw": feature_info["short_window_raw"], "metrics": metrics}
    write_json(out / "results" / f"{label}.json", result)
    print(f"MODEL {label}: NDCG@20={metrics['overall']['user_macro']['ndcg@20']:.5f}, Recall@20={metrics['overall']['user_macro']['recall@20']:.5f}, fit={fit_seconds:.1f}s", flush=True)
    del model, all_scores, train_indices, y
    gc.collect()
    monitor.check()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "configs" / "pilot_001.json")
    parser.add_argument("--output", type=Path, default=TASK / "artifacts" / "pilot_001")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("test_evaluation"):
        raise ValueError("Final test evaluation is forbidden in this development runner")
    out = args.output.resolve()
    ensure_run_writable(out)
    for directory in (out, out / "models", out / "results", out / "predictions"):
        directory.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()
    started = time.perf_counter()
    result = {"run_id": config["run_id"], "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, "config_sha256": sha256(args.config), "source_code_sha256": {file.name: sha256(file) for file in (TASK / "src").glob("pilot*.py")}, "models": [], "final_test_evaluated": False}
    result["source_code_sha256"]["run_pilot.py"] = sha256(Path(__file__))
    write_json(out / "run_status.json", result)
    try:
        data_dir = out / "data"
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["unique_users"] != config["users"] or metadata["seed"] != config["seed"]:
            raise ValueError("Prepared data cohort does not match pilot configuration")
        result["data_metadata_sha256"] = sha256(data_dir / "metadata.json")
        result["prepared_data_sha256"] = {name: sha256(data_dir / name) for name in ("events.parquet", "artist_mapping.parquet", "album_mapping.parquet")}
        preparation_started = time.perf_counter()
        events = load_events(data_dir / "events.parquet")
        artists = load_mapping(data_dir / "artist_mapping.parquet", "artist_id")
        albums = load_mapping(data_dir / "album_mapping.parquet", "album_id")
        queries, bounds, query_selection = select_queries(events, config)
        if not any(q["split"] == "validation" for q in queries):
            raise ValueError("No validation queries")
        assert all(q["timestamp"] < bounds["validation_end"] for q in queries)
        result.update(boundaries_raw=bounds, query_selection=query_selection, data_rows=metadata["rows"], unique_users=metadata["unique_users"], unique_items=metadata["unique_items"], load_and_query_selection_seconds=round(time.perf_counter()-preparation_started, 3))
        print(f"LOADED {metadata['rows']:,} events / {metadata['unique_users']} users; queries={query_selection['sampled_queries']}; final test reserved", flush=True)
        train_span = max(1, bounds["train_end"] - int(events["timestamp"].min()))
        windows = [max(1, int(train_span * fraction)) for fraction in config["short_window_fractions"]]
        half_life = max(1, int(train_span * config["half_life_fraction"]))
        queries_canonical = None
        for attempt, window in enumerate(dict.fromkeys(windows)):
            monitor.check()
            matrix, observed_queries, offsets, sizes, feature_info = replay_features(events, queries, artists, albums, config, window, half_life, monitor, out)
            if queries_canonical is None:
                queries_canonical = observed_queries
                with (out / "queries.jsonl").open("w", encoding="utf-8") as stream:
                    for query in queries_canonical:
                        stream.write(json.dumps(query) + "\n")
                validation = [q for q in observed_queries if q["split"] == "validation"]
                popular = evaluate_ranking(validation, [q["popularity_scores"] for q in validation], ks=tuple(config["metrics_k"]))
                result["models"].append({"model": "B0_popularity", "groups": [], "feature_count": 1, "fit_seconds": 0, "metrics": popular})
                families = [("B1_I", "I"), ("B2_UI", "UI"), ("B3_UID", "UID"), ("B4_UIC", "UIC"), ("B5_UIDC", "UIDC"), ("B6_IDC", "IDC"), ("B7_UDC", "UDC"), ("X1_UIDCX", "UIDCX")]
            else:
                if any(q["candidate_items"] != base["candidate_items"] or q["query_id"] != base["query_id"] for q, base in zip(observed_queries, queries_canonical)):
                    raise AssertionError("Candidate/query sets changed across feature windows")
                families = [(f"A_window_{window}", "UIDC")]
            for label, groups in families:
                result["models"].append(train_and_score(label, set(groups), matrix, observed_queries, offsets, sizes, feature_info, config, out, monitor))
                write_json(out / "run_status.json", result)
            del matrix
            gc.collect()
        result["status"] = "complete"
        result["best_validation_model"] = max(result["models"], key=lambda item: item["metrics"]["overall"]["user_macro"]["ndcg@20"])["model"]
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = round(time.perf_counter()-started, 3)
        result["resources"] = monitor.finish()
        write_json(out / "run_status.json", result)
        write_json(out / "metrics.json", result)
    print(f"PILOT COMPLETE: {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
