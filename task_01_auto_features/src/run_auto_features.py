"""P004: bounded causal feature grammar and inner-only feature selection."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__:
    from .auto_features import AutoFeatureEngine, generate_specs
    from .auto_search import temporal_split, offsets_sizes, eligible_training_rows, prune_columns, select_trial_columns, pick_best
    from .pilot_evaluation import evaluate_ranking
    from .run_feature_benchmark import feature_build
    from .run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from .run_retrieval import load_development_events, load_frozen_queries
else:
    from auto_features import AutoFeatureEngine, generate_specs
    from auto_search import temporal_split, offsets_sizes, eligible_training_rows, prune_columns, select_trial_columns, pick_best
    from pilot_evaluation import evaluate_ranking
    from run_feature_benchmark import feature_build
    from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from run_retrieval import load_development_events, load_frozen_queries

TASK = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_SHA256 = "f7a6b1735bc0abf24b260a29ecd12040153275a9576401af7b8e638d2feede2a"
FROZEN_PROTOCOL_SHA256 = "46d7eed8d9aec15b7a722ccb2f01eca471618457bce14615c452d89e12adb8a8"


def validate_frozen_config(config_path, protocol_path):
    if sha256(config_path) != FROZEN_CONFIG_SHA256 or sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("P004 configuration/protocol changed; freeze a separate experiment before running")


def candidate_rows(positions, offsets, sizes):
    return np.concatenate([np.arange(offsets[int(i)], offsets[int(i)] + sizes[int(i)], dtype=np.int64)
                           for i in positions]) if len(positions) else np.empty(0, dtype=np.int64)


def assert_manual_equivalence(matrix, baseline, specs, schema, output):
    names = [spec["feature_id"] for spec in specs if spec["family"] == "A1"]
    expected = [name for name in schema["feature_names"] if schema["feature_groups"][name] in "UIDC"]
    if names != expected or len(names) != 66 or matrix.shape != (184800, 206):
        raise ValueError("A1 schema or fixed input shape differs from P003")
    automatic_cols = [i for i, spec in enumerate(specs) if spec["family"] == "A1"]
    baseline_cols = [schema["feature_names"].index(name) for name in names]
    digest = hashlib.sha256()
    checked = 0
    for start in range(0, len(matrix), 2048):
        a = matrix[start:start + 2048, automatic_cols]
        b = baseline[start:start + 2048, baseline_cols]
        if a.dtype != np.float32 or b.dtype != np.float32 or a.tobytes() != b.tobytes():
            raise AssertionError("Automatic A1 values differ from frozen manual control; training forbidden")
        digest.update(a.tobytes())
        checked += a.size
    result = {"status": "pass", "cells_compared": checked, "feature_ids": names,
              "identical_float32_bytes": True, "values_sha256": digest.hexdigest()}
    write_json(output, result)
    return result


def union_columns(matrix, fit_rows, manual, automatic):
    """Keep manual control intact and append only novel fit-fold expressions."""
    result, removed = list(manual), []
    digests = {}
    for column in dict.fromkeys([*manual, *automatic]):
        digest = hashlib.sha256()
        for start in range(0, len(fit_rows), 8192):
            values = matrix[fit_rows[start:start + 8192], column].copy()
            values[values == 0] = 0  # numerical equality treats +/-zero equally
            digest.update(values.tobytes())
        digests[column] = digest.hexdigest()
    for column in automatic:
        if column in result:
            removed.append({"column": column, "reason": "already_manual", "representative": column})
            continue
        duplicate = None
        for previous in result:
            if digests[column] != digests[previous]:
                continue
            if all(np.array_equal(matrix[fit_rows[s:s + 8192], column], matrix[fit_rows[s:s + 8192], previous])
                   for s in range(0, len(fit_rows), 8192)):
                duplicate = previous
                break
        if duplicate is not None:
            removed.append({"column": column, "reason": "duplicate_on_inner_fit", "representative": duplicate})
        else:
            result.append(column)
    return sorted(result), removed


def final_masks(specs, first, second, union):
    manual = [i for i, spec in enumerate(specs) if spec["family"] == "A1"]
    masks = {}
    for group_set in ("I", "UI", "UID", "UIC", "UIDC", "IDC", "UDC"):
        masks[f"M_{group_set}"] = [i for i in manual if specs[i]["group"] in group_set]
    for relation in ("artist", "album"):
        masks[f"M_UID_C{relation}"] = [i for i in manual if specs[i]["group"] != "C" or specs[i]["path"] == f"c_{relation}"]
    masks.update(Auto_A1=list(first), Auto_A2=list(second), Union=list(union))
    for group in "UIDC":
        masks[f"Union_without_{group}"] = [i for i in union if specs[i]["group"] != group]
    return masks


class MemoryCallback:
    def __init__(self, monitor):
        self.monitor = monitor

    def after_iteration(self, info):
        if info.iteration % 10 == 0:
            try:
                self.monitor.check()
            except MemoryError:
                return False
        return True


def fit_and_predict(label, columns, fit_positions, evaluation_positions, matrix, queries, offsets, sizes,
                    specs, config, output, monitor, status, stage):
    from catboost import CatBoostClassifier
    if not columns:
        raise ValueError("A fitted feature subset must be nonempty")
    if len(status["attempts"]) >= config["maximum_total_fits"]:
        raise ValueError("P004 total fit budget exhausted")
    rows, labels, train_counts = eligible_training_rows(queries, fit_positions, offsets, sizes)
    if len(np.unique(labels)) != 2:
        raise ValueError("Conditional training must contain retrieved positives and unmarked candidates")
    attempt = {"label": label, "stage": stage, "status": "started", "columns": list(columns), "training": train_counts}
    status["attempts"].append(attempt)
    write_json(output / "run_status.json", status)
    tick = time.perf_counter()
    monitor.check()
    x_train = matrix[np.ix_(rows, columns)]
    model = CatBoostClassifier(**config["model"], random_seed=config["seed"])
    try:
        model.fit(x_train, labels, callbacks=[MemoryCallback(monitor)])
        monitor.check()
        fit_seconds = time.perf_counter() - tick
        importances = model.feature_importances_.tolist()
        del x_train, rows, labels
        gc.collect()
        tick = time.perf_counter()
        scores = []
        for start in range(0, len(evaluation_positions), 64):
            positions = evaluation_positions[start:start + 64]
            flat_rows = candidate_rows(positions, offsets, sizes)
            batch = model.predict_proba(matrix[np.ix_(flat_rows, columns)], thread_count=2)[:, 1]
            local = 0
            for position in positions:
                size = int(sizes[int(position)])
                scores.append(batch[local:local + size].copy())
                local += size
            monitor.check()
        evaluated = [queries[int(i)] for i in evaluation_positions]
        metrics = evaluate_ranking(evaluated, scores, ks=tuple(config["metrics_k"]))
        prediction = output / "predictions" / f"{label}.npz"
        np.savez_compressed(prediction, query_ids=np.asarray([q["query_id"] for q in evaluated]),
                            offsets=np.cumsum([0] + [len(s) for s in scores]), scores=np.concatenate(scores))
        model_path = output / "models" / f"{label}.cbm"
        model.save_model(str(model_path))
        result = {"label": label, "stage": stage, "columns": list(columns),
                  "spec_ids": [specs[i]["feature_id"] for i in columns], "feature_count": len(columns),
                  "training": train_counts, "fit_seconds": round(fit_seconds, 6),
                  "predict_evaluate_seconds": round(time.perf_counter() - tick, 6),
                  "importance_fit_only": importances, "metrics": metrics,
                  "prediction_file": str(prediction.relative_to(output)), "prediction_sha256": sha256(prediction),
                  "model_file": str(model_path.relative_to(output)), "model_sha256": sha256(model_path)}
        attempt.update(status="complete", fit_seconds=result["fit_seconds"])
        write_json(output / "results" / f"{label}.json", result)
        print(f"P004 {label}: cols={len(columns)} NDCG20={metrics['overall']['user_macro']['ndcg@20']:.6f} fit={fit_seconds:.2f}s", flush=True)
        return result
    except Exception as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        del model
        gc.collect()


def verify_inputs(config):
    baseline_root = TASK / config["baseline_dir"]
    snapshot = TASK / "report/p003_snapshot/p003_equivalence_baseline/metrics.json"
    if sha256(baseline_root / "metrics.json") != sha256(snapshot):
        raise ValueError("P003 baseline run metadata changed")
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    if saved["status"] != "complete" or saved["build"]["shape"] != [184800, 92]:
        raise ValueError("P004 requires completed frozen baseline64")
    config_keys = ("boundaries_raw", "short_window_raw", "half_life_raw", "expected_queries", "data_dir", "queries_path")
    if any(config[key] != saved["config"][key] for key in config_keys):
        raise ValueError("P004 fixed cohort/config differs from P003")
    hashes = {}
    for relative, expected in saved["input_sha256"].items():
        if sha256(TASK / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
        hashes[relative] = expected
    for filename, field in (("features.npy", "matrix_file_sha256"), ("feature_inputs.jsonl", "feature_inputs_sha256"),
                            ("feature_schema.json", "feature_schema_sha256")):
        digest = sha256(baseline_root / filename)
        if digest != saved["build"][field]:
            raise ValueError(f"Frozen P003 baseline output changed: {filename}")
        hashes[f"{config['baseline_dir']}/{filename}"] = digest
    original = json.loads((TASK / "report/pilot_001_snapshot/metrics.json").read_text(encoding="utf-8"))
    if sha256(TASK / "src/pilot_features.py") != original["source_code_sha256"]["pilot_features.py"]:
        raise ValueError("P001 feature state changed")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "configs/p004_auto.json")
    parser.add_argument("--output", type=Path, default=TASK / "artifacts/p004_auto64")
    args = parser.parse_args()
    validate_frozen_config(args.config, TASK / "configs/P004_PROTOCOL.md")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["test_evaluation"] or config["scale256"] or config["feature_count"] != 206 or config["maximum_total_fits"] != 28:
        raise ValueError("Unsupported P004 development protocol")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("P004 requires a fresh output directory")
    for folder in (output, output / "models", output / "results", output / "predictions"):
        folder.mkdir(parents=True)
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()
    started = time.perf_counter()
    sources = ["run_auto_features.py", "auto_features.py", "auto_search.py", "pilot_features.py", "pilot_evaluation.py",
               "run_feature_benchmark.py", "run_retrieval.py", "run_pilot.py"]
    status = {"run_id": output.name, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_sha256": sha256(args.config), "protocol_sha256": sha256(TASK / "configs/P004_PROTOCOL.md"),
              "source_code_sha256": {name: sha256(TASK / "src" / name) for name in sources},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TASK, text=True).strip(),
              "command": sys.argv, "attempts": [], "build_progress": {}, "final_models": [], "final_test_evaluated": False}
    write_json(output / "run_status.json", status)
    try:
        monitor.check()
        status["input_sha256"] = verify_inputs(config)
        tick = time.perf_counter()
        queries = load_frozen_queries(TASK / config["queries_path"], config)
        offsets, sizes = offsets_sizes(queries)
        split = temporal_split(queries)
        split_record = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in split.items()}
        write_json(output / "split.json", split_record)
        status["query_counts"] = {key: len(value) for key, value in split.items() if key != "cutoff"}
        specs = generate_specs(config["windows_raw"], config["short_window_raw"])
        if len(specs) != config["feature_count"] or len(specs) > config["maximum_feature_columns"]:
            raise ValueError("DSL grammar exceeds registered shape/budget")
        write_json(output / "specs.json", specs)
        status["specs_sha256"] = sha256(output / "specs.json")
        events = load_development_events(TASK / config["data_dir"] / "events.parquet", config["boundaries_raw"]["validation_end"])
        artists = load_mapping(TASK / config["data_dir"] / "artist_mapping.parquet", "artist_id")
        albums = load_mapping(TASK / config["data_dir"] / "album_mapping.parquet", "album_id")
        status["load_seconds"] = round(time.perf_counter() - tick, 6)
        def automatic_engine(artist_map, album_map, short_window, half_life):
            return AutoFeatureEngine(artist_map, album_map, short_window, half_life, specs=specs)
        status["build"] = feature_build(automatic_engine, events, queries, artists, albums, config, output,
                                         monitor, progress=status["build_progress"])
        del events, artists, albums
        gc.collect()
        matrix = np.load(output / "features.npy", mmap_mode="r")
        baseline = np.load(TASK / config["baseline_dir"] / "features.npy", mmap_mode="r")
        schema = json.loads((TASK / config["baseline_dir"] / "feature_schema.json").read_text(encoding="utf-8"))
        status["a1_manual_equivalence"] = assert_manual_equivalence(matrix, baseline, specs, schema, output / "a1_equivalence.json")
        if sha256(output / "feature_inputs.jsonl") != status["input_sha256"][f"{config['baseline_dir']}/feature_inputs.jsonl"]:
            raise ValueError("P004 query/candidate order changed")
        del baseline
        gc.collect()
        fit_rows = candidate_rows(split["inner_fit"], offsets, sizes)
        winners = {}
        pruning = {}
        inner_results = []
        selection_started = time.perf_counter()
        for family in ("A1", "A2"):
            candidates = [i for i, spec in enumerate(specs) if family == "A2" or spec["family"] == "A1"]
            available, removed = prune_columns(matrix, fit_rows, candidates)
            pruning[family] = {"available": available, "removed": removed}
            write_json(output / "pruning.json", pruning)
            full = fit_and_predict(f"inner_{family}_full", available, split["inner_fit"], split["inner_selection"], matrix,
                                   queries, offsets, sizes, specs, config, output, monitor, status, "inner")
            results = [full]
            importances = {column: importance for column, importance in zip(available, full["importance_fit_only"])}
            proposals = select_trial_columns(available, specs, importances)
            if len(proposals) > config["maximum_inner_fits_per_family"]:
                raise ValueError("Family search budget exceeded")
            for name, columns in proposals.items():
                if columns == available:
                    continue
                results.append(fit_and_predict(f"inner_{family}_{name}", columns, split["inner_fit"], split["inner_selection"],
                                              matrix, queries, offsets, sizes, specs, config, output, monitor, status, "inner"))
            choices = [{"label": row["label"], "columns": row["columns"], "spec_ids": row["spec_ids"],
                        "inner_score": row["metrics"]["overall"]["user_macro"]["ndcg@20"]} for row in results]
            winners[family] = dict(pick_best(choices))
            inner_results.extend(results)
            write_json(output / "inner_results.json", inner_results)
        best = dict(pick_best(list(winners.values())))
        manual = [i for i, spec in enumerate(specs) if spec["family"] == "A1"]
        union, union_removed = union_columns(matrix, fit_rows, manual, best["columns"])
        masks = final_masks(specs, winners["A1"]["columns"], winners["A2"]["columns"], union)
        frozen = {"winners": winners, "chosen_auto_by_inner": best, "final_masks": masks,
                  "union_removed": union_removed, "specs_sha256": status["specs_sha256"],
                  "external_predictions_started": False}
        write_json(output / "selection_frozen.json", frozen)
        frozen_hash = sha256(output / "selection_frozen.json")
        status["selection_frozen_sha256"] = frozen_hash
        status["selection_seconds"] = round(time.perf_counter() - selection_started, 6)
        write_json(output / "run_status.json", status)
        train_positions = [i for i, q in enumerate(queries) if q["split"] == "train"]
        status["external_evaluation_started"] = True
        final_cache = {}
        for label, columns in masks.items():
            if sha256(output / "selection_frozen.json") != frozen_hash:
                raise ValueError("Frozen selection changed during external evaluation")
            if not columns:
                status["final_models"].append({"label": label, "status": "skipped_empty_group_ablation"})
                continue
            signature = tuple(columns)
            if signature in final_cache:
                result = dict(final_cache[signature], label=label, reused_from=final_cache[signature]["label"])
                write_json(output / "results" / f"{label}.json", result)
            else:
                result = fit_and_predict(label, columns, train_positions, split["external_validation"], matrix, queries,
                                         offsets, sizes, specs, config, output, monitor, status, "final")
                final_cache[signature] = result
            status["final_models"].append(result)
            write_json(output / "run_status.json", status)
        status["fits_actual"] = len(status["attempts"])
        status["status"] = "complete"
        status["output_sha256"] = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*")
                                   if path.is_file() and path.name not in {"run_status.json", "metrics.json"}}
        monitor.check()
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        status["resources"] = monitor.finish()
        if status["resources"]["limit_exceeded"]:
            status.update(status="failed", error=status["resources"]["limit_exceeded"])
        write_json(output / "run_status.json", status)
        write_json(output / "metrics.json", status)
    if status["status"] != "complete":
        raise MemoryError(status["error"])
    print(f"P004 COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
