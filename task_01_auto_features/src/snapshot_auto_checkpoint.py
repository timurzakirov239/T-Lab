"""Preserve compact P004 metrics and per-query ranks without large matrices."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

if __package__:
    from .report_auto_features import query_contributions
    from .run_pilot import sha256, write_json
else:
    from report_auto_features import query_contributions
    from run_pilot import sha256, write_json

import numpy as np

TASK = Path(__file__).resolve().parents[1]


def main():
    root = TASK / "artifacts/p004_auto64"
    snapshot = TASK / "report/p004_snapshot"
    if snapshot.exists():
        raise FileExistsError("P004 compact checkpoint already exists")
    status = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if status["status"] != "complete" or not (root / "verification_independent.json").exists():
        raise ValueError("Compact checkpoint requires completed and independently verified P004")
    snapshot.mkdir()
    for name in ("metrics.json", "run_status.json", "split.json", "specs.json", "pruning.json", "feature_schema.json",
                 "a1_equivalence.json", "selection_frozen.json", "inner_results.json", "verification_independent.json"):
        shutil.copyfile(root / name, snapshot / name)
    shutil.copytree(root / "results", snapshot / "results")
    for folder, name in (("configs", "p004_auto.json"), ("configs", "P004_PROTOCOL.md"),
                         ("research", "P004_REVIEW.md"), ("research", "P004_RESULT_REVIEW.md")):
        shutil.copyfile(TASK / folder / name, snapshot / name)
    queries = {q["query_id"]: q for q in
               (json.loads(line) for line in (TASK / status["config"]["queries_path"]).read_text(encoding="utf-8").splitlines())}
    with (snapshot / "query_ranks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "query_id", "user_id", "split", "target_is_repeat", "target_known_before", "target_rank"])
        writer.writeheader()
        for path in sorted((root / "results").glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            with np.load(root / result["prediction_file"]) as pred:
                for i, qid in enumerate(pred["query_ids"]):
                    query = queries[int(qid)]
                    rank = query_contributions(query, pred["scores"][pred["offsets"][i]:pred["offsets"][i + 1]])["rank"]
                    writer.writerow({"model": result["label"], "query_id": int(qid), "user_id": query["user_id"],
                                     "split": query["split"], "target_is_repeat": query["target_is_repeat"],
                                     "target_known_before": query["target_known_before"], "target_rank": rank})
    (snapshot / "README.md").write_text(
        "# Контрольная точка P004\n\n"
        "Сохранены точные копии метрик, split, specs, выбора, reviews и результатов.\n"
        "query_ranks.csv позволяет независимо пересчитать inner/final метрики после clone.\n"
        "Rank table производна от сохранённых scores; верификация самих scores была\n"
        "выполнена локально, её evidence и хеши приложены. Это не повтор обучения.\n"
        "Матрицы, модели, predictions и исходные данные остаются локально;\n"
        "их хеши записаны в оригинальном metrics.json/output_sha256.\n\n"
        "Проверка: python task_01_auto_features/src/verify_auto_snapshot.py\n\n"
        "[Отчёт P004](../P004_AUTO_FEATURES.md). Test не оценивался.\n",
        encoding="utf-8")
    evidence = {path.relative_to(snapshot).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted(snapshot.rglob("*")) if path.is_file()}
    write_json(snapshot / "checkpoint_manifest.json", {"experiment": "P004", "tracked_evidence": evidence,
                                                       "local_output_sha256": status["output_sha256"]})
    print(f"P004 compact evidence: {len(evidence)} files")


if __name__ == "__main__":
    main()
