"""Retain compact P003 evidence while keeping matrices and datasets local."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

if __package__:
    from .run_pilot import sha256, write_json
else:
    from run_pilot import sha256, write_json

TASK = Path(__file__).resolve().parents[1]


def main() -> None:
    snapshot = TASK / "report/p003_snapshot"
    if snapshot.exists():
        raise FileExistsError("P003 compact evidence already exists; never overwrite it")
    run_ids = ("p003_profile", "p003_equivalence_baseline", "p003_equivalence_optimized", "p003_scale256")
    for run in run_ids:
        status = json.loads((TASK / "artifacts" / run / "metrics.json").read_text(encoding="utf-8"))
        if status["status"] not in ("complete", "failed"):
            raise ValueError(f"Unfinished run: {run}")
        if run != "p003_scale256" and status["status"] != "complete":
            raise ValueError(f"Required completed exact-value stage: {run}")
    snapshot.mkdir()
    copies = {"equivalence.json": TASK / "artifacts/p003_equivalence.json",
              "profile_recheck.json": TASK / "artifacts/p003_profile_recheck.json",
              "p003_features.json": TASK / "configs/p003_features.json",
              "P003_PROTOCOL.md": TASK / "configs/P003_PROTOCOL.md",
              "P003_REVIEW.md": TASK / "research/P003_REVIEW.md",
              "P003_RESULT_REVIEW.md": TASK / "research/P003_RESULT_REVIEW.md"}
    local_outputs = {}
    for run in run_ids:
        root = TASK / "artifacts" / run
        for name in ("metrics.json", "run_status.json", "feature_schema.json", "profile_summary.json", "data/metadata.json"):
            if (root / name).exists():
                copies[f"{run}/{name}"] = root / name
        for path in root.rglob("*"):
            if path.is_file():
                local_outputs[path.relative_to(TASK).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for relative, source in copies.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (snapshot / "README.md").write_text(
        "# Контрольная точка P003\n\n"
        "Сохранены точные копии метрик, протокола, gate и независимых проверок.\n"
        "Матрицы, запросы и Parquet остаются локально; их хеши записаны в manifest.\n"
        "Свежий clone содержит компактные доказательства, без исходных данных.\n"
        "Verifier проверяет все доступные файлы и полное равенство локальных матриц64.\n"
        "Failed scale-run, если есть, сохраняется именно как failed.\n\n"
        "Отчёт: [P003_FEATURE_PIPELINE.md](../P003_FEATURE_PIPELINE.md).\n",
        encoding="utf-8")
    evidence = {path.relative_to(snapshot).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted(snapshot.rglob("*")) if path.is_file()}
    write_json(snapshot / "checkpoint_manifest.json", {"experiment": "P003", "tracked_evidence": evidence,
                                                       "local_outputs": local_outputs})
    print(f"Saved {len(evidence)} compact evidence files; {len(local_outputs)} local file hashes")


if __name__ == "__main__":
    main()
