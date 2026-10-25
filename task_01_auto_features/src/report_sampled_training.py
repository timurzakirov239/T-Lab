"""Create and portably verify the compact P008 sampled-training report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from expanded_metrics import paired_user_delta, summarize_ranks

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK / "artifacts/p008_sampled_training"
REPORT = TASK / "report"
SNAPSHOT = REPORT / "p008_snapshot"
COHORTS = ("p001", "p005_20261026", "p005_20261027")
LABELS = (
    "Baseline_UIDC", "Baseline_A1", "Uniform_UIDC", "Uniform_A1",
    "HybridHit_UIDC", "HybridHit_A1", "HybridWarm_UIDC", "HybridWarm_A1",
    "Popularity",
)
MASKS = ("UIDC", "A1")
ARMS = ("Uniform", "HybridHit", "HybridWarm")
MODES = ("K100", "Full")
CSV_FIELDS = (
    "cohort", "mode", "model", "query_id", "user_id", "timestamp",
    "candidate_count", "target_rank", "target_is_repeat", "target_known_before",
)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rank_rows(path: Path, mode: str, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.update(mode=mode, model=label)
            rows.append(row)
    return rows


def contrast_specs() -> dict[str, dict[str, str]]:
    """Return the 29 frozen paired comparisons, with direction mode_b-mode_a."""
    result: dict[str, dict[str, str]] = {}
    for mode in MODES:
        for arm in ARMS:
            for mask in MASKS:
                variant = f"{arm}_{mask}"
                baseline = f"Baseline_{mask}"
                key = f"{mode}:{variant}-minus-{baseline}"
                result[key] = {"mode_a": mode, "model_a": baseline,
                               "mode_b": mode, "model_b": variant,
                               "contrast_family": "arm_vs_baseline"}
        for mask in MASKS:
            for left, right, family in (
                (f"Uniform_{mask}", f"HybridHit_{mask}", "HybridHit_minus_Uniform"),
                (f"HybridHit_{mask}", f"HybridWarm_{mask}", "HybridWarm_minus_HybridHit"),
            ):
                key = f"{mode}:{right}-minus-{left}"
                result[key] = {"mode_a": mode, "model_a": left,
                               "mode_b": mode, "model_b": right,
                               "contrast_family": family}
    for label in LABELS:
        key = f"Full-minus-K100:{label}"
        result[key] = {"mode_a": "K100", "model_a": label,
                       "mode_b": "Full", "model_b": label,
                       "contrast_family": "catalog_mode"}
    if len(result) != 29:
        raise AssertionError(f"expected 29 paired contrasts, got {len(result)}")
    return result


def _group_rows(rows: Iterable[Mapping[str, Any]], *, mode: str | None = None,
                model: str | None = None, cohort: str | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in rows
            if (mode is None or row["mode"] == mode)
            and (model is None or row["model"] == model)
            and (cohort is None or row["cohort"] == cohort)]


def analyze_rows(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        metrics[mode] = {label: summarize_ranks(_group_rows(rows, mode=mode, model=label))
                         for label in LABELS}
    cohort_metrics: dict[str, Any] = {}
    for cohort in COHORTS:
        cohort_metrics[cohort] = {
            mode: {label: summarize_ranks(_group_rows(rows, mode=mode, model=label, cohort=cohort))
                   for label in LABELS}
            for mode in MODES
        }
    seed = int(config["bootstrap"]["seed"])
    resamples = int(config["bootstrap"]["resamples"])
    paired = {}
    for key, spec in contrast_specs().items():
        rows_a = _group_rows(rows, mode=spec["mode_a"], model=spec["model_a"])
        rows_b = _group_rows(rows, mode=spec["mode_b"], model=spec["model_b"])
        paired[key] = {
            **spec,
            "pooled": paired_user_delta(rows_a, rows_b, seed=seed, resamples=resamples,
                                         allow_catalog_change=(spec["mode_a"] != spec["mode_b"])),
            "cohorts": {
                cohort: paired_user_delta(
                    _group_rows(rows, mode=spec["mode_a"], model=spec["model_a"], cohort=cohort),
                    _group_rows(rows, mode=spec["mode_b"], model=spec["model_b"], cohort=cohort),
                    seed=seed, resamples=resamples,
                    allow_catalog_change=(spec["mode_a"] != spec["mode_b"]),
                ) for cohort in COHORTS
            },
        }
    return {
        "experiment": "P008", "status": "exploratory_complete",
        "metrics": metrics, "cohort_metrics": cohort_metrics,
        "paired_contrasts": paired,
        "scope": {"queries": metrics["K100"][LABELS[0]]["overall"]["query_count"],
                  "users": metrics["K100"][LABELS[0]]["overall"]["user_count"],
                  "cohorts": list(COHORTS)},
        "primary": "Full:HybridHit_UIDC-minus-Baseline_UIDC",
        "final_test_evaluated": False,
        "interpretation": "exploratory development only; descriptive paired-user bootstrap; no winner claim",
    }


def _normalize_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for field in ("query_id", "user_id", "timestamp", "candidate_count"):
                row[field] = int(row[field])
            row["target_rank"] = int(row["target_rank"]) if row["target_rank"] else None
            for field in ("target_is_repeat", "target_known_before"):
                if row[field] not in ("True", "False"):
                    raise ValueError(f"invalid boolean in {field}")
                row[field] = row[field] == "True"
            rows.append(row)
    return rows


def verify_snapshot(snapshot: Path = SNAPSHOT) -> dict[str, Any]:
    manifest = read(snapshot / "checkpoint_manifest.json")
    for relative, expected in manifest.items():
        if sha256(snapshot / relative) != expected:
            raise AssertionError(f"checkpoint SHA drift: {relative}")
    verification = read(snapshot / "verification_independent.json")
    if verification.get("status") != "verified":
        raise AssertionError("snapshot lacks a verified P008 independent-check status")
    freeze = read(snapshot / "all_fits_frozen.json")
    freeze_sha = sha256(snapshot / "all_fits_frozen.json")
    if (freeze.get("experiment") != "P008"
            or freeze.get("frozen_before_evaluation") is not True
            or set(freeze.get("cohort_fit_status_sha256", {})) != set(COHORTS)):
        raise AssertionError("snapshot lacks the shared all-fits-before-evaluation ledger")
    for cohort in COHORTS:
        folder = snapshot / cohort
        if freeze["cohort_fit_status_sha256"].get(cohort) != sha256(folder / "status_fit.json"):
            raise AssertionError(f"all-fits ledger does not match frozen fit status: {cohort}")
        for stage in ("gate", "full"):
            status = read(folder / f"status_{stage}.json")
            if status.get("all_fits_frozen_sha256") != freeze_sha:
                raise AssertionError(f"{stage} status does not reference the shared all-fits freeze: {cohort}")
    rows = _normalize_snapshot_rows(snapshot / "ranks.csv")
    expected_n = 1531 * len(LABELS) * len(MODES)
    if len(rows) != expected_n:
        raise AssertionError(f"rank row count mismatch: {len(rows)} != {expected_n}")
    row_groups = {(mode, label): _group_rows(rows, mode=mode, model=label)
                  for mode in MODES for label in LABELS}
    if any(len(group) != 1531 for group in row_groups.values()):
        raise AssertionError("every model/mode must contain all 1531 queries")
    expected_analysis = analyze_rows(rows, read(snapshot / "p008_sampled.json"))
    stored = read(snapshot / "analysis.json")
    if expected_analysis != stored:
        raise AssertionError("portable metric/contrast recomputation differs from analysis.json")
    counts = {cohort: len(_group_rows(rows, mode="K100", model=LABELS[0], cohort=cohort))
              for cohort in COHORTS}
    if counts != {"p001": 499, "p005_20261026": 514, "p005_20261027": 518}:
        raise AssertionError(f"cohort validation denominators differ: {counts}")
    return {"status": "PASS", "tracked_files": len(manifest), "rank_rows": len(rows),
            "pooled_metric_groups": len(MODES) * len(LABELS),
            "cohort_metric_groups": len(COHORTS) * len(MODES) * len(LABELS),
            "paired_contrasts": len(contrast_specs()), "independent_metric_implementation": "NumPy + expanded_metrics; no model/event scan"}


def _status_path(cohort_dir: Path, stage: str) -> Path:
    return cohort_dir / stage / "run_status.json"


def _require_complete(cohort_dir: Path, stage: str, config_sha: str, protocol_sha: str) -> dict[str, Any]:
    status = read(_status_path(cohort_dir, stage))
    if status.get("status") != "complete" or status.get("stage") != stage:
        raise ValueError(f"P008 {stage} stage is not complete: {cohort_dir.name}")
    if status.get("final_test_evaluated") is not False:
        raise ValueError(f"P008 {stage} unexpectedly reports final-test evaluation")
    if status.get("config_sha256") != config_sha or status.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"P008 {stage} config/protocol hash drift")
    return status


def _stage_metrics(full_dir: Path, mode: str) -> dict[str, Any]:
    path = full_dir / ("metrics100.json" if mode == "K100" else "metrics_full.json")
    data = read(path)
    if set(data) != set(LABELS):
        raise ValueError(f"unexpected P008 metrics labels at {path}")
    return data


def _load_cohort_rows(cohort_dir: Path) -> list[dict[str, Any]]:
    full_dir = cohort_dir / "full"
    result = []
    for mode, folder_name in (("K100", "ranks100"), ("Full", "ranksfull")):
        for label in LABELS:
            path = full_dir / folder_name / f"{label}.jsonl"
            rows = load_rank_rows(path, mode, label)
            if len(rows) != (499 if cohort_dir.name == "p001" else 514 if cohort_dir.name.endswith("20261026") else 518):
                raise ValueError(f"incomplete rank file: {path}")
            result.extend(rows)
    return result


def create_snapshot() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if SNAPSHOT.exists():
        raise FileExistsError(f"refusing to overwrite {SNAPSHOT}")
    config_path = TASK / "configs/p008_sampled.json"
    protocol_path = TASK / "configs/P008_PROTOCOL.md"
    review_path = TASK / "research/P008_PROTOCOL_REVIEW.md"
    independent_path = ROOT / "verification_independent.json"
    config, verification = read(config_path), read(independent_path)
    if verification.get("status") != "verified":
        raise ValueError("P008 requires verification_independent.json status=verified")
    if verification.get("final_test_evaluated", False):
        raise ValueError("P008 independent verifier must confirm no final test evaluation")
    config_sha, protocol_sha = sha256(config_path), sha256(protocol_path)
    all_fits_path = ROOT / "all_fits_frozen.json"
    all_fits = read(all_fits_path)
    all_fits_sha = sha256(all_fits_path)
    if (all_fits.get("experiment") != "P008"
            or all_fits.get("frozen_before_evaluation") is not True
            or all_fits.get("config_sha256") != config_sha
            or all_fits.get("protocol_sha256") != protocol_sha
            or set(all_fits.get("cohort_fit_status_sha256", {})) != set(COHORTS)):
        raise ValueError("P008 all_fits_frozen.json is invalid or does not freeze all cohorts before evaluation")
    statuses: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        source = ROOT / cohort
        stage_status = {stage: _require_complete(source, stage, config_sha, protocol_sha)
                        for stage in ("prepare", "fit", "gate", "full")}
        for stage in ("gate", "full"):
            if stage_status[stage].get("all_fits_frozen_sha256") != all_fits_sha:
                raise ValueError(f"P008 {stage} status does not reference the shared all-fits freeze in {cohort}")
        if all_fits["cohort_fit_status_sha256"].get(cohort) != sha256(_status_path(source, "fit")):
            raise ValueError(f"P008 all-fits ledger does not match fit status in {cohort}")
        if stage_status["fit"].get("progress", {}).get("fits_completed") != 6:
            raise ValueError(f"P008 must have six new fits in {cohort}")
        if stage_status["full"].get("reference_parity_exact") is not True:
            raise ValueError(f"P008 lacks exact P007 baseline/Popularity parity in {cohort}")
        if stage_status["gate"].get("exact_gate", {}).get("identical_float32_bytes") is not True:
            raise ValueError(f"P008 exact validation gate did not pass in {cohort}")
        if stage_status["full"].get("exact_gate", {}).get("identical_float32_bytes") is not True:
            raise ValueError(f"P008 Full exact gate did not pass in {cohort}")
        for mode in MODES:
            cohort_metrics = _stage_metrics(source / "full", mode)
            for label in LABELS:
                result = cohort_metrics[label]["overall"]
                rows_expected = int(stage_status["full"]["progress"]["queries_completed"])
                if result["query_count"] != rows_expected:
                    raise ValueError(f"P008 {mode} denominator mismatch in {cohort}/{label}")
        statuses[cohort] = stage_status
        all_rows.extend(_load_cohort_rows(source))
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    files = [
        (protocol_path, SNAPSHOT / "P008_PROTOCOL.md"),
        (config_path, SNAPSHOT / "p008_sampled.json"),
        (review_path, SNAPSHOT / "P008_PROTOCOL_REVIEW.md"),
        (independent_path, SNAPSHOT / "verification_independent.json"),
        (all_fits_path, SNAPSHOT / "all_fits_frozen.json"),
    ]
    for src, dst in files:
        shutil.copyfile(src, dst)
    for cohort in COHORTS:
        source = ROOT / cohort
        destination = SNAPSHOT / cohort
        destination.mkdir()
        for stage in ("prepare", "fit", "gate", "full"):
            shutil.copyfile(_status_path(source, stage), destination / f"status_{stage}.json")
        shutil.copyfile(source / "fit/training_frozen.json", destination / "training_frozen.json")
        shutil.copyfile(source / "full/metrics100.json", destination / "metrics_K100.json")
        shutil.copyfile(source / "full/metrics_full.json", destination / "metrics_Full.json")
    with (SNAPSHOT / "ranks.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    analysis = analyze_rows(all_rows, config)
    write_json(SNAPSHOT / "analysis.json", analysis)
    readme = (
        "# P008 — compact rank checkpoint\n\n"
        "Three existing disjoint development cohorts; all 1531 outer-validation queries and nine labels in K100 and Full modes. "
        "P008 models use fixed UIDC66 or frozen cohort-specific P007 A1 masks. Full reuses the same trained model and scores the "
        "prior-observed cohort catalog. No P006 test replay.\n\n"
        "The CSV contains one-based target ranks and denominators. Training records, sampled feature matrices, teacher pool payloads, "
        "models and event histories are omitted. The copied stage statuses and training_frozen files retain hashes, eligibility, "
        "model provenance and cost. The shared all_fits_frozen.json ledger and gate/full status hashes document that all 18 new fits "
        "were frozen before any validation scoring. Portable verification recomputes pooled/cohort rank metrics and all paired comparisons with NumPy "
        "and expanded_metrics only.\n\n"
        "Run `python task_01_auto_features/src/report_sampled_training.py --verify` with NumPy. Human source verification remains pending.\n"
    )
    (SNAPSHOT / "README.md").write_text(readme, encoding="utf-8")
    manifest = {p.relative_to(SNAPSHOT).as_posix(): sha256(p)
                for p in sorted(SNAPSHOT.rglob("*")) if p.is_file()}
    write_json(SNAPSHOT / "checkpoint_manifest.json", manifest)
    return analysis, statuses


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def generate_report(analysis: Mapping[str, Any], statuses: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    metrics = analysis["metrics"]
    primary = analysis["paired_contrasts"]["Full:HybridHit_UIDC-minus-Baseline_UIDC"]["pooled"]["bootstrap"]["ndcg20"]
    lines = [
        "# P008 — negative sampling и дополнительные warm training queries", "",
        "**Exploratory development-серия.** Повторно использованы все 1531 прежних validation queries в трёх непересекающихся когортах; "
        "объединённые метрики представлены по 158 пользователям с запросами. P006 test не открывался. Никакой результат этой серии "
        "не является независимым подтверждением или популяционным выводом.", "",
        "## Условия и методы", "",
        "Все новые arms используют неизменные UIDC66 и cohort-specific P007 A1 masks. Базовые P007 models переиспользованы. A1 masks "
        "сами заморожены ещё в P004/P005; после P007 A1 выбран дополнительным benchmark, поэтому это сравнение exploratory. Новый feature search и hyperparameter tuning не выполнялись. "
        "Primary — Full user-macro NDCG@20 для HybridHit_UIDC против Baseline_UIDC.", "",
        "Baseline — исходные K100 hit-query training blocks. UniformHit использует те же hit queries с 99 равномерными cohort-catalog "
        "negatives. HybridHit заменяет 49 из них teacher-hard negatives, сохраняя общие 50 uniform draws. HybridWarm добавляет train queries, "
        "чьи цели известны cohort-wide до query; shared hit-query blocks остаются идентичны HybridHit. Здесь warm/cold относится к каталогу всей "
        "когорты (`target_known_before`), а repeat — отдельно к user-level прошлым прослушиваниям. Новые пользователю товары могут входить в warm arm.", "",
        "Teacher — P007 M_UIDC, обученный на той же train выборке; hard mining in-sample и может наследовать её смещения. Он не видел "
        "validation/test labels. Все negatives являются unmarked объектами, а не доказанно нежелательными реакциями. Обучение условно на "
        "наличии target; это не training на всём каталоге. Одна и та же модель каждого arm оценивается в K100 и Full режимах.", "",
        "## Pooled rank metrics", "",
        "User-macro: сначала усредняются запросы пользователя, затем представленные пользователи; когорты не получают равный фиксированный вес.", "",
        "| Arm | K100 NDCG@20 | K100 Recall@20 | Full NDCG@20 | Full Recall@20 |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        k = metrics["K100"][label]["overall"]["user_macro"]
        f = metrics["Full"][label]["overall"]["user_macro"]
        lines.append(f"| {label} | {_fmt(k['ndcg@20'])} | {_fmt(k['recall@20'])} | {_fmt(f['ndcg@20'])} | {_fmt(f['recall@20'])} |")
    lines += [
        "", "## Primary comparison", "",
        f"HybridHit_UIDC − Baseline_UIDC в Full: ΔNDCG@20 = **{primary['mean']:+.6f}**, descriptive paired-user 95% CI "
        f"[{primary['ci_lower']:+.6f}; {primary['ci_upper']:+.6f}], N={primary['users']}. Это заранее заданный primary contrast; "
        "интервал условен на этих пользователях, training/query protocol и фиксированном teacher. Он не учитывает повторное использование "
        "validation или неопределённость cohort sampling.", "",
        "Все 29 pooled paired contrasts (каждый arm к соответствующему baseline, sampling/warm additions и K100→Full) перечислены в "
        "[analysis.json](p008_snapshot/analysis.json); там же сохранены cohort-specific metrics и paired estimates. CI описательные, без "
        "поправки за несколько сравнений. Нулевой интервал не доказывает эквивалентность.", "",
        "## Eligibility, catalog coverage и переходы", "",
        "| Cohort | Train queries | Hit in original K100 | Warm in cohort C(t) | Cold to cohort C(t) | Training queries (Uniform/HybridHit/HybridWarm) |", "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cohort in COHORTS:
        prep = statuses[cohort]["prepare"]
        fit_status = statuses[cohort]["fit"]
        counts = prep["query_counts"]
        sizes = [fit_status["models"][i]["training"]["queries"] for i in range(2, 8)]
        arm_sizes = [fit_status["models"][i]["training"]["queries"] for i in (2, 4, 6)]
        lines.append(f"| {cohort} | {counts['train']} | {counts['hit']} | {counts['warm']} | {counts['cold']} | "
                     f"{arm_sizes[0]} / {arm_sizes[1]} / {arm_sizes[2]} |")
    lines += [
        "", "| Mode | Targets present in catalog / queries | New-to-user targets present / queries |", "| --- | ---: | ---: |",
    ]
    for mode in MODES:
        summary = metrics[mode]["Baseline_UIDC"]
        overall, new = summary["overall"], summary["strata"]["new"]
        present = round(overall["candidate_recall"] * overall["query_count"])
        new_present = round(new["candidate_recall"] * new["query_count"])
        lines.append(f"| {mode} | {present} / {overall['query_count']} ({overall['candidate_recall']:.1%}) | "
                     f"{new_present} / {new['query_count']} |")
    lines += [
        "", "Для каждой модели парные разности Full−K100 сохранены в paired analysis; ранги отдельных целей находятся в ranks.csv. Топ-20 метрики считают один logged listen target, "
        "а не полезность всех unmarked кандидатов. Population/catalog coverage зависит только от каталога и одинаково для rankers внутри режима. "
        "Full оценивает те же замороженные модели на cohort-local C(t); это distribution shift по сравнению с K100 training.", "",
        "## Cohort-specific Full NDCG@20", "",
        "| Arm | P001 | Seed26 | Seed27 |", "| --- | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        values = [analysis["cohort_metrics"][cohort]["Full"][label]["overall"]["user_macro"]["ndcg@20"] for cohort in COHORTS]
        lines.append(f"| {label} | " + " | ".join(_fmt(value) for value in values) + " |")
    lines += ["", "## Разбор sampling и расширения train", "",
        "| Full contrast | Δ user-macro NDCG@20 | Descriptive 95% CI |",
        "| --- | ---: | --- |"]
    for family in ("HybridHit_minus_Uniform", "HybridWarm_minus_HybridHit"):
        for mask in MASKS:
            left = f"Uniform_{mask}" if family == "HybridHit_minus_Uniform" else f"HybridHit_{mask}"
            right = f"HybridHit_{mask}" if family == "HybridHit_minus_Uniform" else f"HybridWarm_{mask}"
            delta = analysis["paired_contrasts"][f"Full:{right}-minus-{left}"]["pooled"]["bootstrap"]["ndcg20"]
            lines.append(f"| {right} − {left} | {delta['mean']:+.6f} | [{delta['ci_lower']:+.6f}; {delta['ci_upper']:+.6f}] |")
    lines += ["",
        "Uniform уступил исходному baseline при обеих masks. HybridHit приблизился к baseline, но основной прирост не подтверждён: "
        "primary point estimate отрицателен, CI включает ноль. Более высокие point estimates HybridHit относительно Uniform не означают "
        "превосходства над baseline; для UIDC интервал этого вторичного сравнения тоже включает ноль.", "",
        "HybridWarm увеличил train с 1015 до 3644 запросов, сохранив sampling на общих запросах, но резко ухудшил итоговое ранжирование "
        "при обеих masks и во всех трёх когортах. Это наблюдаемый эффект конкретного расширения и фиксированного training recipe; "
        "он не доказывает вред дополнительных данных вообще. Состав запросов и их доли одновременно изменились. Причины — изменение "
        "сигналов релевантности, sampling bias, ложные negatives или настройки обучения — требуют отдельных проверок, а не объявляются установленными.", "",
        "| Full arm | Repeat NDCG@20 | Repeat Recall@20 | New-to-user NDCG@20 | New-to-user Recall@20 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for label in LABELS:
        strata = metrics["Full"][label]["strata"]
        repeat, new = strata["repeat"]["user_macro"], strata["new"]["user_macro"]
        lines.append(f"| {label} | {_fmt(repeat['ndcg@20'])} | {_fmt(repeat['recall@20'])} | {_fmt(new['ndcg@20'])} | {_fmt(new['recall@20'])} |")
    lines += ["",
        "Repeat: 978 запросов/146 пользователей; new-to-user: 553/147, из них 363 цели доступны в Full. "
        "Каждый столбец strata усреднён только по представленным в этой группе пользователям; их нельзя складывать для получения overall. "
        "Падение HybridWarm особенно заметно на повторах. Отдельные улучшения new-to-user не компенсируют overall падение. "
        "Coverage внутри каждого режима остаётся одинаковым для всех моделей: изменение обучения не меняло каталог.",
        "", "## Masks, fits и стоимость", "",
        "| Cohort | UIDC columns | P007 A1 columns | New fits | Prepare/mining s | Fit s | Gate s | Full s | Peak RSS max MiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total_fits = 0
    for cohort in COHORTS:
        stage = statuses[cohort]
        frozen = read(SNAPSHOT / cohort / "training_frozen.json")
        fit_records = [row for row in stage["fit"]["models"] if row["model_origin"] == "p008"]
        total_fits += len(fit_records)
        peak = max(float(stage[s].get("resources", {}).get("peak_process_rss_mb", 0)) for s in ("prepare", "fit", "gate", "full"))
        fit_s = sum(float(row.get("fit_seconds", 0)) for row in fit_records)
        lines.append(f"| {cohort} | {len(frozen['masks']['UIDC'])} | {len(frozen['masks']['A1'])} | {len(fit_records)} | "
                     f"{stage['prepare']['elapsed_seconds']:.2f} | {fit_s:.2f} | {stage['gate']['elapsed_seconds']:.2f} | "
                     f"{stage['full']['elapsed_seconds']:.2f} | {peak:.2f} |")
    lines += [
        "", f"Итого: **{total_fits} новых fits** (6 на когорту); baseline модели и teacher переиспользованы. Prepare/mining wall time включает "
        "фильтрацию/first-seen inventory, feature materialization и teacher-pool scoring, поэтому это стоимость этапа, а не отдельный "
        "чистый benchmark teacher inference. Все cohort statuses должны оставаться в пределах 1100 MiB RSS и 150 MiB reserve.", "",
        "Exact validation gate проверяет UIDC66 features на всех исходных K100 rows; Full scoring проверяет их в chunks2048 и требует parity "
        "с P007 для Baseline_UIDC, Baseline_A1 и Popularity. Независимая проверка и portable snapshot status — `verified`/`PASS`; teacher-pool "
        "scores повторно не рассчитаны независимым teacher inference, что отражено в protocol. Полные score vectors каталога не сохранялись: "
        "exact ranks за пределами top20 защищены hashes и проверками границ/вложенности, но независимо из всех scores не восстановлены.", "",
        "## Ограничения и вывод", "",
        "Все queries — ранее использованная development validation; A1 выбран дополнительным benchmark после P007, при неизменных ранних masks. Поэтому различия "
        "и intervals exploratory, не являются независимым test. Training имеет shortlist-derived baseline и условные positives; sampled unmarked "
        "items не доказанные negatives. Teacher in-sample, mapping time availability остаётся предположением, а cohort catalog не равен каталогу "
        "всего сервиса. P006 не открывался, работа 2 не начиналась.", "",
        "P008 не устанавливает победителя. Primary contrast и остальные эффекты следует использовать как ориентиры для следующего заранее "
        "зафиксированного эксперимента с независимым временным holdout либо иной схемой validation. Human source verification остаётся pending.", "",
        "## Материалы", "",
        "- [Протокол](../configs/P008_PROTOCOL.md), [pre-fit review](../research/P008_PROTOCOL_REVIEW.md).",
        "- [Компактный snapshot](p008_snapshot/README.md), [reproducibility](P008_REPRODUCIBILITY.md), [human review checklist](P008_HUMAN_REVIEW.md).",
        "- [Рецензия результатов](../research/P008_RESULT_REVIEW.md), [изолированная проверка](p008_isolated_verification.json), [сохранность 72 файлов P006](p008_frozen_input_verification.json).",
        "- Portable rank analysis содержит pooled и cohort metrics, а также все paired comparisons. Полные rank score vectors и sampled training payload не включены.",
    ]
    (REPORT / "P008_SAMPLED_TRAINING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("P008 comparison chart requires matplotlib and NumPy") from error
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, metric in zip(axes, ("ndcg@20", "recall@20")):
        y = np.arange(len(LABELS)); height = 0.36
        axis.barh(y - height / 2, [metrics["K100"][label]["overall"]["user_macro"][metric] for label in LABELS],
                  height, label="K100", color="#427b99")
        axis.barh(y + height / 2, [metrics["Full"][label]["overall"]["user_macro"][metric] for label in LABELS],
                  height, label="Full", color="#d38b4c")
        axis.set_yticks(y, LABELS); axis.invert_yaxis()
        axis.set_title(f"P008 development · user-macro {metric.upper()}")
        axis.grid(axis="x", alpha=0.2); axis.set_axisbelow(True); axis.legend(fontsize=8)
    fig.savefig(REPORT / "p008_comparison.png", dpi=160)
    fig.savefig(REPORT / "p008_comparison.svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify existing compact snapshot only")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_snapshot(), indent=2))
        return
    analysis, statuses = create_snapshot()
    config = read(SNAPSHOT / "p008_sampled.json")
    generate_report(analysis, statuses, config)
    verification = verify_snapshot()
    write_json(REPORT / "p008_portable_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
