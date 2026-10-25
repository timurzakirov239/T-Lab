"""Generate P004 tables, paired comparisons and report from saved predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_contributions(query, scores, k=20):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), query["candidate_items"][i]))
    ranked = [query["candidate_items"][i] for i in order]
    rank = ranked.index(query["target_item"]) + 1 if query["target_item"] in ranked else None
    hit = rank is not None and rank <= k
    return {"rank": rank, "recall20": float(hit), "ndcg20": 1 / math.log2(rank + 1) if hit else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=TASK / "artifacts/p004_auto64")
    args = parser.parse_args()
    root = args.run.resolve()
    run = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if run["status"] != "complete" or run["final_test_evaluated"]:
        raise ValueError("Report requires completed development P004")
    models = [model for model in run["final_models"] if "metrics" in model]
    frozen = json.loads((root / "selection_frozen.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    specs = json.loads((root / "specs.json").read_text(encoding="utf-8"))
    pruning = json.loads((root / "pruning.json").read_text(encoding="utf-8"))
    queries = [json.loads(line) for line in (TASK / run["config"]["queries_path"]).read_text(encoding="utf-8").splitlines()]
    by_id = {q["query_id"]: q for q in queries}
    rows = []
    for model in models:
        metric = model["metrics"]
        rows.append({"model": model["label"], "feature_count": model["feature_count"],
                     "ndcg20_user_macro": metric["overall"]["user_macro"]["ndcg@20"],
                     "recall20_user_macro": metric["overall"]["user_macro"]["recall@20"],
                     "ndcg10_user_macro": metric["overall"]["user_macro"]["ndcg@10"],
                     "recall10_user_macro": metric["overall"]["user_macro"]["recall@10"],
                     "ndcg20_repeat": metric["strata"]["repeat"]["user_macro"]["ndcg@20"],
                     "ndcg20_new": metric["strata"]["new"]["user_macro"]["ndcg@20"],
                     "fit_seconds": model["fit_seconds"], "reused_from": model.get("reused_from", "")})
    write_csv(TASK / "report/p004_metrics.csv", rows)
    core = {row["model"]: row for row in rows}
    contributions = {}
    for model in models:
        if model["label"] not in ("M_UIDC", "M_I", "Auto_A1", "Auto_A2", "Union"):
            continue
        with np.load(root / model["prediction_file"]) as pred:
            contributions[model["label"]] = {int(qid): query_contributions(by_id[int(qid)], pred["scores"][pred["offsets"][i]:pred["offsets"][i + 1]])
                                            for i, qid in enumerate(pred["query_ids"])}
    paired, user_rows = [], []
    for label in ("Auto_A1", "Auto_A2", "Union"):
        by_user = defaultdict(list)
        for qid, automatic in contributions[label].items():
            base = contributions["M_UIDC"][qid]
            query = by_id[qid]
            row = {"model": label, "query_id": qid, "user_id": query["user_id"],
                   "target_is_repeat": query["target_is_repeat"], "target_known_before": query["target_known_before"],
                   "manual_rank": base["rank"], "auto_rank": automatic["rank"],
                   "ndcg20_delta": automatic["ndcg20"] - base["ndcg20"],
                   "recall20_delta": automatic["recall20"] - base["recall20"]}
            paired.append(row)
            by_user[query["user_id"]].append(row)
        for user, values in sorted(by_user.items()):
            user_rows.append({"model": label, "user_id": user, "queries": len(values),
                              "ndcg20_delta": sum(v["ndcg20_delta"] for v in values) / len(values),
                              "recall20_delta": sum(v["recall20_delta"] for v in values) / len(values)})
    write_csv(TASK / "report/p004_paired_queries.csv", paired)
    write_csv(TASK / "report/p004_paired_users.csv", user_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "T-Lab-P004"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    labels = ["M_I", "M_UIDC", "Auto_A1", "Auto_A2", "Union"]
    for axis, key, title in zip(axes, ("ndcg20_user_macro", "recall20_user_macro"), ("NDCG@20 · user-macro", "Recall@20 · user-macro")):
        values = [core[label][key] for label in labels]
        bars = axis.bar(["Manual I", "Manual UIDC", "Auto A1", "Auto A2", "Union"], values,
                        color=["#718596", "#93a4b2", "#26799b", "#dc8a3e", "#43856e"])
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.22)
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(TASK / "report/p004_comparison.png", dpi=170)
    fig.savefig(TASK / "report/p004_comparison.svg", metadata={"Date": None})
    svg = TASK / "report/p004_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    overall = next(m for m in models if m["label"] == "M_UIDC")["metrics"]
    lines = ["# P004 — автоматическое построение и отбор признаков", "",
             "Ограниченный поиск завершён на фиксированной когорте64. Ниже измеренное development-сравнение; итоговый test не оценивался.", "",
             "![Сравнение P004](p004_comparison.png)", "",
             "## Условия и метод", "",
             "Те же 1 848 запросов P001 и прежние K=100 кандидатов; параметры retrieval P002 сюда не переносились. "
             "Цель — следующий зарегистрированный listen; события доступны строго раньше query. Mapping — фиксированный snapshot с неизвестной временной доступностью. "
             "Полнота Yambda-50M относительно5B не доказана. Внешняя validation уже использовалась ранее: это повторное development-сравнение, не независимый test.", "",
             f"Сгенерировано {len(specs)} выражений: 66 A1 повторяют семантику ручного UIDC, 140 A2 добавляют окна, сглаженные доли и тренды. "
             "A1 совпал с ручной матрицей побитно во всех 12 196 800 ячейках. Грамматика задана человеком; программно выполняются её развёртка и отбор.", "",
             f"Inner cutoff={split['cutoff']} RAW: {len(split['inner_fit'])} fit queries и {len(split['inner_selection'])} selection queries. "
             f"Внешняя оценка — {overall['overall']['query_count']} запросов от {overall['overall']['user_count']} пользователей. "
             "Константы/дубли, importance и выбор поднабора используют только inner fit/selection. Все внешние masks сохранены до внешних predictions.", "",
             "A1 ищет в тех же определениях, что Manual; A2 — во всех206 выражениях. "
             "После full fit проверяются top8/16/32/64 и balanced32 с пропуском одинаковых masks. Primary score — inner user-macro NDCG@20; "
             "ничьи: меньше признаков, затем IDs. Все модели CatBoost120/depth5/lr.07, два потока, один seed. "
             "Auto использует дополнительный поисковый бюджет; фиксированный Manual не получает равного числа попыток.", "",
             "## Отбор до внешней оценки", ""]
    for family in ("A1", "A2"):
        winner = frozen["winners"][family]
        lines.append(f"- {family}: после pruning {len(pruning[family]['available'])} колонок; выбран `{winner['label']}`, "
                     f"{len(winner['columns'])} признаков, inner NDCG@20={winner['inner_score']:.6f}.")
    lines += ["", f"Для Union выбрана `{frozen['chosen_auto_by_inner']['label']}` по внутреннему критерию. Manual66 сохраняется целиком; "
              f"дополнительные точные дубли удалены на inner fit. Итог Union — {len(frozen['final_masks']['Union'])} признаков.", "",
              "A1 top64 совпал с full после pruning, balanced32 совпал с top32; A2 balanced32 также совпал с top32. "
              "Поэтому фактически выполнены четыре A1 и пять A2 inner fits, а не все12 разрешённых попыток.", "",
              "Выбранные A2 выражения (4 из8 являются новыми относительно ручного набора):", "",
              "| Expression ID | Группа | Операция | Окно RAW |", "| --- | --- | --- | --- |"]
    for column in frozen["winners"]["A2"]["columns"]:
        spec = specs[column]
        lines.append(f"| `{spec['feature_id']}` | {spec['group']} | {spec['op']} | {spec['window']} |")
    lines += ["",
              "## Внешние результаты", "", "| Набор | Признаков | NDCG@20 user-macro | Recall@20 user-macro |",
              "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['feature_count']} | {row['ndcg20_user_macro']:.6f} | {row['recall20_user_macro']:.6f} |")
    lines += ["", f"Все варианты имеют те же retrieval-пропуски: цель найдена в {overall['counts']['targets_in_candidates']}/{overall['counts']['queries']} запросов. "
              f"Repeat/new знаменатели: {overall['strata']['repeat']['query_count']}/{overall['strata']['new']['query_count']}. "
              "Остальные кандидаты считаются unmarked, а не подтверждёнными отрицательными реакциями. Обучение условно на train-запросах с найденной целью; "
              "на evaluation все пропуски сохраняются с нулевым качеством.", "",
              "## Парные различия относительно Manual UIDC", ""]
    for label in ("Auto_A1", "Auto_A2", "Union"):
        values = [row for row in user_rows if row["model"] == label]
        delta = sum(v["ndcg20_delta"] for v in values) / len(values)
        signs = [sum(v["ndcg20_delta"] > 0 for v in values), sum(v["ndcg20_delta"] < 0 for v in values), sum(v["ndcg20_delta"] == 0 for v in values)]
        lines.append(f"- {label}: средняя user-level разность NDCG@20={delta:+.6f}; "
                     f"пользователей с ростом/снижением/равенством: {signs[0]}/{signs[1]}/{signs[2]}.")
    lines += ["", "Эти различия не доказывают устойчивое превосходство: малая когорта, один seed, повторная development-оценка и ограниченный retrieval. "
              "Manual I остаётся дополнительным фиксированным сильным контролем; primary comparator — Manual UIDC. "
              "Различие A2 относительно A1 смешивает расширение языка с отбором; вклад отдельных операций отдельно не установлен.", "",
              f"Относительно Manual UIDC: AutoA1 NDCG@20 выше на {(core['Auto_A1']['ndcg20_user_macro']/core['M_UIDC']['ndcg20_user_macro']-1)*100:.2f}% "
              f"при снижении Recall@20 на {core['M_UIDC']['recall20_user_macro']-core['Auto_A1']['recall20_user_macro']:.6f}; "
              f"AutoA2 NDCG@20 выше на {(core['Auto_A2']['ndcg20_user_macro']/core['M_UIDC']['ndcg20_user_macro']-1)*100:.2f}%. "
              "Union равен Manual: выбранный внутри AutoA1 не добавляет новых определений. Расширенный язык A2 в этом опыте "
              "не превзошёл A1 по основному NDCG, хотя дал близкий Recall. Это сохраняем как ограничение, а не объявляем общим выигрышем автоматизации.", "",
              "## Стоимость и проверки", "",
              f"Фактических fits: {run['fits_actual']} из лимита28 "
              f"({sum(a['stage']=='inner' for a in run['attempts'])} inner, {sum(a['stage']=='final' for a in run['attempts'])} final). "
              "Одинаковые final masks переиспользуют модель и predictions; их fit time в таблице относится к исходному fit и не суммируется повторно.", "",
              f"Полный runner {run['elapsed_seconds']:.3f} с; feature replay {run['build']['replay_seconds']:.3f} с; "
              f"отбор {run['selection_seconds']:.3f} с; peak RSS {run['resources']['peak_process_rss_mb']:.2f} МиБ, "
              f"минимум свободной системной памяти {run['resources']['minimum_available_system_mb']:.2f} МиБ. "
              f"Матрица184800×206 float32: {run['build']['matrix_values_bytes']/2**20:.3f} МиБ значений. "
              "Это стоимость всего метода на64; успешный scale256 этим не подтверждается.", "",
              "Независимый verifier пересчитал метрики из scores для всех25 результатов: 9 inner и16 final, "
              "включая переиспользованные masks. Все overall/strata метрики совпали с точностью1e-12; "
              "проверены74 output hashes и17 source/input/config/protocol hashes. Синтетический набор —75 тестов. "
              "Компактная таблица рангов позволяет повторно пересчитать метрики после clone без датасетов и моделей: "
              "отдельная копия с41 файлом evidence и двумя verifier-скриптами прошла пересчёт25результатов "
              "по11 638 rank rows. Это проверка сохранённых рангов, не повтор обучения или feature replay.", "",
              "[Протокол до запуска](../configs/P004_PROTOCOL.md), [конфигурация](../configs/p004_auto.json), "
              "[preflight](../research/P004_REVIEW.md), [независимая проверка](../research/P004_RESULT_REVIEW.md), "
              "[компактные доказательства](p004_snapshot/README.md). "
              "[CSV метрик](p004_metrics.csv), [парные query differences](p004_paired_queries.csv), "
              "[парные user differences](p004_paired_users.csv).", "",
              "Проверка компактных доказательств: `.\\.venv\\Scripts\\python.exe task_01_auto_features/src/verify_auto_snapshot.py`. "
              "Локальный verifier scores: `verify_auto_checkpoint.py --output artifacts/p004_auto64/verification_RECHECK.json`; "
              "исходный verification output не перезаписывается. "
              "Повторный запуск обучения требует нового `--output`; результаты P001–P004 не перезаписываются. "
              "Источники не отмечены проверенными человеком без подтверждения пользователя. Работа2 ещё не начата.", ""]
    (TASK / "report/P004_AUTO_FEATURES.md").write_text("\n".join(lines), encoding="utf-8")
    print("P004 report, metrics, paired contributions and plot saved")


if __name__ == "__main__":
    main()
