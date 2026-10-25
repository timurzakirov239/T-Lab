"""Produce the P002 report from frozen or locally saved aggregate results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK = Path(__file__).resolve().parents[1]
NAMES = {"baseline": "P001 baseline", "repeat_heavy": "Repeat-heavy", "affinity": "Affinity32", "rank_fusion": "RRF"}


def main() -> None:
    out = TASK / "report"
    frozen = out / "p002_snapshot/metrics.json"
    path = frozen if frozen.exists() else TASK / "artifacts/retrieval_002/metrics.json"
    raw = path.read_bytes()
    run = json.loads(raw)
    assert run["status"] == "complete" and not run["final_test_evaluated"] and not run["model_training"]
    strategies = run["strategies"]
    rows = []
    for key, info in strategies.items():
        value = info["validation"]
        rows.append({"strategy": key, "hits": value["hits"], "queries": value["queries"],
                     "users": value["users"], "candidate_recall_user_macro": value["candidate_recall_user_macro"],
                     "candidate_recall_query_micro": value["candidate_recall_query_micro"],
                     "repeat_hits": info["strata"]["repeat"]["hits"], "new_hits": info["strata"]["new"]["hits"],
                     "cold_hits": info["strata"]["cold"]["hits"]})
    with (out / "p002_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "T-Lab-P002"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    labels = [NAMES[row["strategy"]] for row in rows]
    colors = ["#9baebc", "#d87524", "#2274a5", "#738d6d"]
    macro = [100 * row["candidate_recall_user_macro"] for row in rows]
    new = [100 * strategies[row["strategy"]]["strata"]["new"]["candidate_recall_query_micro"] for row in rows]
    for axis, values, title in zip(axes, (macro, new),
                                  ("Все цели · user-macro", "Новые пользователю · query-micro")):
        bars = axis.bar(np.arange(4), values, color=colors, width=.62)
        axis.bar_label(bars, fmt="%.2f%%", padding=4)
        axis.set_xticks(np.arange(4), labels, rotation=18)
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_ylabel("Candidate recall@100, %")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("P002 · одинаковые запросы · K=100\n499 validation-запросов / 52 пользователя; новых целей 165", fontsize=12)
    fig.savefig(out / "p002_comparison.png", dpi=180)
    svg = out / "p002_comparison.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    plt.close(fig)
    table = "\n".join(f"| {NAMES[r['strategy']]} | {r['hits']}/499 | {100*r['candidate_recall_user_macro']:.4f}% | "
                      f"{100*r['candidate_recall_query_micro']:.4f}% | {r['repeat_hits']}/334 | {r['new_hits']}/165 |" for r in rows)
    source_table = "\n".join(
        f"| {source} | {strategies['baseline']['validation']['source_pool_recall']['100'][source]['hits']} | "
        f"{strategies['baseline']['validation']['source_pool_recall']['200'][source]['hits']} | "
        f"{strategies['affinity']['validation']['source_pool_recall']['100'][source]['hits']} | "
        f"{strategies['affinity']['validation']['source_pool_recall']['200'][source]['hits']} |"
        for source in ("popular", "repeat", "artist", "album"))
    actual_table = "\n".join(
        f"| {name} | {base['mean_candidates']:.2f} | {base['target_hits']} | "
        f"{strategies['repeat_heavy']['validation']['actual_selection'][name]['mean_candidates']:.2f} | "
        f"{strategies['repeat_heavy']['validation']['actual_selection'][name]['target_hits']} |"
        for name, base in strategies['baseline']['validation']['actual_selection'].items())
    b = strategies["baseline"]["validation"]
    best = strategies["repeat_heavy"]["validation"]
    delta = 100 * (best["candidate_recall_user_macro"] - b["candidate_recall_user_macro"])
    text = f"""# P002 — диагностика генератора кандидатов

Дата: 25 октября 2026 года. Статус: завершён. **Development-эксперимент; бустинг не обучался, итоговый test не оценивался.**

Увеличение квоты повторных треков дало лучший candidate recall среди четырёх
заранее заданных вариантов: 153/499 против исходных 130/499, user-macro
29,7115% против 25,2885% (+{delta:.4f} процентного пункта). Рост относится
к повторному потреблению: новые пользователю треки найдены в 4/165 запросов
против 5/165 у baseline. Поэтому этот вариант пока является кандидатом для
следующего сравнения, а проблема новых треков остаётся открытой.

## Что зафиксировано до результатов

Основание — [P001](PILOT_001.md), который сохраняется как отдельная обязательная
контрольная точка итогового отчёта. Протокол P002 закоммичен до выполнения:
`dc02423`; код исполнения — `{run['git_commit']}`. [Протокол](../configs/P002_PROTOCOL.md),
[конфигурация](../configs/retrieval_002.json), [предварительная независимая проверка](../research/P002_REVIEW.md).

Сохранены точные 1 349 train и 499 validation-запросов P001 от той же когорты
64 пользователей. Метрики — только по 499 validation-запросам от 52 пользователей.
Все четыре варианта имеют K=100 и один и тот же доступный каталог.

| Вариант | Квоты popular/repeat/artist/album | Глубина профиля | Использование пулов |
| --- | --- | ---: | --- |
| Baseline | 25/25/35/15 | 5 | Первые 100 в каждом источнике, как P001 |
| Repeat-heavy | 10/60/20/10 | 5 | Те же source rankings и operational depth100 |
| Affinity32 | 10/20/50/20 | 32 | Первые100, более широкий профиль исполнителей/альбомов |
| RRF | Без квот | 5 | Ранговое объединение четырёх top200 pools, c=60 |

В RRF score(item) = сумма `1/(60 + rank)` по источникам, где item присутствует;
rank начинается с 1. Ничьи разрешаются по item ID. Недобор квот заполняется
популярностью; источники обходятся в фиксированном порядке. Target не передаётся
генератору; все четыре списка формируются до обращения к target-derived полям.
Число кандидатов каждого варианта — ровно 100 на каждом validation-запросе.

## Время, доступность и воспроизводимость baseline

Границы исходного времени P001 сохранены без пересчёта: train от 9 752 280
до 20 125 775, validation до 23 380 165. До преобразования данных в Python
применён Arrow batch-фильтр `timestamp < 23 380 165`; дальнейший replay
подаёт генератору только строки `timestamp < query_timestamp`. В Python загружены
{run['development_rows_loaded']:,} development-событий, до последнего запроса
наблюдены {run['past_events_observed']:,}. Объекты и отзывы из будущего в кандидаты не добавляются.

Baseline совпал с P001 **по точной последовательности item ID на всех
{run['baseline_exact_queries_checked']:,} запросах**. Repeat/new и warm/cold также
сверены с историей до запроса. Хеши query file и трёх prepared inputs совпали
с сохранённой P001 контрольной точкой. Исходные результаты P001 сохранены.

Документация Yambda описывает гранулярность 5 секунд, но эпоха, календарный
период, способ построения 50M и полнота относительно 5B не установлены;
используется raw timestamp. Фиксированный mapping snapshot и доступность
logged events по timestamp остаются допущениями. [Проверяемые цитаты и пробелы](../research/YAMBDA_SEMANTICS.md).

## Результаты retrieval

User-macro — сначала среднее попаданий внутри пользователя, затем между
пользователями; query-micro — доля найденных целей во всех 499 запросах.
В знаменателе остаются misses и 70 cold targets; cold recall у всех вариантов 0.
Повторные/новые цели — разбиение относительно конкретного пользователя;
warm/cold — относительно доступного каталога когорты.

| Вариант | Найдено | User-macro recall | Query-micro recall | Повторные | Новые пользователю |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

В парном сравнении Repeat-heavy нашёл цели в 25 запросах, где baseline
промахнулся, и потерял 2 baseline-попадания: чистый прирост 23. Пользователей
с ростом/равенством/снижением среднего recall — 17/33/2 из 52. Это
описательные разницы; доверительные интервалы и значимость не рассчитывались.

![Retrieval и новые треки](p002_comparison.png)

Масштабы осей двух панелей графика разные. Affinity32 дал небольшое увеличение
общего покрытия и 6 новых целей вместо 5. Он одновременно меняет квоты и
глубину профиля, поэтому приписывать эффект одному расширению профиля нельзя.
RRF дал 136 попаданий — меньше Repeat-heavy; это результат заданной формулы,
глубины и единственной когорты, а не вывод о ранговом объединении вообще.

## Где теряются цели

Ниже число targets, встречающихся в source pool до квот, с общим знаменателем
499. Цель может находиться в нескольких источниках; складывать столбцы нельзя.

| Источник | Baseline top100 | Baseline top200 | Affinity32 top100 | Affinity32 top200 |
| --- | ---: | ---: | ---: | ---: |
{source_table}

Объединение baseline top200-пулов содержит 231/499 целей, Affinity32 top200 —
256/499. Это диагностическое покрытие объединённого пула переменного размера,
а не качество списка из 100 рекомендаций. Квоты и ранжирование пула теряют
часть уже доступных целей; другие цели отсутствуют даже в объединении пулов.

Для новых пользователю треков даже объединение top200-пулов покрывает только
13/165 целей baseline и 19/165 Affinity32. Значит, одна перестановка квот
не решает задачу новых треков: нужно отдельно улучшать сами источники кандидатов
и проверять доступность объектов. Это наблюдение по этой когорте, а не готовый
вывод о качестве следующего метода.

Фактические уникальные добавления в shortlist после дедупликации:

| Источник первого отбора | Baseline: среднее объектов | Baseline: целей | Repeat-heavy: среднее объектов | Repeat-heavy: целей |
| --- | ---: | ---: | ---: | ---: |
{actual_table}

Attribution означает первый источник в фиксированном порядке квот;
это не причинный вклад источника. Например, один трек может присутствовать
и в repeat, и в artist. В RRF уникальное авторство источника не назначается:
сохраняются memberships и score contributions через исходные ranks.
Средние pairwise overlaps depth200 сохранены в метриках, полные пулы — локально.

## Ресурсы и сохранённые доказательства

Сквозной runner: **{run['elapsed_seconds']:.3f} с**, peak process memory
**{run['resources']['peak_process_rss_mb']:.2f} МиБ**, минимум свободной системной
RAM **{run['resources']['minimum_available_system_mb']:.2f} МиБ**. Пороги
1 100/150 МиБ не нарушены. Фильтрация/загрузка {run['filtered_load_seconds']:.3f} с,
replay с записью diagnostics {run['replay_and_record_seconds']:.3f} с.
VPS для P002 не потребовалась. Этот retrieval-only замер не сопоставим напрямую
со временем/памятью полного feature+boosting P001.

Полные candidate records занимают {run['output_bytes']['candidate_records.jsonl']/2**20:.2f} МиБ,
evaluation rows — {run['output_bytes']['evaluation_rows.jsonl']/2**20:.2f} МиБ.
Они сохранены локально в `artifacts/retrieval_002/`; компактные метрики,
протокол и хеши — в Git. Source pool cache разделяется между стратегиями:
индивидуальные generation timers зависят от порядка вызова и не являются
независимыми бенчмарками вариантов.

39 синтетических тестов прошли до запуска, включая mixed-rowgroup sentinel,
exact baseline на искусственном примере, RRF hand example, user-macro,
fallback attribution и сохранение misses. [Независимая проверка результатов](../research/P002_RESULT_REVIEW.md)
и [пересчёт](../research/P002_RECHECK.json) описывают фактическую проверку outputs.

## Ограничения и дальнейшее решение

Это четыре заранее заданных development-варианта, выбранный capped query sample,
одна когорта и один seed. Итоговый test не оценивался; статистическая устойчивость
и качество ранжирования на новых кандидатах не установлены. Датасет содержит
наблюдаемые действия, а не полный эксперимент с экспозициями. Human verification
не объявляется выполненной: [реестр источников](../../SOURCES.md).

Для следующего этапа Repeat-heavy можно принять как development-кандидат
для общей retrieval-конфигурации, сохранив отдельный анализ новых треков.
Перед увеличением когорты нужно измерить и оптимизировать построение признаков.
Затем сравнить ручные/автоматические признаки на **одинаковых** кандидатах и
временных границах. Рост retrieval здесь не является ростом NDCG и не является
доказательством преимущества автоматизации признаков. P001 и P002 оба войдут
в итоговый отчёт как последовательные этапы исследования.

## Команды и файлы

Из корня проекта с сохранёнными входами P001 и установленными зависимостями:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s task_01_auto_features/tests -p 'test_*.py' -q
.\\.venv\\Scripts\\python.exe -u task_01_auto_features/src/run_retrieval.py
.\\.venv\\Scripts\\python.exe task_01_auto_features/src/report_retrieval.py
```

Runner отказывается перезаписывать завершённый или непустой output: повторный
опыт получает отдельный каталог/ID с указанием причины. Отчёт можно пересоздать
после clone по компактной копии метрик, без чтения event-данных и обучения.

- [Замороженные метрики и хеши](p002_snapshot/metrics.json), [CSV](p002_metrics.csv), [SVG](p002_comparison.svg).
- [Контрольная точка P002](p002_snapshot/README.md); локальные raw records не входят в clone.
- SHA256 исходного metrics JSON: `{hashlib.sha256(raw).hexdigest()}`.
"""
    (out / "P002_RETRIEVAL.md").write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(out / "P002_RETRIEVAL.md"), "strategies": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
