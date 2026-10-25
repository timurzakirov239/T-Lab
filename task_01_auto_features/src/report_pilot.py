"""Render the completed pilot's saved results; never train or read test events."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifacts = ROOT / "artifacts" / "pilot_001"
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    metrics_path = artifacts / "metrics.json"
    if not metrics_path.exists():
        metrics_path = out / "pilot_001_snapshot" / "metrics.json"
    raw = metrics_path.read_bytes()
    run = json.loads(raw)
    if run["status"] != "complete" or run["final_test_evaluated"]:
        raise ValueError("Expected a completed development-only pilot")
    models = run["models"]
    rows = []
    for model in models:
        overall = model["metrics"]["overall"]
        rows.append({
            "model": model["model"],
            "features": model["feature_count"],
            "ndcg20_user_macro": overall["user_macro"]["ndcg@20"],
            "recall20_user_macro": overall["user_macro"]["recall@20"],
            "candidate_recall_query_micro": overall["candidate_recall"],
            "fit_seconds": model["fit_seconds"],
        })
    with (out / "pilot_001_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    labels = [r["model"] for r in rows]
    y = np.arange(len(rows))
    colors = ["#2274a5" if r["model"] == "B1_I" else
              "#d87524" if r["model"] == "B5_UIDC" else "#9baebc" for r in rows]
    for ax, key, title in zip(axes, ["ndcg20_user_macro", "recall20_user_macro"],
                              ["NDCG@20", "Recall@20"]):
        values = [r[key] for r in rows]
        bars = ax.barh(y, values, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_xlim(0, max(values) * 1.24)
        ax.set_title(title)
        ax.set_xlabel("Среднее по пользователям")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("P001 · Yambda · validation: 499 запросов, 52 пользователя\n"
                 "64 пользователя в когорте · один seed · пропуски кандидатов дают 0", fontsize=12)
    fig.savefig(out / "pilot_001_comparison.png", dpi=180)
    fig.savefig(out / "pilot_001_comparison.svg")
    plt.close(fig)

    table = "\n".join(
        f"| {r['model']} | {r['features']} | {r['ndcg20_user_macro']:.6f} | "
        f"{r['recall20_user_macro']:.6f} | {r['fit_seconds']:.3f} |" for r in rows)
    strata = models[0]["metrics"]["strata"]
    resources = run["resources"]
    builds = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(metrics_path.parent.glob("feature_build_*.json"))]
    build_table = "\n".join(
        f"| {b['short_window_raw']} | {b['elapsed_seconds']:.3f} | {b['rows']:,} | "
        f"{b['features']} | {b['matrix_mb']:.3f} |" for b in builds)
    report = f"""# P001 — локальный пилот построения признаков

Дата: 25 октября 2026 года. Статус: завершён. **Это development-пилот, а не итоговый отчёт работы 1.**

Сквозной процесс подготовки историй, генерации кандидатов, причинных агрегатов,
CatBoost и оценки работает локально. На этой небольшой выборке признаки объектов
дали лучший NDCG@20, а полный U/I/D/C — лучший Recall@20. Дополнительное затухание
и альтернативные окна не улучшили основной критерий. Статистическая устойчивость
различий этим запуском не установлена.

## Данные и постановка

Источник: [Yambda](https://huggingface.co/datasets/yandex/yambda), опубликованный
вариант 50M, revision `dd6f3a19eef5866e346c3270e098baa641a44948`.
В исходной таблице 47 790 449 событий. Seed 20261025; равновероятно выбраны
64 пользователя, сохранены все их 318 157 событий в этом варианте: 46 678 объектов.
Полнота относительно Yambda-5B не установлена. Глобальные популярности и каталог
вычисляются внутри выбранной когорты. Покрытие объектов связями artist — 99,719%,
album — 98,794%; связи дедуплицированы. Предполагается доступность фиксированного
снимка справочников на момент рекомендации.

Цель — зарегистрированный listen в момент запроса. Запросы задаются по listen,
а неоднозначные группы с несколькими listen одного пользователя в одно время
исключаются: 10 515 train-групп и 4 069 validation-групп. Это оценка logged targets,
не CTR по показам и не доказательство причинного эффекта рекомендаций.

Время используется в исходных единицах. Границы вычислены по квантилям времён
выбранной офлайн-когорты, а не заданы календарными датами:

- train-запросы: `9 752 280 ≤ t < 20 125 775`;
- validation: `20 125 775 ≤ t < 23 380 165`;
- последний период `t ≥ 23 380 165` зарезервирован для итоговой оценки.

Перед запросом видны только события с `timestamp < t`; равные метки не видят
друг друга. Истории не сокращаются при отборе запросов. Максимум 30 train и 10
validation запросов на пользователя: получены 1 349 train и 499 validation,
в validation представлены 52 пользователя. До ограничения числа запросов
доступны 85 343 и 32 062 однозначных запроса соответственно. Результаты относятся
к выбранным 499 запросам. Финальные played ratio, длина и текущий organic flag
не используются как признаки; доступность прошлых событий по logged timestamp
остаётся явным допущением replay.

## Кандидаты и обучение

Фиксированный генератор использует прошлую популярность, повторные треки,
предпочитаемых исполнителей и альбомы, квоты 25/25/35/15, максимум 100 объектов.
В validation получилось ровно 100 кандидатов на запрос. Целевой объект не
добавляется принудительно. Один и тот же набор кандидатов используется во всех
сравнениях; пропущенный target остаётся в знаменателе и даёт нулевое качество.

В обучении target найден для 349 из 1 349 запросов. Эти 349 запросов дают
34 900 строк: один положительный объект и 99 остальных кандидатов в каждом.
Остальные 1 000 запросов в обучение классификатора не входят. Остальные кандидаты
являются неотмеченными, а не подтверждёнными отрицательными предпочтениями.
Это условная выборка с balanced weights; предсказания не трактуются как
калиброванная вероятность естественного отклика.

Для всех обучаемых вариантов: CatBoostClassifier, Logloss, 120 итераций, depth=5,
learning_rate=0.07, l2=5, два CPU-потока, общий seed. U — история пользователя;
I — популярность/давность объекта и степени mapping; D — история пары;
C — профили исполнителей и альбомов с вкладом `1/k` для k уникальных связей.
Полный основной набор содержит 66 признаков. X добавляет 26 признаков затухания.
ID напрямую в бустинг не передаются.

Ручное короткое окно — 670 390 исходных единиц; ограниченный поиск проверяет
ещё 201 117 и 2 011 171. Half-life X — 670 390. Это три варианта окна,
а не полноценная автоматизация поиска связей, агрегатов и сочетаний признаков.

## Качество на validation

Основной критерий выбора — **user-macro NDCG@20**. Сначала метрика усредняется
по запросам пользователя, затем по 52 пользователям. Recall@20 для одной цели —
попадание в первые 20; NDCG учитывает позицию. Ничьи разрешаются по item ID.
Время в таблице относится только к обучению модели.

| Вариант | Признаков | NDCG@20 | Recall@20 | Обучение, с |
| --- | ---: | ---: | ---: | ---: |
{table}

![Сравнение вариантов](pilot_001_comparison.png)

Наибольший NDCG@20 у B1_I: 0,075981. Полный B5_UIDC: 0,075126,
но Recall@20 выше: 0,177885 против 0,158654. Малые различия между
бустингами не позволяют объявить устойчивое преимущество. X и два альтернативных
окна основной критерий не улучшили. Выбор и сравнение сделаны на development-данных,
один seed; доверительные интервалы и повторные когорты пока не рассчитаны.

## Ограничение генератора кандидатов

Target найден в 130 из 499 запросов: **query-micro candidate recall 26,0521%**.
369 targets пропущены, включая 70 объектов, ещё не известных в выбранной истории.
Этот показатель имеет другой способ усреднения, чем user-macro метрики выше;
его нельзя напрямую использовать как их численный верхний предел.

| Тип цели | Запросов | Target найден | Candidate recall, query-micro |
| --- | ---: | ---: | ---: |
| Повторная для пользователя | 334 | 125 | {strata['repeat']['candidate_recall']:.6f} |
| Новая для пользователя | 165 | 5 | {strata['new']['candidate_recall']:.6f} |
| Уже известная в когорте | 429 | 130 | {strata['warm']['candidate_recall']:.6f} |
| Ещё неизвестная в когорте | 70 | 0 | 0 |

Строки повторная/новая и известная/неизвестная — два разных разбиения.
Генератор особенно редко находит новые для пользователя треки; ранжирование
не может исправить отсутствие target среди кандидатов. Следующий приоритет —
проверить retrieval и перенести сравнения на более представительную когорту,
сохранив одинаковые кандидаты внутри каждого сравнения признаков.

## Время и память

Подготовка данных заняла 8,276 с по отдельному таймеру. Основной runner —
**{run['elapsed_seconds']:.3f} с**, включая три построения признаков и 10 обучений
(плюс popularity). Эти таймеры не включают написание кода и установку зависимостей.

| Короткое окно, raw | Построение, с | Строк матрицы | Столбцов | Матрица, МиБ |
| --- | ---: | ---: | ---: | ---: |
{build_table}

Матрица включает 1 848 запросов × 100 кандидатов; обучающие строки с retrieval miss
позже исключаются. Построение признаков занимает основную часть времени;
каждое обучение — около 1–3 с. Replay наблюдал 270 367 событий до конца validation.

Пиковая память процесса runner: **{resources['peak_process_rss_mb']:.2f} МиБ**;
минимум свободной системной памяти: **{resources['minimum_available_system_mb']:.2f} МиБ**.
Монитор с интервалом 0,25 с дополнен Windows peak working set. Пороги остановки
1 100 МиБ процесса и 150 МиБ свободной памяти не сработали. Пик отдельного
процесса подготовки данных не измерялся. CatBoost выдавал предупреждения
`used_ram_limit=512mb`: это мягкая настройка, фактическая память процесса выше.

Локальных ресурсов для этого пилота хватило; VPS сейчас не требуется.
Свободной RAM оставалось мало. Масштабирование требует измерений и оптимизации,
линейный прогноз на весь 50M по этим значениям не делаем.

## Проверки и границы выводов

23 синтетических теста прошли: временные границы, полнота выбранных историй,
mapping, причинные признаки, дробный вклад, decay, знаменатели и ничьи.
Независимый Luna-субагент пересчитал полное дерево метрик B1_I и B5_UIDC
из сохранённых predictions с совпадением до `1e-12`; проверил IDs/offsets
10 файлов и контрольные суммы подготовленных данных.
См. [рецензию результатов](../research/PILOT_RESULT_REVIEW.md).

Процедурное отклонение проверки: для вспомогательной проверки доступности
кандидатов субагент прочитал item_id/timestamp всего подготовленного файла,
включая зарезервированный период. Отклонение зафиксировано; test-метрики
не вычислялись, эти строки не использованы для обучения/ранжирования или
подбора вариантов. Проверка не была полностью изолирована от чтения test.

Дополнительные ограничения: небольшая когорта; один seed; исключение неоднозначных
listen-групп; ограничение запросов на пользователя; условное обучение; фиксированный
снимок mapping; raw-время; каталог только по когорте; отсутствие показов и
экспозиционных меток; подбор на validation. Полный статистический аудит исходного
50M не завершён. **Инструментальная проверка не заменяет требуемую человеческую
проверку фактов; её статус ведётся в [SOURCES.md](../../SOURCES.md).**

## Воспроизведение и артефакты

Из корня `D:\\bot\\T-Lab`, с уже загруженными исходными данными:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r task_01_auto_features/requirements-pilot.lock.txt
.\\.venv\\Scripts\\python.exe task_01_auto_features/src/pilot_data.py --users 64 --seed 20261025 --output artifacts/pilot_001/data
.\\.venv\\Scripts\\python.exe -m unittest discover -s task_01_auto_features/tests -p 'test_pilot*.py' -v
.\\.venv\\Scripts\\python.exe -u task_01_auto_features/src/run_pilot.py
.\\.venv\\Scripts\\python.exe task_01_auto_features/src/report_pilot.py
```

Последняя команда пересоздаёт только отчёт/CSV/график из готовых результатов,
без обучения и чтения тестовых событий. Повторное обучение выполняйте в копии
папки проекта, чтобы сохранить артефакты этого запуска.

- [Протокол до обучения](../configs/PILOT_PROTOCOL.md) и [конфигурация](../configs/pilot_001.json).
- [Замороженные метрики](pilot_001_snapshot/metrics.json), [CSV](pilot_001_metrics.csv), [SVG](pilot_001_comparison.svg).
- [Статус, хеши кода и данных](pilot_001_snapshot/run_status.json).
- [Контрольная точка P001](pilot_001_snapshot/README.md), [структура итогового отчёта](REPORT_OUTLINE.md).
- [Описание подготовки данных](../research/PILOT_DATA.md) и [признаков](../research/PILOT_FEATURES.md).
- Сохранены `queries.jsonl`, `models/*.cbm`, `predictions/*.npz`, `feature_build_*.json` в `artifacts/pilot_001/`.

SHA256 исходного `metrics.json` этого отчёта: `{hashlib.sha256(raw).hexdigest()}`.

## Следующий этап работы 1

1. Проверить календарную семантику времени, происхождение 50M и доступность метаданных.
2. Разобрать retrieval по новым/повторным объектам и проверить улучшенный генератор.
3. Оптимизировать построение признаков, затем измерить ступень 256–1 000 пользователей
   с полными доступными историями и несколькими seed/когортами.
4. Реализовать ограниченный бюджетом поиск связей, агрегатов и окон, сопоставить
   с сильным ручным набором, включая отдельные artist/album абляции.
5. После фиксации метода выполнить единственную итоговую оценку на test и подготовить
   полный исследовательский отчёт. Работа 2 начнётся после завершения работы 1.
"""
    (out / "PILOT_001.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(out / "PILOT_001.md"), "models": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
