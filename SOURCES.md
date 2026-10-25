# Реестр источников

**Актуальный статус работы 1, 25.10.2026:** автор явно подтвердил личную проверку всех источников и ключевых утверждений. [Основание подтверждения](task_01_auto_features/report/AUTHOR_REVIEW_CONFIRMATION.json). Прежние записи «ожидается» в исторических отчётах фиксируют состояние на дату соответствующего этапа. Статьи P1/P2 второго задания не входят в это подтверждение.


Здесь перечислены источники из присланных условий. Документация датасетов открыта и сохранена при загрузке 25 октября 2026 года; статьи пока не открывались. Проверка инструментами не означает проверку человеком.

| ID | Источник | Откуда | Статус |
| --- | --- | --- | --- |
| A1 | `task_01_auto_features/assignment.txt` | Первое присланное условие | Прочитано при подготовке структуры |
| A2 | `task_02_rl_recommendations/assignment.txt` | Второе присланное условие | Прочитано при подготовке структуры |
| D1 | <https://github.com/irsafilo/KION_DATASET> | Оба задания | README прочитан инструментами; данные скачаны, хеши проверены; числа строк и даты KION проверены по CSV; проверка автором подтверждена для работы 1 |
| D2 | <https://huggingface.co/datasets/yandex/yambda> | Оба задания | README и LICENSE прочитаны инструментами; Yambda-50M и два справочника скачаны, SHA256 и Parquet metadata/schema проверены; полный событийный аудит таблиц не выполнен; личная проверка источника и утверждений работы 1 подтверждена автором25.10.2026 |
| P1 | <https://arxiv.org/pdf/2403.04875> | Второе задание | Существование, название и содержание статьи не проверены |
| P2 | <https://arxiv.org/pdf/2607.02818> | Второе задание | Существование, название и содержание статьи не проверены |

Для каждого источника, который войдёт в итоговый отчёт, добавим проверенное название, авторов, дату, конкретный подтверждаемый тезис и место в источнике: раздел, страницу или таблицу. Отдельно отметим человеческую проверку. Проверку инструментами нельзя автоматически отмечать как проверку человеком.

Собственные результаты P001–P006 связаны с конфигурациями, версиями кода и компактными проверяемыми метриками; P006 выполнен по отдельно закреплённому test-протоколу. Автор подтвердил личную проверку источников и утверждений работы 1 25.10.2026; основание сохранено отдельно.

Версии датасетов и обнаруженные расхождения с документацией: [task_01_auto_features/data/DOWNLOADS.md](task_01_auto_features/data/DOWNLOADS.md).

## Материалы итогового отчёта работы 1

| ID | Утверждения | Проверяемое основание | Человеческая проверка |
| --- | --- | --- | --- |
| R1 | Пилот, U/I/D/C, окна, ресурсы и процедура аудита | [P001](task_01_auto_features/report/PILOT_001.md), [snapshot](task_01_auto_features/report/pilot_001_snapshot/README.md) | Подтверждено автором 25.10.2026 |
| R2 | Retrieval-покрытие | [P002](task_01_auto_features/report/P002_RETRIEVAL.md), компактные метрики | Подтверждено автором 25.10.2026 |
| R3 | Эквивалентность без ускорения; failed scale256 | [P003](task_01_auto_features/report/P003_FEATURE_PIPELINE.md) | Подтверждено автором 25.10.2026 |
| R4 | Автоматическое развёртывание DSL и ограниченный отбор | [P004](task_01_auto_features/report/P004_AUTO_FEATURES.md), src/auto_features.py и src/auto_search.py | Подтверждено автором 25.10.2026 |
| R5 | Две новые development-когорты | [P005](task_01_auto_features/report/P005_ROBUSTNESS.md), [snapshot](task_01_auto_features/report/p005_snapshot/README.md) | Подтверждено автором 25.10.2026 |
| R6 | Методика единственного финального test | [P006 protocol](task_01_auto_features/configs/P006_PROTOCOL.md), [pre-open manifest](task_01_auto_features/configs/p006_freeze_manifest.json) | Подтверждено автором 25.10.2026 |
| R7 | Итоговая оценка P006:18 результатов, pooled метрики и условный bootstrap | [P006 результат](task_01_auto_features/report/P006_FINAL_TEST.md), [snapshot](task_01_auto_features/report/p006_snapshot/README.md), [рецензия](task_01_auto_features/research/P006_RESULT_REVIEW.md) | Подтверждено автором 25.10.2026 |
| R8 | P007: новые selectors и оценка всех validation queries на K100/полном cohort catalog; выбранные признаки, стоимость, переходы targets и условный bootstrap | [P007 протокол](task_01_auto_features/configs/P007_PROTOCOL.md), [отчёт](task_01_auto_features/report/P007_EXPANDED_COMPARISON.md), [snapshot](task_01_auto_features/report/p007_snapshot/README.md), [runtime recheck](task_01_auto_features/research/P007_RUNTIME_RECHECK.md) | Подтверждено автором 25.10.2026 |
| R9 | P008: uniform/hard negatives, расширение train при фиксированных признаках; primary и вторичные сравнения, падение на повторах | [Протокол](task_01_auto_features/configs/P008_PROTOCOL.md), [отчёт](task_01_auto_features/report/P008_SAMPLED_TRAINING.md), [snapshot](task_01_auto_features/report/p008_snapshot/README.md), [рецензия](task_01_auto_features/research/P008_RESULT_REVIEW.md), [чек-лист автора](task_01_auto_features/report/P008_HUMAN_REVIEW.md) | Подтверждено автором 25.10.2026 |

Для итоговой авторской проверки: [HUMAN_REVIEW.md](task_01_auto_features/report/HUMAN_REVIEW.md); общий текст — [FINAL_REPORT.md](task_01_auto_features/report/FINAL_REPORT.md). Автоматические численные проверки завершены; авторское подтверждение работы 1 получено25.10.2026.

Для чтения человеком: [аудит утверждений](task_01_auto_features/research/FINAL_CLAIM_AUDIT.md). Его наличие означает подготовку проверяемых оснований, а не подтверждённую человеческую проверку. Непроверенные статьи P1/P2 относятся к работе2 и не используются как обоснование метода работы1.

## Методические источники для раздела 12

В отчёт добавлены ссылки на официальную документацию CatBoost, scikit-learn и SciPy, а также на закреплённые исходники собственных процедур. Проверены содержание общих принципов и доступность 16 уникальных ссылок; квитанция: `task_01_auto_features/report/submission_sources_link_check_v13.json`. Ссылки scikit-learn/SciPy служат объяснением общих методов, их готовые функции в собственных процедурах не вызывались. Stable206 не приравнивается к классическому Stability Selection. Новые ссылки проверены инструментами; прежнее подтверждение автора не переносится автоматически на эти добавления.

| Материал | Что обосновывает | Статус новых ссылок |
| --- | --- | --- |
| CatBoost: algorithm-main-stages и fstr | Модель и PredictionValuesChange для A1/A2 | Прочитано инструментами; нового подтверждения автора нет |
| scikit-learn: grid_search, permutation_importance, SequentialFeatureSelector | Общие принципы случайного, перестановочного и пошагового отбора | Прочитано инструментами; нового подтверждения автора нет |
| scikit-learn: ndcg_score; SciPy: bootstrap | Общие определения метрики и парной повторной выборки | Прочитано инструментами; нового подтверждения автора нет |
| Исходники ffb2ed5: auto_search, auto_features, expanded_search, run_expanded_search, sampled_negatives, pilot_evaluation, final_test_analysis; P008_PROTOCOL | Точные собственные процедуры, отличие от библиотечных алгоритмов | Реализация сверена; ссылки на закреплённый commit доступны |
