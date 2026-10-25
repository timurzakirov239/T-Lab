# Реестр источников

Здесь перечислены источники из присланных условий. Документация датасетов открыта и сохранена при загрузке 25 октября 2026 года; статьи пока не открывались. Проверка инструментами не означает проверку человеком.

| ID | Источник | Откуда | Статус |
| --- | --- | --- | --- |
| A1 | `task_01_auto_features/assignment.txt` | Первое присланное условие | Прочитано при подготовке структуры |
| A2 | `task_02_rl_recommendations/assignment.txt` | Второе присланное условие | Прочитано при подготовке структуры |
| D1 | <https://github.com/irsafilo/KION_DATASET> | Оба задания | README прочитан инструментами; данные скачаны, хеши проверены; числа строк и даты KION проверены по CSV; человеческая проверка не выполнена |
| D2 | <https://huggingface.co/datasets/yandex/yambda> | Оба задания | README и LICENSE прочитаны инструментами; Yambda-50M и два справочника скачаны, SHA256 и Parquet metadata/schema проверены; полное чтение таблиц и человеческая проверка не выполнены |
| P1 | <https://arxiv.org/pdf/2403.04875> | Второе задание | Существование, название и содержание статьи не проверены |
| P2 | <https://arxiv.org/pdf/2607.02818> | Второе задание | Существование, название и содержание статьи не проверены |

Для каждого источника, который войдёт в итоговый отчёт, добавим проверенное название, авторов, дату, конкретный подтверждаемый тезис и место в источнике: раздел, страницу или таблицу. Отдельно отметим человеческую проверку. Проверку инструментами нельзя автоматически отмечать как проверку человеком.

Собственные результаты P001–P006 связаны с конфигурациями, версиями кода и компактными проверяемыми метриками; P006 выполнен по отдельно закреплённому test-протоколу. Human verification этих результатов и источников не подтверждена пользователем.

Версии датасетов и обнаруженные расхождения с документацией: [task_01_auto_features/data/DOWNLOADS.md](task_01_auto_features/data/DOWNLOADS.md).

## Материалы итогового отчёта работы 1

| ID | Утверждения | Проверяемое основание | Человеческая проверка |
| --- | --- | --- | --- |
| R1 | Пилот, U/I/D/C, окна, ресурсы и процедура аудита | [P001](task_01_auto_features/report/PILOT_001.md), [snapshot](task_01_auto_features/report/pilot_001_snapshot/README.md) | Ожидается |
| R2 | Retrieval-покрытие | [P002](task_01_auto_features/report/P002_RETRIEVAL.md), компактные метрики | Ожидается |
| R3 | Эквивалентность без ускорения; failed scale256 | [P003](task_01_auto_features/report/P003_FEATURE_PIPELINE.md) | Ожидается |
| R4 | Автоматическое развёртывание DSL и ограниченный отбор | [P004](task_01_auto_features/report/P004_AUTO_FEATURES.md), src/auto_features.py и src/auto_search.py | Ожидается |
| R5 | Две новые development-когорты | [P005](task_01_auto_features/report/P005_ROBUSTNESS.md), [snapshot](task_01_auto_features/report/p005_snapshot/README.md) | Ожидается |
| R6 | Методика единственного финального test | [P006 protocol](task_01_auto_features/configs/P006_PROTOCOL.md), [pre-open manifest](task_01_auto_features/configs/p006_freeze_manifest.json) | Ожидается |
| R7 | Итоговая оценка P006:18 результатов, pooled метрики и условный bootstrap | [P006 результат](task_01_auto_features/report/P006_FINAL_TEST.md), [snapshot](task_01_auto_features/report/p006_snapshot/README.md), [рецензия](task_01_auto_features/research/P006_RESULT_REVIEW.md) | Ожидается |
| R8 | P007: новые selectors и оценка всех validation queries на K100/полном cohort catalog; выбранные признаки, стоимость, переходы targets и условный bootstrap | [P007 протокол](task_01_auto_features/configs/P007_PROTOCOL.md), [отчёт](task_01_auto_features/report/P007_EXPANDED_COMPARISON.md), [snapshot](task_01_auto_features/report/p007_snapshot/README.md), [runtime recheck](task_01_auto_features/research/P007_RUNTIME_RECHECK.md) | Ожидается |

Для итоговой авторской проверки: [HUMAN_REVIEW.md](task_01_auto_features/report/HUMAN_REVIEW.md); общий текст — [FINAL_REPORT.md](task_01_auto_features/report/FINAL_REPORT.md). Автоматические численные проверки завершены; человеческое подтверждение остаётся открытым.

Для чтения человеком: [аудит утверждений](task_01_auto_features/research/FINAL_CLAIM_AUDIT.md). Его наличие означает подготовку проверяемых оснований, а не подтверждённую человеческую проверку. Непроверенные статьи P1/P2 относятся к работе2 и не используются как обоснование метода работы1.
