# Независимая итоговая проверка P002 retrieval-only

**Итог: PASS по сохранённым артефактам, baseline equality и независимому пересчёту метрик.** P002 завершён `complete`; обучение не выполнялось, `final_test_evaluated=false`. Проверял только P001 development queries и P002 records/evaluation outputs/status/config/source code. Содержимое event Parquet не читал и replay не запускал; SHA256 event Parquet пересчитал только как bytes для проверки provenance.

## Проверки целостности и протокола

- В `candidate_records.jsonl` потоково проверено 7 392 записей: четыре стратегии на тех же 1 848 train+validation queries. Варианты используют одинаковые query IDs, user, timestamp, split, target и strata. Сохранённые validation evaluation rows — по 499 на стратегию.
- P002-Baseline совпадает с сохранёнными P001 `candidate_items` **по порядку и item ID** на всех 1 848 запросах; `baseline_exact_queries_checked=1848`.
- Кандидатов в списке не больше 100, дубликатов нет; source pools не глубже 200 и не содержат повторов. Все candidates входят в объединение source pools. Ни один из 70 cold validation targets не попал в список.
- Независимо восстановил baseline/repeat-heavy/affinity quota selection из сохранённых pools и повторно посчитал RRF scores по `Σ 1/(60+rank)` с `c=60`; итоговые candidate IDs, source attribution, fallback counts, pool provenance и pairwise overlaps совпали с сохранёнными. Пересобранные per-query evaluation rows совпали с `evaluation_rows.jsonl`.
- Самостоятельно пересчитал query-micro и user-macro candidate recall, counts и strata repeat/new, warm/cold по per-query rows; полные validation и stratum summary совпали с `metrics.json`. Формулы усреднения не брались из `run_retrieval.summarize`.
- Кодовая проверка подтверждает, что `RetrievalCandidateGenerator.candidates` не принимает target, runner вызывает генератор только с user/timestamp/strategy и проверяет target лишь после формирования всех candidate lists. Это проверка контракта по коду и сохранённым артефактам; историю item на event-строках независимо не проверял, поскольку event Parquet читать запрещено.
- Проверены SHA256 config, protocol, четырех source-файлов, P001 query/checkpoint/metrics inputs, всех трёх prepared data-файлов и двух P002 output JSONL. Все хеши совпали с run manifest. Зафиксированный commit — `dadc4eda6f93b7e7a62c8cf8c7ef2797ea074976` (`dadc4ed`); рабочий HEAD на момент проверки совпал.
- Run status и config подтверждают `test_evaluation=false`, `model_training=false`. Process RSS — `197.38 MiB` при лимите `1 100 MiB`; минимум доступной памяти — `2 146.05 MiB` при резерве `150 MiB`; превышений нет. Время run — `42.695 s`.

## Validation candidate recall

Все четыре списка содержат в среднем ровно 100 кандидатов на запрос. Основная метрика — candidate recall: число target hits из 499 / query-micro / user-macro.

| Стратегия | Hits / 499 | Query-micro | User-macro |
| --- | ---: | ---: | ---: |
| Baseline | 130 / 499 | 0.260521 | 0.252885 |
| Repeat-heavy | 153 / 499 | 0.306613 | 0.297115 |
| Affinity32 | 139 / 499 | 0.278557 | 0.270192 |
| RRF | 136 / 499 | 0.272545 | 0.264423 |

Значения совпали с указанными до проверки наблюдаемыми результатами. Это описательные значения на зафиксированном validation sample; они не дают статистического вывода о популяционном эффекте.

## Срезы

Числа в ячейках — hits/queries; macro — user-macro candidate recall на срезе.

| Стратегия | Repeat | New | Warm | Cold |
| --- | --- | --- | --- | --- |
| Baseline | 125/334; 0.409606 | 5/165; 0.020973 | 130/429; 0.301442 | 0/70; 0 |
| Repeat-heavy | 149/334; 0.484549 | 4/165; 0.018845 | 153/429; 0.355062 | 0/70; 0 |
| Affinity32 | 133/334; 0.437773 | 6/165; 0.031611 | 139/429; 0.320578 | 0/70; 0 |
| RRF | 131/334; 0.427166 | 5/165; 0.020973 | 136/429; 0.314423 | 0/70; 0 |

Знаменатель во всех случаях включает все 499 validation queries, включая 70 cold targets и targets, отсутствующие в candidates. Метрики по срезам рассчитаны внутри своих 334/165 и 429/70 queries; это диагностические разрезы одного и того же sample.

## Парные изменения относительно baseline

«Лучше/равно/хуже» для query — сравнение наличия target в candidate list. Для пользователя сравнивались query-averaged recall пользователя и затем знак разности; значимость не оценивалась.

| Стратегия | Queries: лучше / равно / хуже; net hits | Users: лучше / равно / хуже из 52 | Δ user-macro recall |
| --- | --- | --- | ---: |
| Repeat-heavy | 25 / 472 / 2; +23 | 17 / 33 / 2 | +0.044231 |
| Affinity32 | 13 / 482 / 4; +9 | 10 / 39 / 3 | +0.017308 |
| RRF | 9 / 487 / 3; +6 | 6 / 44 / 2 | +0.011538 |

## Source diagnostics и границы вывода

Selection attribution и overlaps независимо восстановлены из сохранённых source pools/provenance и совпали с summary. Например, в baseline среднее число уникально добавленных candidates на validation query было: popular 25.00, repeat 24.50, artist 34.88, album 12.75, popular fallback 2.87. Квоты могут недобираться из-за пересечений и недостатка доступных объектов; доли этих причин отдельно не измерялись. В repeat-heavy repeat добавил 58.13 кандидата в среднем; Affinity32 дал наибольшие средние пересечения artist–album (68.89 item на запрос в pools), что видно в диагностике до квот.

Результат относится к 64-user когорте и 499 отобранным validation queries из P001, фиксированному snapshot artist/album, условному каталогу прошлых item и четырём заданным retrieval-вариантам. Здесь нет оценки full-catalog retrieval, независимого test-периода или доверительных интервалов. P001 NDCG не сравнивается с P002 candidate recall как эффект признаков; P002 не обучал ranking model. Event Parquet был открыт только на уровне raw bytes для SHA256; его строки и p85+ период не декодировались и не использовались.

Подробный агрегированный пересчёт и проверенные хеши: [P002_RECHECK.json](P002_RECHECK.json).
