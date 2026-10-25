# P004 — независимая проверка результата

Дата проверки: 2026-10-25. Проверены завершённые P004 manifests, inner-результаты, frozen selection, pruning, schema/input manifests, run hash ledger и сохранённые финальные prediction-файлы. Повторная генерация признаков, чтение событий, обучение и пересчёт внешних метрик не выполнялись. Внешний пересчёт метрик predictions ведётся отдельно.

## Целостность и A1 gate

Run `p004_auto64` имеет статус `complete`; итоговый test не оценивался. Сохранённые `specs.json` содержат 206 уникальных feature IDs: 66 A1 и 140 A2. Состав A2 соответствует протоколу: 56 новых оконных count, 42 smoothed share и 42 trend выражения; A1 является первым блоком с исходными UIDC именами.

A1 gate run manifest сообщает byte-for-byte совпадение float32 для 12 196 800 ячеек (184 800 строк × 66 признаков); сохранённый SHA сырых значений — `9758c104535d9f34b2ba5d552c5074a5d75abd8e68eed1df5824cbdbba2a866c`. В этой независимой review полной матрицы заново не читалась и `numpy.array_equal` не повторялся; сверены gate manifest, feature IDs и зарегистрированные хеши. Ранее выполненные synthetic A1 проверки являются отдельным доказательством реализации, не заменяющим full-matrix gate.

Pruning выполнен по inner fit: A1 — 2 константных и 8 дублирующих столбцов удалены, осталось 56 из 66; A2 — 2 константных и 28 дублирующих удалены, осталось 176 из 206. Представители дубликатов и причины записаны в `pruning.json`. Все candidate masks используют только оставшиеся колонки.

## Разбиение и выбор победителей

Frozen split: 943 inner-fit запросов, 406 inner-selection запросов, 499 внешних validation запросов; временная граница inner split — RAW `17 870 780`. На inner fit у моделей было 943 запроса, из которых target присутствовал в фиксированном candidate pool в 244 (699 пропущены). Inner selection охватил 406 запросов и 41 пользователя.

Протокол включает `balanced32`. Я независимо восстановил full-fit importance ranking из `inner_results.json`, применил стабильную сортировку по importance и feature ID и построил balanced masks по группам U/I/D/C. Для A1 balanced32 совпал с top32; для A2 также совпал с top32. Дедупликация одинаковых masks в `select_trial_columns` поэтому корректно оставила один fit для каждого из этих совпавших предложений. Наблюдаемые fits: A1 — full/top8/top16/top32 (4 уникальные masks), A2 — full/top8/top16/top32/top64 (5 уникальных masks). Восстановленные masks совпали с сохранёнными trial columns. Лучший A1 — `top16`, user-macro NDCG@20 `0,1086752`; лучший A2 — `top8`, `0,0962142`. Внутренний выбор победителя Auto сделан по этим значениям: A1, без использования внешних метрик.

Frozen manifests согласованы: A1 победитель содержит 16 features; A2 — 8; `chosen_auto_by_inner` совпадает с A1 winner. SHA256 `selection_frozen.json` пересчитан и равен `selection_frozen_sha256` в run status (`c7f43fef5c7e93b9e64a4f7ee6063b4247fb7cb8cb768378b787303003c7139d`). Внешняя оценка в статусе началась после сохранения frozen selection.

## Fits, masks и сохранённые predictions

Run ledger содержит 9 успешных inner fits и 11 успешных финальных fits: 20 фактических обучений суммарно, ниже лимита 28. Из 11 финальных уникальных fits девять — ручные controls и два — Auto A1/A2. В `final_models` есть 16 протокольных вариантов, поскольку пять Union/ablation masks уже совпадали с ранее обученными ручными masks и были переиспользованы, а не обучены повторно:

- Union совпал с `M_UIDC` (66 columns): выбранный Auto A1 целиком находился внутри ручного набора.
- Union без U/I/D/C соответственно совпал с `M_IDC` / `M_UDC` / `M_UIC` / `M_UID`.

Сохранённые prediction payloads для M_I, M_UIDC, Auto_A1 и Auto_A2 имеют по 499 query IDs, offsets `0..49 900` и ровно 100 candidate score slots на запрос; query order и offsets совпадают между вариантами. Все ledger hashes для source files и сохранённых manifests/models/predictions/results были сверены с файлами. Config и protocol SHA256 также совпали с текущими `p004_auto.json` и `P004_PROTOCOL.md`. Зафиксированные frozen input hashes связывают опыт с P001 queries/data и P003 baseline matrix/schema/input.

## Внешние результаты, сохранённые в run outputs

Ниже — значения из завершённых `results/*.json` и `run_status.json`. Независимый verifier в [`verification_independent.json`](../artifacts/p004_auto64/verification_independent.json) имеет статус `verified`: он пересчитал predictions и подтвердил метрики с tolerance `1e-12` для всех 25 результатов, включая 16 внешних вариантов и 9 inner-selection trials. Этот пересчёт выполнял отдельный verifier, не автор текущей review.

| Вариант | Признаков | User-macro NDCG@20 | User-macro Recall@20 |
| --- | ---: | ---: | ---: |
| M_I | 12 | 0,075981 | 0,158654 |
| M_UIDC | 66 | 0,075126 | 0,177885 |
| Auto A1 | 16 | 0,078346 | 0,170192 |
| Auto A2 | 8 | 0,078076 | 0,175962 |

Все варианты использовали те же 499 запросов и candidate pool; цели присутствовали в нём для 130/499 запросов (26,05%), поэтому retrieval ограничивает потолок end-to-end метрик. Сохранённые результаты показывают небольшой прирост NDCG@20 у Auto A1/A2 относительно M_UIDC и более низкий Recall@20, но 52 validation-пользователя, один seed и повторное dev использование не позволяют считать это устойчивым превосходством. Поиск не расширялся после чтения внешней validation.

Runner занял 160,273 с; построение 206 признаков — 108,671 с, из них transform 100,451 с. Peak RSS — 678,77 MiB при лимите 1 100 MiB; минимальная доступная память — 986,31 MiB при резервном пороге 150 MiB. Stage завершился в пределах заданных ресурсных ограничений.

## Итог

P004 method proof завершён: 206-выражений DSL воспроизводим, A1 точно совпал с manual UIDC на всех 12 196 800 ячейках, выбор Auto победителя был заморожен по inner-selection, а общий fit budget составил 20/28. Сохранённые внешние оценки Auto выше по NDCG@20 и ниже по Recall@20, чем M_UIDC; малая когорта и ранее использовавшаяся development validation ограничивают вывод. Это не подтверждает надёжный quality gain или масштабируемость, не является итоговым test результатом и не разрешает экстраполяцию на большую когорту.
