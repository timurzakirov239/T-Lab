# P007 — финальная read-only рецензия результатов

Сверены `report/P007_EXPANDED_COMPARISON.md` с compact snapshot:
`analysis.json`, `ranks.csv`, cohort-specific status/metrics и
`verification_independent.json`. Raw histories, models и event payload не
читались; P006 не запускался и его артефакты не изменялись.

## Сверка охвата и независимой проверки

Verifier сообщает `verified` для всех трёх cohorts, всех восьми fitted
rankers и девяти rank groups с Popularity. В snapshot 34 hash-tracked files
(35 файлов с самим checkpoint manifest) и 27 558 rank rows — ровно
1 531 queries × 9 методов × 2 evaluation modes. У каждой пары mode/method
1 531 строка; per-cohort строки — 499, 514, 518. В pooled snapshot 158
представленных пользователей, а не 192 номинально выбранных UID.

Full catalog — строго cohort-local события до query, не глобальный каталог
Yambda. Сохранённые диапазоны catalog size: 35 580–41 240 (P001),
38 306–43 423 (seed26), 35 614–42 345 (seed27), то есть общий диапазон
35 580–43 423. Разделы отчёта верно называют evaluation фиксированных
K100-trained models на более широком каталоге; training не переобучалось для
Full. Все 1 531 ранее установленных outer-validation queries учтены. Отчёт и
analysis явно сохраняют `final_test_evaluated=false`; P006 test не replay-ился.

## Сверка результатов и transitions

Основная pooled таблица NDCG@20/Recall@20 согласуется с `analysis.json` до
показанных шести знаков. Pooled user-macro даёт одинаковый вес каждому из
158 пользователей; cohort-specific значения также представлены, что
показывает различие направлений между cohorts.

Покрытие целей рассчитано верно: K100 — 357/1 531 (23.3%), Full —
1 341/1 531 (87.6%), прирост 984 query-targets; 190 целей не входят даже в
prior-observed cohort catalog. Для new stratum — 12/553 в K100 и 363/553 в
Full. Таблица target top-20 transitions совпадает с пересчётом rank rows по
паре `(cohort, query_id)` для каждого метода. Она описывает попадание
**целевого трека запроса**, а не пересечение списков рекомендаций; отчёт это
уточняет. Проверка точных scores прежних кандидатов и отсутствие улучшения
ранга прежнего target при расширении каталога согласуются с nested-catalog
инвариантом.

Проверка доверительных интервалов подтверждает, что отчёт показывает
парные user-bootstrap сравнения с Manual UIDC, а snapshot также хранит
paired Full−K100 для каждого метода (23 сравнения в общей схеме: 14
vs Manual и 9 между режимами). Например, для M_UIDC Full−K100 ΔNDCG@20
= +0.002742, 95% CI [−0.001858; +0.007900]; интервал включает ноль.
Интервалы Original_A1/Permutation и другие условны на этих selected cohorts,
повторно использованных development queries и фиксированных моделях. Они не
учитывают selection/multiple-comparison или cohort uncertainty. Отчёт прямо
обозначает интервалы как описательные и не трактует интервал с нулём как
доказательство эквивалентности.

Когортная таблица показывает изменение K100→Full, а не только pooled
среднее. Например, M_UIDC ΔNDCG@20 составляет −0.000904 / +0.008977 /
+0.000086; Original_A1 +0.004985 / +0.005045 / +0.001800. Эти cohort
направления подчёркивают условность pooled результата. Числа признаков и
распределение по U/I/D/C указаны; variance masks между cohorts не скрыта.

## Scope, cost и предел независимой сверки

Результаты соответствуют ограниченному условному training design:
K100 queries с найденной целью дали 349/1 349, 330/1 517 и 336/1 397
положительных train queries; остальные не вошли в classifier fit. Отчёт
объясняет fixed generator bias, отрицательные unmarked candidates,
непереобучение для full catalog, неизвестную историческую доступность mapping,
logged-listen цель и повторное использование development validation. Full
coverage поэтому не следует читать как качество production retrieval или
популяционную оценку.

Runtime table согласуется с сохранёнными статусами: gate и Full завершены
для каждой cohort, peak Full RSS 665.99 / 758.00 / 772.75 MiB, каталоги
покрывают заявленные диапазоны. Fits и per-method search costs раскрыты,
включая permutation prediction budget, shared full-inner fit и Stable prefix
fit. Синтетические проверки и независимый verifier PASS приведены отдельно.

Verifier независимо восстанавливает top-20 rank/score evidence и основные
метрики. Rank выше 20 сохраняется в хешированном CSV, но точный rank выше 20
не восстанавливается независимым пересчётом, поскольку полные Full score
vectors не сохранены. Это явно указано в README snapshot; ограничение не
затрагивает проверку NDCG/Recall@20.

## Вывод

Не обнаружено существенного расхождения основной таблицы, coverage,
target transitions, cohort trends, cost или интерпретации CI с сохранённым
snapshot. В отчёте приведены описательные максимумы K100 и Full, но явно
оговорено, что это не независимо подтверждённые победители; никаких
популяционных claims не делается. P007 остаётся exploratory сравнением
фильтрованных development queries и K100-trained фиксированных моделей.
