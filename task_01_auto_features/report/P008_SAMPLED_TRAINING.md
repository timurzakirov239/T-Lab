# P008 — negative sampling и дополнительные warm training queries

**Exploratory development-серия.** Повторно использованы все 1531 прежних validation queries в трёх непересекающихся когортах; объединённые метрики представлены по 158 пользователям с запросами. P006 test не открывался. Никакой результат этой серии не является независимым подтверждением или популяционным выводом.

## Условия и методы

Все новые arms используют неизменные UIDC66 и cohort-specific P007 A1 masks. Базовые P007 models переиспользованы. A1 masks сами заморожены ещё в P004/P005; после P007 A1 выбран дополнительным benchmark, поэтому это сравнение exploratory. Новый feature search и hyperparameter tuning не выполнялись. Primary — Full user-macro NDCG@20 для HybridHit_UIDC против Baseline_UIDC.

Baseline — исходные K100 hit-query training blocks. UniformHit использует те же hit queries с 99 равномерными cohort-catalog negatives. HybridHit заменяет 49 из них teacher-hard negatives, сохраняя общие 50 uniform draws. HybridWarm добавляет train queries, чьи цели известны cohort-wide до query; shared hit-query blocks остаются идентичны HybridHit. Здесь warm/cold относится к каталогу всей когорты (`target_known_before`), а repeat — отдельно к user-level прошлым прослушиваниям. Новые пользователю товары могут входить в warm arm.

Teacher — P007 M_UIDC, обученный на той же train выборке; hard mining in-sample и может наследовать её смещения. Он не видел validation/test labels. Все negatives являются unmarked объектами, а не доказанно нежелательными реакциями. Обучение условно на наличии target; это не training на всём каталоге. Одна и та же модель каждого arm оценивается в K100 и Full режимах.

## Pooled rank metrics

User-macro: сначала усредняются запросы пользователя, затем представленные пользователи; когорты не получают равный фиксированный вес.

| Arm | K100 NDCG@20 | K100 Recall@20 | Full NDCG@20 | Full Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| Baseline_UIDC | 0.066412 | 0.148418 | 0.069154 | 0.159177 |
| Baseline_A1 | 0.068791 | 0.150316 | 0.072727 | 0.164241 |
| Uniform_UIDC | 0.059228 | 0.145886 | 0.060863 | 0.154747 |
| Uniform_A1 | 0.056325 | 0.145253 | 0.057188 | 0.151582 |
| HybridHit_UIDC | 0.063612 | 0.140190 | 0.068288 | 0.152215 |
| HybridHit_A1 | 0.067509 | 0.145886 | 0.071635 | 0.157700 |
| HybridWarm_UIDC | 0.036922 | 0.079747 | 0.009811 | 0.018354 |
| HybridWarm_A1 | 0.028372 | 0.063291 | 0.010174 | 0.020253 |
| Popularity | 0.006793 | 0.014557 | 0.006793 | 0.014557 |

## Primary comparison

HybridHit_UIDC − Baseline_UIDC в Full: ΔNDCG@20 = **-0.000867**, descriptive paired-user 95% CI [-0.005067; +0.003618], N=158. Это заранее заданный primary contrast; интервал условен на этих пользователях, training/query protocol и фиксированном teacher. Он не учитывает повторное использование validation или неопределённость cohort sampling.

Все 29 pooled paired contrasts (каждый arm к соответствующему baseline, sampling/warm additions и K100→Full) перечислены в [analysis.json](p008_snapshot/analysis.json); там же сохранены cohort-specific metrics и paired estimates. CI описательные, без поправки за несколько сравнений. Нулевой интервал не доказывает эквивалентность.

## Eligibility, catalog coverage и переходы

| Cohort | Train queries | Hit in original K100 | Warm in cohort C(t) | Cold to cohort C(t) | Training queries (Uniform/HybridHit/HybridWarm) |
| --- | ---: | ---: | ---: | ---: | --- |
| p001 | 1349 | 349 | 1153 | 196 | 349 / 349 / 1153 |
| p005_20261026 | 1517 | 330 | 1282 | 235 | 330 / 330 / 1282 |
| p005_20261027 | 1397 | 336 | 1209 | 188 | 336 / 336 / 1209 |

| Mode | Targets present in catalog / queries | New-to-user targets present / queries |
| --- | ---: | ---: |
| K100 | 357 / 1531 (23.3%) | 12 / 553 |
| Full | 1341 / 1531 (87.6%) | 363 / 553 |

Для каждой модели парные разности Full−K100 сохранены в paired analysis; ранги отдельных целей находятся в ranks.csv. Топ-20 метрики считают один logged listen target, а не полезность всех unmarked кандидатов. Population/catalog coverage зависит только от каталога и одинаково для rankers внутри режима. Full оценивает те же замороженные модели на cohort-local C(t); это distribution shift по сравнению с K100 training.

## Cohort-specific Full NDCG@20

| Arm | P001 | Seed26 | Seed27 |
| --- | ---: | ---: | ---: |
| Baseline_UIDC | 0.074222 | 0.071102 | 0.062234 |
| Baseline_A1 | 0.083331 | 0.064675 | 0.070376 |
| Uniform_UIDC | 0.068424 | 0.056659 | 0.057648 |
| Uniform_A1 | 0.069569 | 0.059957 | 0.042273 |
| HybridHit_UIDC | 0.075775 | 0.069513 | 0.059717 |
| HybridHit_A1 | 0.084579 | 0.064508 | 0.066061 |
| HybridWarm_UIDC | 0.011782 | 0.010612 | 0.007076 |
| HybridWarm_A1 | 0.011607 | 0.016511 | 0.002432 |
| Popularity | 0.007139 | 0.008401 | 0.004845 |

## Разбор sampling и расширения train

| Full contrast | Δ user-macro NDCG@20 | Descriptive 95% CI |
| --- | ---: | --- |
| HybridHit_UIDC − Uniform_UIDC | +0.007425 | [-0.000766; +0.016387] |
| HybridHit_A1 − Uniform_A1 | +0.014446 | [+0.005840; +0.024197] |
| HybridWarm_UIDC − HybridHit_UIDC | -0.058477 | [-0.072445; -0.045012] |
| HybridWarm_A1 − HybridHit_A1 | -0.061460 | [-0.075807; -0.047818] |

Uniform уступил исходному baseline при обеих masks. HybridHit приблизился к baseline, но основной прирост не подтверждён: primary point estimate отрицателен, CI включает ноль. Более высокие point estimates HybridHit относительно Uniform не означают превосходства над baseline; для UIDC интервал этого вторичного сравнения тоже включает ноль.

HybridWarm увеличил train с 1015 до 3644 запросов, сохранив sampling на общих запросах, но резко ухудшил итоговое ранжирование при обеих masks и во всех трёх когортах. Это наблюдаемый эффект конкретного расширения и фиксированного training recipe; он не доказывает вред дополнительных данных вообще. Состав запросов и их доли одновременно изменились. Причины — изменение сигналов релевантности, sampling bias, ложные negatives или настройки обучения — требуют отдельных проверок, а не объявляются установленными.

| Full arm | Repeat NDCG@20 | Repeat Recall@20 | New-to-user NDCG@20 | New-to-user Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| Baseline_UIDC | 0.109937 | 0.256436 | 0.002356 | 0.004176 |
| Baseline_A1 | 0.117008 | 0.260206 | 0.008418 | 0.009751 |
| Uniform_UIDC | 0.099938 | 0.249462 | 0.000991 | 0.003798 |
| Uniform_A1 | 0.094943 | 0.248611 | 0.002923 | 0.011546 |
| HybridHit_UIDC | 0.106781 | 0.235152 | 0.004196 | 0.012793 |
| HybridHit_A1 | 0.113052 | 0.253180 | 0.013328 | 0.021088 |
| HybridWarm_UIDC | 0.008981 | 0.018580 | 0.012332 | 0.015627 |
| HybridWarm_A1 | 0.008179 | 0.018735 | 0.010822 | 0.016694 |
| Popularity | 0.010470 | 0.025796 | 0.001312 | 0.001652 |

Repeat: 978 запросов/146 пользователей; new-to-user: 553/147, из них 363 цели доступны в Full. Каждый столбец strata усреднён только по представленным в этой группе пользователям; их нельзя складывать для получения overall. Падение HybridWarm особенно заметно на повторах. Отдельные улучшения new-to-user не компенсируют overall падение. Coverage внутри каждого режима остаётся одинаковым для всех моделей: изменение обучения не меняло каталог.

## Masks, fits и стоимость

| Cohort | UIDC columns | P007 A1 columns | New fits | Prepare/mining s | Fit s | Gate s | Full s | Peak RSS max MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p001 | 66 | 16 | 6 | 185.39 | 14.63 | 108.93 | 282.24 | 675.45 |
| p005_20261026 | 66 | 62 | 6 | 196.34 | 17.11 | 113.93 | 327.09 | 664.74 |
| p005_20261027 | 66 | 8 | 6 | 164.78 | 11.88 | 111.12 | 292.71 | 675.88 |

Итого: **18 новых fits** (6 на когорту); baseline модели и teacher переиспользованы. Prepare/mining wall time включает фильтрацию/first-seen inventory, feature materialization и teacher-pool scoring, поэтому это стоимость этапа, а не отдельный чистый benchmark teacher inference. Все cohort statuses должны оставаться в пределах 1100 MiB RSS и 150 MiB reserve.

Exact validation gate проверяет UIDC66 features на всех исходных K100 rows; Full scoring проверяет их в chunks2048 и требует parity с P007 для Baseline_UIDC, Baseline_A1 и Popularity. Независимая проверка и portable snapshot status — `verified`/`PASS`; teacher-pool scores повторно не рассчитаны независимым teacher inference, что отражено в protocol. Полные score vectors каталога не сохранялись: exact ranks за пределами top20 защищены hashes и проверками границ/вложенности, но независимо из всех scores не восстановлены.

## Ограничения и вывод

Все queries — ранее использованная development validation; A1 выбран дополнительным benchmark после P007, при неизменных ранних masks. Поэтому различия и intervals exploratory, не являются независимым test. Training имеет shortlist-derived baseline и условные positives; sampled unmarked items не доказанные negatives. Teacher in-sample, mapping time availability остаётся предположением, а cohort catalog не равен каталогу всего сервиса. P006 не открывался, работа 2 не начиналась.

P008 не устанавливает победителя. Primary contrast и остальные эффекты следует использовать как ориентиры для следующего заранее зафиксированного эксперимента с независимым временным holdout либо иной схемой validation. Human source verification остаётся pending.

## Материалы

- [Протокол](../configs/P008_PROTOCOL.md), [pre-fit review](../research/P008_PROTOCOL_REVIEW.md).
- [Компактный snapshot](p008_snapshot/README.md), [reproducibility](P008_REPRODUCIBILITY.md), [human review checklist](P008_HUMAN_REVIEW.md).
- [Рецензия результатов](../research/P008_RESULT_REVIEW.md), [изолированная проверка](p008_isolated_verification.json), [сохранность 72 файлов P006](p008_frozen_input_verification.json).
- Portable rank analysis содержит pooled и cohort metrics, а также все paired comparisons. Полные rank score vectors и sampled training payload не включены.
