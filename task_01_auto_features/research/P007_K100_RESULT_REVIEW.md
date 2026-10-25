# P007 — read-only review K100 outer-validation результатов

Проверены только `artifacts/p007_expanded_search/{cohort}/run_status.json`,
`selection_frozen.json`, `metrics100.json` и
`report/p007_k100_development.json`. Все три K100 search runs завершены;
полный доступный catalog остаётся pending. `final_test_evaluated=false`.
P006 test не открывался и не переоценивался; здесь приведены только
exploratory development результаты.

## Знаменатели и сверка pooled summary

| Когорта | Train queries в frozen наборе | Outer validation queries | Пользователи outer validation | Успешные fits |
| --- | ---: | ---: | ---: | ---: |
| P001 | 1 349 | 499 | 52 | 38 |
| P005 seed 20261026 | 1 517 | 514 | 53 | 38 |
| P005 seed 20261027 | 1 397 | 518 | 53 | 38 |
| **Итого** | **4 263** | **1 531** | **158** | **114** |

Outer validation counts — 499/514/518. В pooled report для каждой модели
получается 1 531 query и 158 пользователей. Я пересчитал pooled user-macro
NDCG@20 и Recall@20 как взвешенное среднее cohort user-macro по фактическому
числу пользователей (52/53/53); значения совпадают с report с точностью
машинного округления. Это уместно здесь, поскольку cohort memberships
попарно disjoint. Это сверка сохранённых метрик, не повторная оценка
predictions.

## Выбранные masks и стабильность состава

| Метод | Число колонок P001 / P005-26 / P005-27 | Попарный Jaccard feature sets, минимум |
| --- | ---: | ---: |
| M_I | 12 / 12 / 12 | фиксированный контроль |
| M_UIDC | 66 / 66 / 66 | фиксированный контроль |
| Original A1 | 16 / 62 / 8 | 0.129 |
| Original A2 | 8 / 197 / 64 | 0.041 |
| Random206 | 16 / 16 / 8 | 0.000 |
| Stable206 | 32 / 64 / 16 | 0.091 |
| Permutation206 | 8 / 64 / 64 | 0.014 |
| ForwardGroups66 | 10 / 20 / 54 | 0.185 |

Jaccard рассчитан по frozen spec IDs для трёх cohort masks, по всем трём
попарным сочетаниям; это описывает изменчивость состава, а не качество или
статистический тест. A1/A2 были ранее выбранными cohort-local masks; их
включение — reuse контрольных masks, не новые P007 search trials.

## Фактический fit/search budget

На каждую когорту записано 38 успешных fits: 30 inner-stage и 8 outer
refits. Inner состав: один общий `common_full_inner`, 10 Random206 proposals,
один Stable early-prefix fit, четыре дополнительных Stable206 subset fits,
четыре дополнительных Permutation206 subset fits и десять ForwardGroups66
fits. Общая модель inner используется повторно для importance/selection, а не
обучается заново для Stable и Permutation. Outer refits покрывают восемь
final masks; модели с совпадающими mask alias-ятся и используют общий fit.
Итого по трём когортах — 114 fits (90 inner-stage, 24 outer-stage).

Это не равные вычислительные бюджеты между методами: Random и Forward имеют
по 10 subset fits; Stable и Permutation используют по четыре subset fit и
общий importance model, Stable дополнительно обучает ранний prefix. Permutation
также делает feature-wise query-block perturbation/scoring на inner selection
(206 perturbations по наличным признакам), не являющийся отдельным CatBoost
fit. Manual controls не тратят P007 search fits, а Original A1/A2 повторно
используют ранее выбранные masks. Поэтому результаты сравнивают процедуры
отбора при фактически разных search costs, а не при одинаковом общем бюджете.

Inner fitting оставалось условным на найденной цели: на common full-inner fit
включено 244/943 P001 запросов, 225/1 061 P005-26 и 247/977 P005-27;
остальные inner-fit queries без цели в K100 не входят в classifier training.
Внешняя оценка сохраняет все 1 531 query. Число кандидатов и retrieval
coverage одинаковы между rankers, так как K100 кандидатные списки общие.

## Наблюдённые K100 метрики

Pooled user-macro значения сверены с сохранённой report-сводкой:

| Вариант | NDCG@20 | Recall@20 | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.006793 | 0.014557 | 0.005392 | 0.008861 |
| M_I | 0.061830 | 0.134494 | 0.049954 | 0.087342 |
| M_UIDC | 0.066412 | 0.148418 | 0.054834 | 0.102848 |
| Original A1 | 0.068791 | 0.150316 | 0.057216 | 0.105063 |
| Original A2 | 0.068403 | 0.152848 | 0.055391 | 0.101582 |
| Random206 | 0.065199 | 0.142722 | 0.053400 | 0.096203 |
| Stable206 | 0.065453 | 0.150316 | 0.054108 | 0.105380 |
| Permutation206 | 0.068936 | 0.154747 | 0.056160 | 0.104430 |
| ForwardGroups66 | 0.065688 | 0.147785 | 0.054651 | 0.104114 |

На уровне когорт направления и величины различаются. Например, Auto A1
NDCG@20 составляет 0.078346 / 0.059630 / 0.068576, а Permutation206 —
0.084733 / 0.059682 / 0.062691. Эти cohort scores — повторно использованная
outer development validation, поэтому не являются независимыми подтверждениями.

Все rankers используют один K100 candidate universe: цель присутствует в
кандидатах в 357/1 531 запросах (23.32%), отсутствует в 1 174. В 190 cold
queries цель ещё не известна в истории и retrieval coverage равна нулю; у new
stratum target coverage составляет 12/553 (2.17%). Следовательно, metrics
ограничены фиксированным retrieval и преимущественно доступными warm/repeat
целями; это не full-catalog оценка.

## Стоимость

| Когорта | Wall time | Peak process RSS | Минимум свободной системной RAM |
| --- | ---: | ---: | ---: |
| P001 | 113.630 с | 313.50 MiB | 619.68 MiB |
| P005 seed 20261026 | 122.216 с | 370.26 MiB | 724.22 MiB |
| P005 seed 20261027 | 89.870 с | 315.94 MiB | 802.98 MiB |

Все три search runs уложились в сохранённые limits 1 100 MiB RSS и 150 MiB
reserve; суммарное wall time 325.716 с. Это стоимость K100 search/refit
runner, не стоимость pending full-catalog replay.

## Ограничения интерпретации

- Random/Stable/Permutation masks выбраны по внутреннему selection fold, а
  ForwardGroups перебирает до десяти group additions; многократный поиск
  создаёт selection optimism. Bootstrap/внешние validation не превращают этот
  сравниваемый набор в новый независимый test.
- Состав большинства выбранных masks меняется между когортами; отсутствие
  стабильного набора и разная search cost затрудняют интерпретацию как
  воспроизводимого feature discovery.
- Original A1/A2 masks уже были выбраны в P004/P005; сравнивать их с новыми
  P007 selectors следует с оговоркой о разной истории поиска и exposure.
- Обучение ограничено прежним K100 generator и queries с найденной target;
  positives 244/943, 225/1 061 и 247/977 для inner selection setup. Full-catalog
  перенос сохранённых моделей оценивается отдельно и пока pending.
- Cohorts — три пользовательские подвыборки одного датасета и одного
  временного протокола. P001/P005 validation уже использовалась в предыдущих
  этапах, так что значения exploratory; статистическое популяционное
  превосходство или победитель не устанавливаются.

**Вывод:** отчётная pooled сводка согласуется с cohort `metrics100.json`, а
поиски и refits завершены в записанном бюджете. Результаты показывают разные
фиксированные K100 оценки и заметную вариацию masks между когортами; по ним
не выбираю метод-победитель. Полный catalog ещё pending и может изменить
наблюдаемое ранжирование при тех же K100-trained models.
