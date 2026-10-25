# P007 — review проектирования протокола поиска и full-catalog оценки

## Область и безопасные границы

P007 следует оформить отдельным run ID и отдельными immutable outputs. P006,
его manifest, predictions и one-shot markers остаются закрытой завершённой
контрольной точкой: никакого повторного test чтения, fit или изменения.
Допустимая оценка P007 — только exploratory на прежних outer validation
P001/P005; эти validation уже многократно видели в development, поэтому не
являются независимым подтверждением. Не читать и не генерировать запросы из
резервированного периода P006.

Под «всей выборкой» для этого этапа следует понимать все заранее записанные
eligible evaluation queries трёх существующих cohort (P001, P005 seed 20261026,
P005 seed 20261027) и полный определённый для каждого query доступный каталог
товаров, а не все 50M временных событий либо все UID датасета. Это определение
нужно записать в protocol и freeze до новых feature builds и scoring.

## Что можно повторно использовать без изменений

Существуют три frozen набора запросов/candidate lists K=100: P001 (1 349 train,
499 outer validation) и два P005 (по 1 397/1 517 train и 518/514 validation).
В P005 сохранены полные списки кандидатов и 206-колоночные `features.npy`,
а также baseline `features.npy`; в P001 есть frozen P003 manual matrix и
P004 auto matrix. Их можно использовать **побайтно, только как legacy-K=100
ветку**: обучать/оценивать прежний protocol там, где это требуется, и строить
парные результаты на прежних запросах. Не менять row order, candidates,
labels, splits, masks и старые файлы; перед reuse проверить хеши manifest и
структурный порядок `(query_id, item_id)`.

Старые 100-кандидатные матрицы не являются full-catalog матрицами: в них нет
строк для остальных доступных товаров, а candidate lists и candidate features
были получены по прежнему generator. Их нельзя расширить индексированием,
дублированием строк или подменой item ID. Полный каталог требует нового
causal replay/materialization для всех предусмотренных query timestamps.

Основные source interfaces уже позволяют строить новые строки без labels:
`FeatureEngine` / `AutoFeatureEngine.observe(event)` накапливают только события
прошлого префикса, а `.transform(user_id, item_ids, timestamp)` возвращает
float32 признаки переданного списка товаров. `feature_build()` показывает
потоковый порядок replay и запись memmap; текущая реализация принимает
`query["candidate_items"]`, поэтому full catalog потребует отдельного runner,
который подаёт доступные item IDs батчами и пишет shard/memmap, не меняя старые
функции/артефакты. CandidateGenerator `catalog` формируется из ранее
наблюдённых событий; его `_related` также ограничивает товары известным
каталогом. Эта семантика даёт возможное определение cohort-local catalog,
но не доказывает наличие товара в реальном каталоге сервиса в момент запроса.

## Зафиксировать знаменатель и temporal availability

До запуска определить доступный товар `i` для query `(u,t)` одним из вариантов
и не смешивать их в одной метрике. Рекомендуемая реалистичная граница при
имеющихся источниках — `C(t)`: distinct item IDs, наблюдавшиеся в истории
соответствующей когорты строго до `t`, плюс явное правило treatment target.
Сохранять target в знаменателе, даже если его нет в `C(t)`; такой случай даёт
нулевое end-to-end попадание и отдельно учитывается как unavailable target.
Включать target только потому, что он стал известен в будущем, нельзя. Снимок
artist/album mappings — прежнее неподтверждённое допущение; mapping не следует
трактовать как доказанно доступный на дату query. Данные не позволяют назвать
`C(t)` полным production catalog или проверить item launch/retirement.

Ранжирование полного `C(t)` может содержать повторно прослушанные товары.
Если пользовательская задача подразумевает рекомендации только неслушанных,
нужен отдельный заранее закреплённый eligibility rule; результат такой
фильтрации нельзя сопоставлять как будто это прежний K=100 ranking. Вывести
catalog size на query и cohort, число target отсутствующих/доступных, число
unrankable queries и причины исключения.

## Bridge: модель, обученная на K=100, и новые модели

Рекомендуется сохранить две явно раздельные оси сравнения:

1. **Frozen-model catalog bridge.** Без переобучения взять прежние сохранённые
   модели и применить к full-catalog feature rows для тех же queries, где
   row schema/feature order строго совпадает. Это отвечает узкому вопросу:
   как существующий ranker упорядочивает более широкий scoring universe.
   Сравнивать его full-catalog результаты с прежним K=100 как sensitivity к
   evaluation universe. Записать, что обучение использовало generator-biased
   candidate distribution и negatives из K=100.
2. **Retrained full-catalog-training arm.** Если методам разрешено переобучение,
   получить единый заранее закреплённый training design для всех baseline и
   новых selectors. Создать negatives/rows детерминированным sampling из
   доступного на train query `C(t)` с фиксированными квотами/seed и одинаковыми
   labels, query weights, CatBoost settings и количеством попыток для каждой
   ветви. Нельзя обучать одну ветвь на K=100, другую на полном каталоге или
   менять negative sampling отдельно под метод и называть это чистым сравнением
   feature selection.

Эти arms отвечают разным вопросам и не должны сливаться в один рейтинг. Если
P007 ограничивает training, основной бюджет разумно отдать frozen-model bridge
и ограниченному сопоставимому training subset; отсутствие full-catalog fit
следует явно указывать. Для режима с uniform sampled negatives full-catalog
ранжирование остаётся distribution shift между обучением и оценкой, но уже
контролируемым и общим для вариантов.

## Ограничение вычислений и честность поиска

Full catalog на query может быть на порядки больше K=100. Не материализовать
одну гигантскую RAM matrix. Использовать последовательный causal replay и
фиксированные item-ID чанки, например 1–5k rows, memmap/shard по cohort/query
batch; хранить mapping `query_id,item_id -> score`, hashes, schema и progress.
Все варианты получают идентичные query/item rows и target-blind feature inputs.
Удалить дубликаты item IDs детерминированно, одинаковые timestamps не
разрывать; каждое событие времени `t` скрыто от всех queries с timestamp `t`.
Проверить синтетически chunk-boundary, past-only и target leakage, прежде чем
разрешать scoring. Оценивать ranking streaming top-K с tie-break `score DESC,
item_id ASC`; для exact NDCG@20 достаточно top-20 при уникальном item ID,
но candidate recall/coverage и target rank должны вычисляться по всему
eligible catalog, не только по top-K.

Новые селекторы должны быть сформулированы заранее с reproducible rule,
feature provenance и tie-break. Разделить генерацию/отбор и внешний exploratory
validation: winners выбираются на внутреннем временном selection fold только;
внешние P001/P005 validation используются один раз для заранее объявленных
frozen masks. Уравнять budget по числу evaluated proposals, inner fit attempts,
CatBoost iterations, train rows/negative count, CPU/RAM и wall-clock cap. Учесть
failed attempts в бюджетах. У фиксированного Manual нет поиска; поэтому
сообщать отдельно цену селектора и общую цену, а также не выдавать
сравнение Auto с большими tuning-бюджетами за равный поиск. Никакого выбора
победителя по объединённой outer validation.

## Главные угрозы валидности

- **Train/eval shift:** прежнее обучение условно на target retrieved среди
  100 generator candidates; full catalog содержит иной и гораздо больший
  negative distribution. При одном положительном товаре на query ранжирование
  каталога может резко изменить NDCG/Recall без изменения модели.
- **Candidate availability:** исторический event catalog не равно каталог
  сервиса; mappings статичны и имеют не подтверждённую временную доступность.
- **Repeated exposure:** P001/P005 validation использовались в P002–P006 и
  будут использоваться P007, значит все новые итоги — exploratory.
- **Multiple comparisons:** несколько новых feature-selection методов и
  cohort-specific winners увеличивают selection bias; все методы/бюджеты и
  primary comparison нужно freeze до validation scoring.
- **Unequal input cost:** full catalog multiplies feature computation and
  prediction rows; измерять время, peak RSS, matrix bytes/queries и misses,
  без сравнения throughput только по candidate lists.
- **Query-level weighting:** pooled user-macro давать только представленным
  пользователям равный вес; отдельно показывать query-micro, user counts и
  cohort-specific results. Не смешивать недоступные цели с retrieval misses.

## Рекомендуемая последовательность gate

1. Зафиксировать каталог `C(t)`, cohorts/outer query IDs, masks/novel methods,
   train arms, sampling rules, metrics, equal budgets и hashes в новом P007
   protocol, не меняя исходные P001–P006.
2. До обучения на tiny synthetic fixtures проверить causal replay, равные
   timestamp, target invisibility, future-event perturbation invariance,
   duplicate IDs, chunk stitching и full-list top-K tie-break.
3. Dry-run одного сохранённого query batch каждого размера когорты; измерить
   projected full-catalog rows, disk, elapsed/RSS. Жёстко остановить до больших
   запусков при превышении заранее выбранных лимитов; не урезать cohorts после
   просмотра качества.
4. Reuse старые matrices/models read-only в legacy/bridge arm. Для full catalog
   сформировать новые item rows в новом output namespace, проверить schema и
   hashes. Запускать training только после успешного resource gate и
   независимо верифицировать predictions/metrics.
5. Отчёт отдельно показывает K=100 bridge и full `C(t)` ranking для всех трёх
   когорт, denominators, availability, paired user deltas и стоимость. Вывод
   формулировать как exploratory для этих заранее sampled users/queries, без
   claims о полной популяции или production catalog.

**Итог:** старые P001/P005 query matrices полезны как неизменяемый K=100
контроль и для frozen-model bridge, но не могут представлять полный каталог.
Для full-catalog scoring нужны новые causal item rows по каждому query; основное
методическое решение до запуска — определить temporal item universe и отделить
K=100-trained frozen models от любого нового training design.
