# P008 — обучающие конкуренты и условный отбор запросов

Статус до fits: отдельный exploratory development-протокол. Пользователь поручил продолжить предложенный после P007 эксперимент. P001–P007, их исходники и outputs не перезаписываются. P006 test не открывается.

## Вопрос и фиксированные условия

Проверяем влияние негативного sampling при неизменных признаках, затем отдельно включение дополнительных warm train queries. Три прежние непересекающиеся когорты, те же все 1 531 validation queries/158 представленных пользователей, границы P007. Train events фильтруются строго до train_end; full evaluation использует development history строго до validation_end. История каждого запроса только timestamp<t, ties невидимы. Каталог C(t) — все ранее наблюдавшиеся cohort items; повторы разрешены. Это не весь Yambda-50M и не независимый holdout.

Два неизменных набора: UIDC66 и cohort-specific A1 masks P007 (16/62/8 колонок, все внутри первых 66). Выбор A1 для дополнительного сравнения сделан после P007 и описывается как exploratory. Нового поиска признаков и hyperparameter tuning нет. CatBoost P007:120 iterations,depth5,learning_rate=.07,l2=5,Balanced,2threads,seed20261025. Primary — Full user-macro NDCG20 HybridHit_UIDC против Baseline_UIDC; остальные сравнения дополнительные. Интервалы описательные, без поправки на множественный поиск.

## Обучающие arms

1. Baseline: переиспользуются P007 M_UIDC/Original_A1 models. Их original K100 train blocks включали только запросы с target в shortlist (349/330/336). Новых baseline fits нет.
2. UniformHit: тот же набор hit train queries; target +99 равномерных различных объектов из C(t) без target.
3. HybridHit: те же hit queries; target +50 uniform negatives +49 teacher-hard negatives.
4. HybridWarm: тот же hybrid sampling, но все исходные train queries с target в C(t). Cold targets исключаются с явным счётчиком; первоначальные hit queries сохраняют ровно те же blocks, что HybridHit. Так отличие HybridWarm от HybridHit — добавление warm запросов, не смена их negatives.

Warm/cold здесь означает известность объекта в каталоге **всей когорты**, а не конкретному пользователю. Это та же семантика `target_known_before` в run_pilot.py:243 и run_robustness.py:120. Новые пользователю треки допускаются, если ранее наблюдались у другого пользователя. Повторное прослушивание пользователем задаётся отдельно `target_is_repeat`.

Для каждой warm train query: независимые детерминированные SHA256 seed streams, зависящие от cohort/query_id и seed20261103; draw99 uniform без target. Первые50 draws служат uniform частью hybrid. Hard pool — union исходных K100 и отдельного равномерного draw256 (или всех доступных, если меньше256) из C(t) без target и uniform50. Pool sorted itemID. Teacher — неизменный P007 M_UIDC classifier соответствующей когорты. Берём49 наибольших teacher scores, tie scoreDESC/itemIDASC. Дубликаты и пересечение uniform/hard запрещены. Все новые train blocks сортируются по itemID, y задаётся сравнением с target. Недостаточный каталог/негативы — ошибка, без тихого fallback.

Teacher обучен на этой же train-выборке; это in-sample hard mining, который может наследовать её смещения. Он не использует validation/test labels. Его weights обучены до validation_start; mining является процедурой offline fit, а не историческим онлайн-рекомендателем внутри train. Unmarked negatives не являются доказанно нежелательными объектами. В train target явно включается как известный положительный label; в evaluation target не вставляется.

## Реализация, проверка и порядок

Новые файлы P008, output artifacts/p008_sampled_training/{cohort}. Разделены prepare/fit/gate/full. Prepare строит features только для train, двумя read-only матрицами: UniformHit (hit_queries×100×66) и HybridWarm (warm_queries×100×66); HybridHit использует ровно hit subset второй матрицы. Для mining используется исходный AutoFeatureEngine с первыми66 specs; никаких A2 columns для этих masks не требуется. На всех исходных train K100 rows проверяется byte equality с frozen features.npy[:,:66].

До первого нового validation scoring завершить и зафиксировать все18 новых fits (3arms×2masks×3cohorts). Все masks/blocks/teacher SHA фиксируются до fits. Затем standalone exact gate всех validation K100 features66, далее full scoring C(t) чанками2048 с FullCatalogFeatures на той же66-spec grammar. Full сохраняет K100 scores и проверяет exact features, nested rank и reference Baseline/Popularity parity с P007. Один и тот же новый model используется на K100 и Full. Stage output не перезаписывается; ошибки/partial attempts сохраняются с отдельным ID.

Каждый sampled matrix block содержит признаки именно новых sorted [target+99 negatives] item IDs, а не прежний порядок K100. Они материализуются из causal engine по соответствующим новым IDs; frozen P007 matrix служит только reference gate исходных K100. Teacher pool также получает новые causal feature rows. Сохранённый records.jsonl задаёт отображение query→item IDs→target label и offsets. Лимит fits —6 на когорту,18 суммарно; сохранённые P007 baseline models не входят в число новых fits.

Сохраняются: source/input/model SHA, eligibility counts, train blocks, teacher-pool IDs/scores и выбранные hard IDs, item-first-seen inventory для независимой проверки sampling causal catalog, labels/counts, K100 prediction vectors, Full target rank/top20 score evidence, progress/time/RSS. Новые сырые матрицы/модели/истории остаются вне Git. Независимый verifier восстанавливает sampling из inventory/seed, проверяет hard ordering по recorded teacher scores, training eligibility/subset и ranks/metrics/hashes. Teacher pool scores являются сохранённым evidence, не независимым повтором teacher inference; это ограничение раскрывается. Full exact ranks>20 проверяются на bounds/hash, top20 — по evidence, полные score vectors не сохраняются.

Ресурсы: свежие последовательные процессы; RSS1100MiB/reserve150MiB. Допускается выбор scalar/vector implementation до fits по exact value gate и runtime, без изменения sampling или критериев. Иначе конфигурация фиксирована до запусков. Если лимит нарушен, фиксируем отрицательный результат, не выдаём partial за завершённую оценку.

## Анализ

Все9 labels: Baseline_UIDC, Baseline_A1, Uniform_UIDC, Uniform_A1, HybridHit_UIDC, HybridHit_A1, HybridWarm_UIDC, HybridWarm_A1, Popularity. NDCG/Recall@10/@20, query-micro/user-macro, repeat/new/warm/cold. Coverage каталога совпадает P007; сравниваем ранжирование на тех же целях. Парные user bootstrap2000 seed20261104: каждый arm к соответствующему baseline, HybridHit−Uniform, HybridWarm−HybridHit, K100→Full. Cohort-specific/pooled, actual training sizes/costs и цена mining сообщаются отдельно. Человеческая проверка ожидается; работа2 не начинается. После результатов обсуждаем дальнейшие действия.
