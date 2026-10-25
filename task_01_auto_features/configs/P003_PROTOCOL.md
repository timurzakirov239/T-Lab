# P003 — value-preserving feature pipeline optimization

Статус: protocol до измерений и оптимизаций. Цель — проверить время/память вычисления уже фиксированных признаков и сохранить значения побитно идентичными baseline. P003 не является поиском признаков или модельным экспериментом. Не обучать CatBoost, не оценивать test и не трактовать Stage 3 как сравнение качества 64 против 256 пользователей.

## Общие замороженные условия

- Для Stage 1–2 используются cohort и exact 1 848 P001 queries: 1 349 train + 499 validation, в исходном порядке, с сохранёнными P001 candidate IDs/порядком. Входы признакового расчёта содержат только user, query timestamp и candidates; target, label, `target_is_repeat`, `target_known_before` и остальные target-derived поля не передавать в feature code.
- Границы P001 фиксированы в RAW units: query start `9 752 280`, train end `20 125 775`, validation/p85 end `23 380 165`. Не пересчитывать квантильные границы. История запроса строго `< t`; равные timestamp не видны до следующего момента.
- Использовать `short_window=670390` и `half_life=670390` в RAW units. Фиксированное значение подтверждается формулой P001: floor((train_end − timestamp_min) / 30); не пересчитывать его на 256-user cohort.
- Stage 1–2 используют базовый P001 FeatureEngine (U/I/D/C/X, 92 float32 столбца) и один оптимизированный новый module. Не менять `src/pilot_features.py`; улучшение должно быть изолировано отдельным source file и frozen config.
- Строить rows в неизменном порядке query IDs, затем candidate IDs. Сохранять query IDs, candidate offsets/sizes и candidate IDs рядом с матрицей. Target используется только для запроса/сопоставления после формирования candidate/feature inputs и не входит ни в один столбец.
- Все stages записывают run ID, status, команду, config/protocol/source hashes, input hashes, фазовые времена, размеры матриц, peak RSS, минимум доступной памяти и ошибку/stop reason. Завершённый run не перезаписывать; повтор после исправления — новый subrun ID.

## Stage 1 — baseline profile

Один baseline replay всех 1 848 P001 queries и фиксированных candidates, `short_window=half_life=670390`. Запустить cProfile ровно один раз. Явно измерить отдельно: чтение уже подготовленных development events/query/mappings; `observe`/обновление истории; `transform` по кандидатам; materialization в output memmap/файлы. Не включать цели в profile inputs. Записать `cProfile` output, фазовые секунды и фактический размер матрицы. На P001 candidates верхний ожидаемый размер — 184 800 × 92 × float32 (68 006 400 байт), если все candidate lists имеют 100 элементов; фактическую форму брать из сохранённых offsets, не предполагать.

## Stage 2 — equivalence и свежие процессы

После Stage 1 baseline и оптимизированный модуль запускаются каждый в своём новом subprocess и выполняют по одному replay на тех же 64-user development inputs, frozen query file, candidates, окнах и timestamp порядке. Сравнение включает:

1. точное равенство query IDs, candidate IDs, offsets/sizes и feature column names/order;
2. полную `numpy.array_equal` по всем float32 ячейкам baseline и optimized matrices;
3. одинаковый SHA256 сырых сериализованных матриц и запись формы/dtype/bytes.

Любое несовпадение — fail Stage 2; не сообщать speedup как value-preserving и не продолжать к Stage 3 до новой версии/протокола. Отдельно показать elapsed и фазовые времена baseline/optimized, peak RSS и throughput. Не обещать ускорение заранее; cProfile overhead Stage 1 не сравнивать с unprofiled Stage 2.

## Stage 3 — ограниченный scale run на 256 пользователях

Запускать только если equivalence Stage 2 пройден. Равномерно выбрать 256 уникальных UID из офлайн-популяции Yambda-50M с seed `20261025`; UID-популяцию разрешено определить отдельным Arrow pass только по колонке uid на полном источнике, без фильтрации по активности. Тот же seed не гарантирует, что набор из 64 P001 пользователей является подмножеством нового набора; записать фактическое пересечение и не трактовать данные как nested sample.

Сохранить исходный опубликованный источник и существующие P001 prepared files; не обрезать историю выбранных пользователей по строкам. В Python-level per-row logic передавать только выбранных пользователей и event rows строго ниже frozen p85. Применить PyArrow batch filter `timestamp < 23 380 165` **до** `to_pydict`, Python loop, статистик, query selection или replay. Чтение смешанной Parquet row group физически допустимо, но строки p85+ не декодировать в Python и не анализировать. Временные границы не пересчитывать по 256 пользователям. Mapping snapshot фильтровать по item IDs development-каталога из выбранных пользователей; читать только нужные mapping columns.

Строить до 30 train и до 10 validation однозначных listen queries на пользователя из development интервалов, воспроизводимо с тем же seed. Идентичные uid/timestamp группы с несколькими listen исключать. Сначала выполнить фильтр p85; цели target и query strata не передавать генератору/feature module. Кандидаты: фиксированный P002 repeat-heavy generator, K=100, его неизменённые параметры и past-only правила. Порядок query/candidates и candidate lists сохранять до расчёта признаков.

Сделать один optimized feature build, без CatBoost и без grid search; писать float32 матрицу инкрементально в NumPy `.npy` memmap, не собирать полный плотный массив в RAM. Проверить конечность значений, форму/offsets, число features=92, SHA256 сырых float32 bytes и файла с заголовком. Главные показатели Stage 3 — throughput, elapsed, память и объём сохранённых артефактов. Candidate recall можно сохранить только как описательный sanity-check; его нельзя сравнивать с P001/P002 как эффект модели/признаков или доказательство качества на большей популяции.

## Run IDs, ресурсы и остановка

Использовать раздельные output IDs `p003_profile`, `p003_equivalence_baseline`, `p003_equivalence_optimized`, `p003_scale256`. Все stages имеют лимит RSS 1 100 MiB и минимум свободной системной памяти 150 MiB; одновременно один тяжёлый процесс. Останавливать безопасно при превышении лимита или ошибке, записывать failed/stopped status и сохранять уже созданный checkpoint/memmap metadata. Не возобновлять завершённый или partial run in-place; новый запуск получает новый ID. Stage 3 не начинать после ошибки или отсутствия успешной проверки полной эквивалентности.

Для сравнения scale фиксировать source/config/query/input hashes и cohort overlaps. Разные cohorts и query sets (64 против 256) делают quality-разницу несопоставимой; отчёт ограничить ресурсами и воспроизводимостью, без вывода о выигрыше рекомендаций.
