# P003 — протокольный preflight

Дополнение основного агента перед Stage1: автоматическая проверка SHA256
baseline против P001 добавлена; общие source hashes Stage2 также сравниваются.
В `finally` feature build выполняет best-effort flush dirty memmap страниц;
failed output остаётся неполным и не допускается к equivalence/scale.

Статус: protocol подготовлен до результатов; выполнен read-only preflight только Stage 1 runner/config и предоставленных synthetic tests. Optimized feature module и Stage 2/3 runner flow ещё не допущены к независимой проверке. События, реальные replay и benchmarking в этом preflight не читались/не выполнялись.

Зафиксированы три независимые стадии: baseline cProfile; value-equivalence baseline/optimized в свежих процессах с полной `array_equal` и hash; scale-256 только после доказанной эквивалентности. Для первых стадий зафиксированы P001 queries/candidates, границы RAW timestamp и окно/half-life `670390`; для scale сохранены фиксированные границы P001, одинаковый seed выборки и ограничение на выводы для несопоставимых cohorts.

Основные guard-ы перед будущим запуском: события `timestamp >= p85` должны быть отфильтрованы в Arrow до Python-level преобразования; target и strata-поля должны быть изолированы от генератора/feature code; baseline `src/pilot_features.py` должен остаться неизменным; Stage 2 обязан сравнить все 92 float32 columns по одинаковым query/candidate offsets; Stage 3 запускается только после полного exact pass и пишет матрицу инкрементально под лимитами памяти.

## Stage 1 runner preflight

Проверены `src/run_feature_benchmark.py`, `configs/p003_features.json`, `tests/test_feature_benchmark.py`, общие loaders и P003 protocol; реальные events/output не открывались. **Stage 1 baseline profile не имеет найденных блокирующих дефектов.** Config закрепляет `short_window_raw=half_life_raw=670390`, 92 features, query counts 1 349/499, P001 boundaries и лимиты 1100/150 MiB. Loader использует Arrow-filtered development events; `load_frozen_queries` сверяет frozen count/order/time bounds. Baseline path импортирует только существующий `FeatureEngine`; optimized module импортируется условно и нужен только для `--engine optimized`, поэтому Stage 1 не зависит от наличия нового файла.

Feature build получает только `query_id/user_id/timestamp/candidate_items`, пишет такую же label-free structural input list, пишет matrix в `.npy` `open_memmap` инкрементально и проверяет shape, float32, конечность значений. `observe`, `transform` и запись/materialization имеют раздельные phase timers; input load отдельно отражён в `load_seconds`. cProfile включается только вокруг replay/build loop, поэтому его phase/replay времена загрязнены profiler overhead и не должны использоваться как unprofiled speed benchmark. Синтетические тесты, рассмотренные статически, покрывают target event exclusion, отсутствие target labels в feature inputs, memmap shape/values и full-cell comparison/fail gate.

Сверил baseline engine SHA256 с сохранённым P001 metrics: ожидаемый `274d7e717226bcae817873e3afc8509d7f5d4078b894547f178e29b5fb0b9dc7`, текущий `src/pilot_features.py` имеет тот же hash. Неблокирующее усиление provenance: runner записывает исходный SHA, но сам не отклоняет build, если он не совпал с P001 source SHA; перед Stage 2 полезно сделать эту проверку fail-closed или явно сохранять ручной hash gate в run record. Сейчас фактический baseline hash совпадает.

На Stage 1 runner создаёт failed status в `finally`; memmap объект освобождается при выходе из build/process, однако при исключении во время build явный `flush/close` не выполняется и матрица должна считаться незавершённым failed artifact. Это допустимо только при запрете её использовать как complete output и при новом subrun после исправления. `verify_builds` сравнивает матрицы mapped chunks, сравнивает все float32 cells при успехе, проверяет file/value hashes, schema и row ordering; Stage 2 memory/time measurements всё ещё нужно независимо проверить до запуска. Scale preparation/exception-safe writer остаётся вне этого preflight.

Никакие speedup, экономия памяти или quality improvement пока не измерены. Этот preflight не разрешает Stage 2 или Stage 3; для них требуется отдельная проверка optimized module, memmap/lifecycle, full equivalence и scale-preparation guards.

## Stage 2 optimized engine и Stage 3 cleanup — read-only review

Проверены `src/optimized_features.py`, `tests/test_optimized_features.py`,
`tests/test_feature_scale.py`, `src/scale_data.py` и текущий
`validate_equivalence_gate` в `src/run_feature_scale.py`; реальные events и
матрицы не читались. По сообщению основного агента, полный набор из 54 тестов
прошёл; здесь тесты не запускались.

**Допуск Stage 2 optimized exact comparison: да**, при неизменности source/config
хешей после свежей baseline сборки. Оптимизированный engine наследует причинное
состояние и helpers базового engine; query-local caches сбрасываются на каждом
вызове transform. Проверенные пути сохраняют исходный порядок колонок, entity
порядок и арифметику агрегатов; есть синтетическое покрытие общих entities,
граничного окна, отсутствующих mappings/истории, пустого candidate списка,
временных равенств, underflow и повторного вызова после обновления состояния.
Этот допуск разрешает только exact comparison всех ячеек и структур. Сам по себе
он не разрешает Stage 3: перед ним gate должен пройти fail-closed проверку
точных source keysets и frozen P001 input pins, которые основной агент ещё
добавляет.

В `scale_data.py` закрытие уже открытых event и mapping Parquet writers в
`finally` выполнено. Однако `_write_failed()` охватывает только исключения
event-pass: отмена монитором в UID population pass либо ошибка/остановка при
сканировании mappings оставит частичный `data/` без локального `FAILED.json`.
Верхний `run_feature_scale` запишет общий `run_status.json` как failed, так что
такой результат нельзя принять за complete, но локальная подготовка не
самодокументирует failure. До Stage 3 следует охватить весь `_prepare` общим
failure marker, сохранив уже имеющееся закрытие writers и запрет повторного
использования каталога.

## Финальный Stage 3 preflight

Основной агент сообщил об усилении provenance gate и regression tests для
mutation случаев. Повторно проверены только код gate/wrapper, текущие gate и
run metrics, а также хеши текущих source/config/protocol файлов; events и
матрицы не читались и не пересчитывались. Текущие source hashes совпадают с
метриками обеих сборок; config SHA256 —
`b8cc706d79691be41b413708bc4d04148a7fec91cd3a2b90248e5f5f99caf3f1`, protocol
SHA256 — `7f5cbc2ca269d3f8fcde0e8decde8f5ba1e54bb6b6d1703fd36a9161783aafa7`.

**Stage 3 preflight blockers сняты.** Gate имеет статус `pass`: сравнивались все
17 001 600 float32 cells матриц shape `184800 x 92`; `numpy.array_equal`, raw
values SHA256 и file SHA256 совпадают. Обе сборки complete и unprofiled; совпадают
source keys, frozen P001 input hashes, protocol SHA, structural input и schema
hashes, а baseline source привязан к сохранённому P001. `scale_data._prepare`
теперь оборачивает все стадии и пишет failure marker на исключении; открытые
writers закрываются локальными `finally` блоками. Каталога Stage 3 на момент
этой read-only проверки ещё нет.

В свежем unprofiled парном запуске baseline replay занял 23,410649 с
(observe 9,323760 с; transform 13,809617 с), optimized — 52,210886 с
(observe 17,195409 с; transform 34,505170 с). Это отдельный сырой timing
результат одной пары запусков, не основание заявлять ускорение; он не меняет
exact value-equivalence. Тяжёлые сборки и обработка событий в preflight не
повторялись.
