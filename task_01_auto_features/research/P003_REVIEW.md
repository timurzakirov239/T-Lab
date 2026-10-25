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
