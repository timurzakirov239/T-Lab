# P005 — read-only preflight

Проверены `configs/P005_PROTOCOL.md`, `configs/p005_robustness.json`,
`src/robustness_data.py`, `src/run_robustness.py` и доступные P004 snapshot
provenance files. Реальные источники событий, feature builds и модели не
читались и не запускались. Тесты не выполнялись.

## Что проверено

Методический план закрывает прежние вопросы: оба disjoint UID набора выбираются
одним UID-only pass до событийной подготовки; UID sampling отделён от query и
model seed; границы и candidate protocol P001 сохранены; P004 остаётся
исторической development reference; inner p70, fit-only pruning/importance,
A1 byte gate и freeze внешних masks повторяются отдельно в каждой когорте.
Ограничения по выводам и запрет test закреплены.

`robustness_data.py` выполняет uniform `random.sample` по отсортированной
популяции оставшихся UID, записывает memberships до подготовки cohorts,
сохраняет полную историю выбранных UID и отсекает p85+ в Arrow до Python
конверсии/статистик. Event writers закрываются в `finally`, partial output
получает failure marker. В runner parent сверяет raw data hashes с manifest,
выбирает исключения из P001 snapshot и вызывает каждый cohort child в свежем
последовательном subprocess; даже после неуспеха первого child он переходит ко
второму и фиксирует `complete_with_failures`. В child сохраняются P001 candidates,
строятся baseline92 и auto206, а обучение начинается после exact A1/manual
comparison.

Самостоятельно пересчитанные frozen config/protocol SHA256 совпали с кодовыми
pins:

- config: `835928602418aca0a4b11683ea4f080556a874908d02115d5c0f256b69e7bbf6`;
- protocol: `908b10b2a12e150a1a01e25a5bac9e14b377c074f6d1e6a1f7b20a8ffdc686e0`.

## Финальная provenance проверка и readiness

Ранее найденный provenance blocker закрыт. Runner hard-pins P004 snapshot
metrics SHA256 `dfd2abf3c7ac5bfc35fff49566162847dce43251b3efafdc48e851654949b2a3`
и P001 exclusion metadata SHA256
`52a44696948ea7ebcf285c8a80b56429a19a6a4f61eb30e2132be63b76103699`. P004
metrics SHA совпадает с записью `metrics.json` в
`report/p004_snapshot/checkpoint_manifest.json`. Child проверяет hash и
содержимое membership file, seeds/counts, UID hashes, disjointness и отсутствие
пересечения с exclusions; parent передаёт raw source hashes в child.

Сверил `report/pilot_001_snapshot/data/metadata.json` с локальным
`artifacts/pilot_001/data/metadata.json`: файлы идентичны, SHA256 обоих
`52a44696948ea7ebcf285c8a80b56429a19a6a4f61eb30e2132be63b76103699`, и списки
`selected_users` совпадают (64 UID). Frozen P004 metrics указывает SHA256
`artifacts/pilot_001/data/events.parquet` как
`b65dd4c15e1772e42c7971230550f6559b3614ce07db153106576552391f5f36`, что
соответствует текущему локальному prepared event file. P004 не хранит отдельный
cohort list; его frozen config использует тот же P001 `data_dir`, а hard-pinned
P001 metadata закрепляет этот набор из 64 UID.

**Read-only preflight допускает P005 run** с зафиксированными config/protocol и
исходным кодом. Сканирование raw UID population, preparation, feature build,
CatBoost и тесты в этой проверке не запускались. После завершения обеих когорт
независимый verifier predictions/метрик, paired deltas и итоговый P005 report
остаются обязательны до интерпретации и закрытия этапа; output hashes runner-а
их не заменяют.
