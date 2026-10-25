# P004 — независимый preflight

Статический read-only review `configs/P004_PROTOCOL.md`,
`configs/p004_auto.json`, `src/auto_features.py`, `src/auto_search.py` и
`src/run_auto_features.py`. Реальный runner, события, feature matrix и CatBoost
не запускались/не читались. Отчёт root о 7 + 6 проходящих тестах не проверялся
повторным запуском.

## Метод и реализация

Основной дизайн зафиксирован достаточно конкретно: frozen P001 запросы и
candidates, успешный P003 equivalence64 как gate, raw p85 cutoff до
Python-materialization, strict past-only история, p70 train-query split с
equal timestamp группами, A1 byte equality, A2 ограниченная грамматика,
pruning и union dedup только на inner-fit rows, inner-only importance/choice,
freeze masks до внешней validation, отсутствие test, fit/query/resource budgets.
Недостающие ранее уточнения по inner split, feature group, Union и знаменателям
отражены в текущем протоколе.

В просмотренном коде не нашёл дополнительной утечки: `AutoFeatureEngine`
наследует causal state базового engine, а агрегаты/доли/trend используют только
его префиксные stats. A1 спецификации следуют порядку ручных UIDC features;
runner сравнивает все A1 float32 bytes с manual matrix до первого fit.
Inner pruning получает flattened candidate rows только для inner-fit queries;
importance — с обученной на fit модели, без external eval set. Inner selection
передаёт все train queries метрикам, включая отсутствующие targets; train
labels создаются только для запросов с target в frozen candidate list. Union
сохраняет полный manual control и добавляет только novel automatic columns.
Финальные варианты замораживаются до evaluation positions внешней validation.

## Финальный config guard и допуск

Блокер config drift закрыт. До чтения/создания output `main()` вызывает
`validate_frozen_config`, который сверяет точные SHA256 конфигурации
`f7a6b1735bc0abf24b260a29ecd12040153275a9576401af7b8e638d2feede2a` и протокола
`46d7eed8d9aec15b7a722ccb2f01eca471618457bce14615c452d89e12adb8a8`. Независимо
пересчитанные hashes совпадают. Синтетический regression test изменяет лимит
поиска и protocol text и проверяет отказ guard; сообщённые root результаты —
73 passing tests, самостоятельно тесты не запускались.

**Read-only preflight допускает следующий реальный P004 запуск** при текущих
замороженных файлах и source. Реальных features/models/events в этой проверке
не запускал и не читал. Тестовый mutation coverage config/protocol есть.

Неблокирующее provenance замечание остаётся: поле
`selection_frozen.json.external_predictions_started` записывается как `false`
и не меняется после старта финальных оценок. Его следует считать снимком
замороженного отбора, а не фактическим статусом; `run_status.json` содержит
ход попыток.
