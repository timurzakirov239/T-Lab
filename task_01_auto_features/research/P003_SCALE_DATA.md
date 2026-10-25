# P003: подготовка scale-256 данных

Подготовлен runner `src/scale_data.py`; реальный источник не сканировался и P003 scale-256 не запускался на Yambda. Реализация предназначена для подготовки файлов и статистик только; features, запросы, обучение и оценка остаются в основном runner.

## Контракт

`prepare_scale(output, users=256, seed=20261025, cutoff=23380165, event_source=..., artist_source=..., album_source=..., monitor=None)` делает равновероятный `random.sample(sorted(unique_uid_population), users)` после отдельного UID-прохода. Второй проход сохраняет целые исходные Arrow batches выбранных UID в `events_full.parquet`, включая строки позднее cutoff. Поздний payload передаётся Arrow writer напрямую и не переводится в Python, не суммируется и не используется для выбора.

Для development данных timestamp-фильтр `timestamp < cutoff` применяется в Arrow прежде, чем любые значения переводятся в Python. В `events.parquet` записываются только `uid`, `item_id`, `timestamp`, `event_type`. Dev-only статистики и уникальные item IDs считаются из этого фильтра; mappings из закреплённой локальной версии фильтруются только по dev item IDs через существующий `_mapping` helper. Metadata фиксирует hash списка UID, размер UID population/просканированные строки, полные сохранённые строки (Arrow row count), development counts/пользователей/items/типы, batch-aware dev chronology, mapping coverage и runtime. Значения событий на/после cutoff не агрегируются; помимо общего числа сохранённых полных строк отдельные post-cutoff statistics не вычисляются.

Output обязан быть новым каталогом; существующий каталог не перезаписывается. Перед каждым event input batch и каждым dev stats chunk вызывается `monitor.check()`, если monitor передан. При переданном monitor mappings также проходят локальный monitored batch helper; иначе используется `_mapping` из P001. При исключении на любом этапе preparation создаётся `FAILED.json`; writers event pass закрываются в `finally`, partial parquet остаётся читаемым для диагностики, а повторный запуск в этот output path запрещён.

## Проверки

Из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m unittest task_01_auto_features.tests.test_scale_data -v
```

Шесть tiny-fixture тестов проверяют точную равновероятную выборку UID по seed, полную историю включая post-cutoff sentinel только в `events_full.parquet`, скрытие sentinel из dev статистик, равные timestamp на границе batch и сортировку, нулевое покрытие mappings/схему пустого файла, вызовы monitor и его отказ как до event pass, так и после записи batch с readable partial Parquet, `FAILED.json` и запретом повторного запуска. Per-user statistic использует математическую медиану. Это synthetic validation, не измерение полной Yambda подготовки.

CLI, когда основной агент завершит подготовительные runner проверки:

```powershell
.\.venv\Scripts\python.exe task_01_auto_features\src\scale_data.py --users 256 --seed 20261025 --cutoff 23380165 --output artifacts/p003_scale256/data
```

Факт полного хранения выбранных UID ограничен опубликованным 50M источником и сам по себе не доказывает полноту истории относительно 5B. Metadata и mappings являются локальными snapshot данными; временная availability mappings остаётся неизвестной.
