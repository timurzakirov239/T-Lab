# Воспроизведение и проверка работы 1

Команды ниже запускаются из корня репозитория. Основной отчёт: [FINAL_REPORT.md](FINAL_REPORT.md). Существующий P006 завершён: **его runner повторно не запускать и markers не удалять**.

## 1. Переносимая проверка результата без данных и моделей

После клонирования достаточно Python 3.12 и NumPy. В исходном окружении установлен NumPy2.5.4; полный список версий: [requirements-pilot.lock.txt](../requirements-pilot.lock.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.4
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_final_snapshot.py
```

Проверяются SHA всех файлов [snapshot](p006_snapshot/README.md), frozen masks, привязка к протоколу и membership, все18 ranking groups, 10 158 строк рангов, pooled метрики, парные CSV и bootstrap. Ожидается `status=PASS`, `complete_cohorts=3`, `pooled_metrics_recomputed=true`, `primary_bootstrap_recomputed=true`. Нужны только `verify_final_snapshot.py`, `verify_auto_checkpoint.py` и папка snapshot; histories, feature matrices, CatBoost, исходные predictions и обученные модели не читаются.

Ранги получены из проверенных predictions по score DESC / item_id ASC. Проверка подтверждает вычисление метрик из сохранённого результата, но не воспроизводит обучение и не подтверждает семантику исходных данных.

## 2. Разработка P001–P005

Команды подготовки, конфигурации и ресурсы сохранены в отдельных отчётах: [P001](PILOT_001.md), [P002](P002_RETRIEVAL.md), [P003](P003_FEATURE_PIPELINE.md), [P004](P004_AUTO_FEATURES.md), [P005](P005_ROBUSTNESS.md). Начальная загрузка: `python task_01_auto_features/src/download_datasets.py`. Она использует закреплённые версии и проверяет SHA. Для обучения нужны зависимости из lock-файла.

Новые попытки должны иметь новые run ID. Не перезаписывать исходные artifacts и snapshots. Особенно сохранить failed scale256 и исходный P001. Не считать независимым подтверждением повтор того же протокола на тех же данных.

## 3. Точное повторение замороженного P006 в отдельном checkout

До test создан тег `p006-frozen-before-test-2026-10-25` (commit `5bf0a3ae8b57b3330e541be20fe218274152b826`). [Манифест](../configs/p006_freeze_manifest.json) закрепляет SHA72 исходных файлов и входов, в том числе сохранённых development-матриц, query blocks, истории и масок отбора. Git хранит компактные материалы, большие входы находятся только локально. Обычное клонирование Git не восстанавливает эти файлы.

Для сохранения точных входов предусмотрен локальный архив `task_01_auto_features/artifacts/p006_frozen_inputs.zip`. Его состав, SHA и размер фиксируются в [p006_frozen_inputs_archive.json](p006_frozen_inputs_archive.json). В нём только72 frozen files, сам manifest и lock-файл с путями от корня репозитория. Test markers и test outputs в архив не входят. Архив не публикуется в Git; для повторного обучения его нужно передать отдельно. Сам ZIP содержит сохранённые пользовательские истории исходного публичного датасета.

На другой машине/в отдельном чистом checkout с указанным тегом:

1. Проверить SHA полученного ZIP по JSON-квитанции, распаковать в корень checkout и проверить72 SHA относительно manifest. Не смешивать с иными artifacts.
2. Установить Python3.12 и точные зависимости из lock-файла. Среда другого типа может изменить время, память или численные детали модели.
3. Прочитать [протокол](../configs/P006_PROTOCOL.md), проверить доступный ресурсный бюджет и выполнить в чистом checkout:

```powershell
.\.venv\Scripts\python.exe task_01_auto_features/src/run_final_test.py
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_final_test.py
```

Это воспроизведение уже опубликованной оценки с неизменными масками, а не подбор на test и не новый независимый эксперимент. В текущем рабочем каталоге one-shot markers блокируют такой запуск. Если архив точных входов утрачен, нельзя обещать побайтное воспроизведение P006 только по raw-данным: заново созданные metadata/timings и development-артефакты могут не совпасть с frozen hashes.

## 4. Что уже проверено

До открытия test прошли105 синтетических тестов. После однократного P006 независимый verifier пересчитал18 результатов, pooled метрики и bootstrap; [verification_independent.json](p006_snapshot/verification_independent.json). Переносимость компактного snapshot отдельно проверяется в изолированном каталоге без данных и моделей: [p006_portable_verification.json](p006_portable_verification.json).

Повторно обучать модели для проверки таблиц не требуется. Обязательная авторская проверка источников остаётся открытой: [HUMAN_REVIEW.md](HUMAN_REVIEW.md).
