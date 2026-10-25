# P007 — проверка и воспроизведение

P007 — отдельная исследовательская development-серия. Она сравнивает девять методов на всех 1 531 прежних validation-запросах трёх непересекающихся когорт. Full — все объекты, наблюдавшиеся в соответствующей когорте строго до запроса. Это не весь Yambda-50M и не новый независимый test. P006 и его frozen inputs остаются неизменными.

## Быстрая переносимая проверка

Из корня репозитория, Python с NumPy:

```powershell
python task_01_auto_features/src/report_expanded_search.py --verify
```

Используются только `report/p007_snapshot/`, `src/report_expanded_search.py` и `src/expanded_metrics.py`. Проверка сверяет SHA всех файлов snapshot, заново считает pooled и cohort-specific NDCG/Recall@10/@20, переходы target между K100/Full и описательные парные bootstrap-сравнения. Обучение и чтение датасетов не нужны. Наличие файла отчёта без успешной проверки не подтверждает корректность чисел.

CSV хранит 27 558 строк: 1 531 query × 9 методов × 2 режима. Пустой rank означает отсутствие цели и остаётся нулём в знаменателях. Target transitions — изменения попадания **целевого трека** запроса, а не пересечение рекомендованных списков. Полные scores и истории в snapshot отсутствуют.

## Локальная независимая проверка

При наличии исходных development artifacts и всех завершённых outputs:

```powershell
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_expanded_results.py --output artifacts/p007_expanded_search/verification_recheck.json
```

Для повторной проверки нужен новый путь `--output`: существующие квитанции не перезаписываются. Verifier проверяет hashes, masks, неизменность controls, восстанавливает K100 ranks из prediction NPZ, проверяет Full top20 evidence, target scores, вложенность и знаменатели. Для Popularity независимо читает SHA-pinned frozen matrix, schema и feature inputs, извлекает `i_listen_all` и восстанавливает candidate scores/ranks; сохранённые в P001 query scores дополнительно сверяются. Истории событий не декодирует, модели не обучает. Для исторических controls явно сопоставляется их прежняя схема метрик с четырьмя обязательными P007 метриками.

Ранги Full выше 20 защищены SHA, но не восстанавливаются независимо: полный вектор scores не сохранён. Все используемые top20-метрики проверяются по сохранённым top20 lists и target score/rank.

## Повтор вычислений в новом каталоге

Требуются Python 3.12, зависимости `requirements-pilot.lock.txt`, точные development matrices/queries/schema/masks из P004/P005, истории выбранных когорт и их справочники. Пути перечислены в `configs/p007_expanded.json`, SHA закреплены P006 input manifest и P007 cohort status. Данные, модели и большие промежуточные файлы в Git не входят. Существующий P006 архив содержит frozen inputs; он не заменяет P007 outputs.

Команды ниже предназначены для нового, ещё не существующего каталога. Оригинальный запуск не перезаписывать. Выполнять из корня репозитория последовательно:

```powershell
$p007ReplayRoot = Join-Path (Get-Location) 'task_01_auto_features/artifacts/p007_replay_01'
foreach ($cohortId in @('p001', 'p005_20261026', 'p005_20261027')) {
    .\.venv\Scripts\python.exe task_01_auto_features/src/run_expanded_search.py --cohort $cohortId --output $p007ReplayRoot
    if ($LASTEXITCODE -ne 0) { throw "Search failed: $cohortId" }
    .\.venv\Scripts\python.exe task_01_auto_features/src/run_expanded_catalog.py --cohort $cohortId --root $p007ReplayRoot --gate-only --gate-name catalog_gate_replay
    if ($LASTEXITCODE -ne 0) { throw "Exact gate failed: $cohortId" }
    .\.venv\Scripts\python.exe task_01_auto_features/src/run_expanded_catalog.py --cohort $cohortId --root $p007ReplayRoot --gate-name catalog_gate_replay
    if ($LASTEXITCODE -ne 0) { throw "Full catalog failed: $cohortId" }
}
.\.venv\Scripts\python.exe task_01_auto_features/src/verify_expanded_results.py --root $p007ReplayRoot
```

Все методы обучаются на одной ограниченной conditional K100 train-выборке; Full использует те же модели. Это перенос оценки на больший каталог. Target не вставляется. Пользователи без validation queries не появляются в pooled: представлены 158 пользователей из номинальных 192.

Reporter без `--verify` создаёт оригинальный snapshot по стандартному пути и отказывается перезаписывать его. Для просмотра нового replay используйте его verifier/metrics; не удаляйте исходную контрольную точку ради повторной генерации.

## Ограничения проверки

Автоматическая проверка показывает воспроизводимость вычислений в указанном протоколе. Она не устанавливает причинный вклад каждого признака, репрезентативность малых когорт или преимущество на независимых данных. Development-данные уже использовались многократно. Человеческая проверка источников и ключевых утверждений остаётся ожидаемой в `SOURCES.md`.
