# P008 — команды и воспроизводимость

Этот документ описывает сохранённые шаги P008. Report generation требует три завершённых stage chains и независимый `status=verified`; он отказывается перезаписать существующий snapshot.

## До запуска

1. Прочитать [P008 protocol](../configs/P008_PROTOCOL.md) и [pre-fit review](../research/P008_PROTOCOL_REVIEW.md).
2. Зафиксировать config/protocol/source state до подготовительных запусков. Для каждого cohort output ID должен быть новым; старые/partial directories не переиспользовать.
3. Выполнять каждый stage последовательно для каждой когорты; не переходить к validation scoring, пока все 18 new fits не завершены.

## Команды stage runner

Запускать из корня репозитория. Для этого запуска используется отдельный новый output root `artifacts/p008_replay_01`; существующие output directories не переиспользовать и не перезаписывать. Runner создаёт подпапки когорт внутри `--root`. В первом цикле выполняются `prepare` и `fit` для всех когорт; только после успешного завершения цикла начинаются validation stages. Во втором цикле для всех когорт выполняются `gate` и `full`. Проверка `$LASTEXITCODE` после каждой команды немедленно останавливает последовательность при ошибке.

```powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$runner = Join-Path (Get-Location) 'task_01_auto_features/src/run_sampled_training.py'
$replayRoot = Join-Path (Get-Location) 'task_01_auto_features/artifacts/p008_replay_01'
if (Test-Path $replayRoot) { throw "Replay root already exists; choose a new immutable output root: $replayRoot" }
$cohorts = @('p001', 'p005_20261026', 'p005_20261027')

foreach ($cohort in $cohorts) {
    foreach ($stage in @('prepare', 'fit')) {
        & $python $runner --cohort $cohort --stage $stage --root $replayRoot
        if ($LASTEXITCODE -ne 0) { throw "P008 $cohort/$stage failed with exit code $LASTEXITCODE" }
    }
}

foreach ($cohort in $cohorts) {
    foreach ($stage in @('gate', 'full')) {
        & $python $runner --cohort $cohort --stage $stage --root $replayRoot
        if ($LASTEXITCODE -ne 0) { throw "P008 $cohort/$stage failed with exit code $LASTEXITCODE" }
    }
}
```

`prepare` строит frozen eligibility/sampled blocks и causal item inventory; `fit` создаёт шесть новых моделей на когорту. После завершения всех 18 fits первый `gate` создаёт общий `all_fits_frozen.json`; все `gate` и `full` statuses должны ссылаться на его SHA-256. Ledger фиксирует `frozen_before_evaluation=true` и хеши завершённых fit statuses. `gate` проверяет validation feature equality; `full` формирует K100 и cohort-catalog ranks. Report generator проверяет соответствие ledger и статусов всех когорт. Baseline teacher/reference models переиспользуются из P007 и не считаются новыми P008 fits. Не редактировать P006/P007 artifacts.

После завершения всех cohorts выполнить verifier, который подготовлен для P008 в `src/verify_sampled_training.py`:

```powershell
$verifier = Join-Path (Get-Location) 'task_01_auto_features/src/verify_sampled_training.py'
& $python $verifier --root $replayRoot
if ($LASTEXITCODE -ne 0) { throw "P008 independent verification failed with exit code $LASTEXITCODE" }
```

`--root` получает абсолютный путь, а `--config` опущен намеренно: verifier по умолчанию читает конфигурацию и protocol из папки задачи. Если передать `task_01_auto_features/...` как относительный `--root`, verifier присоединит путь к `--task-root`, что даст неверный удвоенный путь.

Verifier должен сохранить `verification_independent.json` со `status=verified` в `$replayRoot`. Он проверяет seed-reconstructed negative blocks, eligibility, teacher-score ordering, stage hashes, rank metrics и reference parity; teacher pool scores являются сохранёнными доказательствами и не означают повторное independent teacher inference.

## Snapshot и report

Для стандартного запуска, записанного в исходный `artifacts/p008_sampled_training`, после verifier можно выполнить report commands ниже. Они используют встроенный исходный root и не подходят для replay в `artifacts/p008_replay_01`. Для replay используйте независимый verifier и сохранённые им метрики; не запускайте report generator, поскольку у него нет параметра `--root` и он обращается к исходному root.

```powershell
& $python (Join-Path (Get-Location) 'task_01_auto_features/src/report_sampled_training.py')
if ($LASTEXITCODE -ne 0) { throw "P008 report generation failed with exit code $LASTEXITCODE" }
& $python (Join-Path (Get-Location) 'task_01_auto_features/src/report_sampled_training.py') --verify
if ($LASTEXITCODE -ne 0) { throw "P008 snapshot verification failed with exit code $LASTEXITCODE" }
```

Первый вызов создаёт `report/p008_snapshot/` (если каталог ещё отсутствует), `P008_SAMPLED_TRAINING.md`, `p008_comparison.png`, `p008_comparison.svg` и `p008_portable_verification.json`. Snapshot не включает event histories, sampled matrices, training records/payloads, model files или prediction score vectors. Он содержит frozen stage statuses, общий `all_fits_frozen.json`, `training_frozen.json`, rank CSV, metrics, paired contrasts и hashes; ledger подтверждает заморозку всех 18 новых fits до validation scoring. Второй вызов только сверяет snapshot manifest и полностью пересчитывает pooled/cohort metrics и 29 paired contrasts с NumPy и `expanded_metrics`; он не читает модели или events.

Если stage завершился `failed` либо verifier не `verified`, report generation не считать успешным. Не редактировать результаты в уже созданном snapshot; для новой версии нужен отдельный versioned output location и соответствующее изменение процесса до запуска.
