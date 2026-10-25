# P008 — команды и воспроизводимость

Этот документ описывает сохранённые шаги P008. Report generation требует три завершённых stage chains и независимый `status=verified`; он отказывается перезаписать существующий snapshot.

## До запуска

1. Прочитать [P008 protocol](../configs/P008_PROTOCOL.md) и [pre-fit review](../research/P008_PROTOCOL_REVIEW.md).
2. Зафиксировать config/protocol/source state до подготовительных запусков. Для каждого cohort output ID должен быть новым; старые/partial directories не переиспользовать.
3. Выполнять каждый stage последовательно для каждой когорты; не переходить к validation scoring, пока все 18 new fits не завершены.

## Команды stage runner

Запускать из корня репозитория; повторить команды для `p001`, `p005_20261026`, `p005_20261027`:

```powershell
python task_01_auto_features/src/run_sampled_training.py --cohort p001 --stage prepare
python task_01_auto_features/src/run_sampled_training.py --cohort p001 --stage fit
python task_01_auto_features/src/run_sampled_training.py --cohort p001 --stage gate
python task_01_auto_features/src/run_sampled_training.py --cohort p001 --stage full
```

`prepare` строит frozen eligibility/sampled blocks и causal item inventory; `fit` создаёт шесть новых моделей на когорту; `gate` проверяет validation feature equality; `full` формирует K100 и cohort-catalog ranks. После завершения всех 18 fits первый `gate` создаёт общий `artifacts/p008_sampled_training/all_fits_frozen.json`; все `gate` и `full` statuses должны ссылаться на его SHA-256. Ledger фиксирует `frozen_before_evaluation=true` и хеши завершённых fit statuses. Report generator проверяет совпадение ledger и статусов всех когорт и переносит ledger в snapshot. Baseline teacher/reference models переиспользуются из P007 и не считаются новыми P008 fits. Не редактировать P006/P007 artifacts.

После завершения всех cohorts выполнить verifier, который подготовлен для P008 в `src/verify_sampled_training.py`:

```powershell
python task_01_auto_features/src/verify_sampled_training.py --root task_01_auto_features/artifacts/p008_sampled_training --config task_01_auto_features/configs/p008_sampled.json
```

Verifier должен сохранить `artifacts/p008_sampled_training/verification_independent.json` со `status=verified`. Он проверяет seed-reconstructed negative blocks, eligibility, teacher-score ordering, stage hashes, rank metrics и reference parity; teacher pool scores являются сохранёнными доказательствами и не означают повторное independent teacher inference.

## Snapshot и report

После verifier:

```powershell
python task_01_auto_features/src/report_sampled_training.py
python task_01_auto_features/src/report_sampled_training.py --verify
```

Первый вызов создаёт `report/p008_snapshot/` (если каталог ещё отсутствует), `P008_SAMPLED_TRAINING.md`, `p008_comparison.png`, `p008_comparison.svg` и `p008_portable_verification.json`. Snapshot не включает event histories, sampled matrices, training records/payloads, model files или prediction score vectors. Он содержит frozen stage statuses, общий `all_fits_frozen.json`, `training_frozen.json`, rank CSV, metrics, paired contrasts и hashes; ledger подтверждает заморозку всех 18 новых fits до validation scoring. Второй вызов только сверяет snapshot manifest и полностью пересчитывает pooled/cohort metrics и 29 paired contrasts с NumPy и `expanded_metrics`; он не читает модели или events.

Если stage завершился `failed` либо verifier не `verified`, report generation не считать успешным. Не редактировать результаты в уже созданном snapshot; для новой версии нужен отдельный versioned output location и соответствующее изменение процесса до запуска.
