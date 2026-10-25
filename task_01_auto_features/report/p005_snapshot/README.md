# Компактная контрольная точка P005

Снимок содержит замороженные конфигурацию, membership, родительскую и дочерние независимые проверки, метрики и компактные per-query ranks для завершённых когорт.
Неуспешные когорты сохранены вместе с доступной диагностикой; их ranking checks пропускаются.

Reviews: [P005_REVIEW.md](P005_REVIEW.md), [P005_RESULT_REVIEW.md](P005_RESULT_REVIEW.md).

Исходные таблицы, матрицы признаков, модели и архивы predictions не копируются. Их хеши и статусы остаются в сохранённых manifests/status evidence. Snapshot verifier работает только по компактным rank CSV.

Проверка: `python task_01_auto_features/src/verify_robustness_snapshot.py`
