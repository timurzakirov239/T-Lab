# Данные исследования

Эксперименты P001-P008 выполнены на Yambda, вариант flat/multievent 50M. Использованная версия: `dd6f3a19eef5866e346c3270e098baa641a44948`.

Источник: https://huggingface.co/datasets/yandex/yambda

Исходные файлы размещаются в `task_01_auto_features/data/raw/yambda/`:

- `flat/50m/multi_event.parquet` — события пользователей;
- `artist_item_mapping.parquet` — связи треков с исполнителями;
- `album_item_mapping.parquet` — связи треков с альбомами.

Датасеты и обученные модели не включены в Git. Версии, размеры и контрольные суммы загрузок сохранены в [манифесте](task_01_auto_features/data/download_manifest.json), особенности файлов — в [описании загрузки](task_01_auto_features/data/DOWNLOADS.md).

Скрипт `python task_01_auto_features/src/download_datasets.py` загружает закреплённые версии Yambda и KION, сверяет файлы и продолжает незавершённые загрузки. KION был загружен при подготовке проекта, но в опубликованных экспериментах не использовался.

При сокращении числа пользователей сохраняются все их события из опубликованной версии. Полнота истории выбранных пользователей относительно исходного Yambda-5B не установлена.

Для проверки сохранённых метрик скачивать данные не требуется:

```powershell
python task_01_auto_features/src/verify_submission.py
```

Достаточно Python 3.12 и NumPy. Подготовка окружения, повторение этапов и ограничения точного повторного обучения описаны в [инструкции воспроизведения](task_01_auto_features/report/REPRODUCIBILITY.md).
