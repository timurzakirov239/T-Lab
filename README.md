# T-Lab: исходники исследования рекомендательных систем

Это рабочая ветка **research**. [Финальный русский отчёт](task_01_auto_features/report/SUBMISSION_REPORT_RU.md) объединяет P001-P008; [главная страница репозитория](https://github.com/timurzakirov239/T-Lab) содержит только README с этим отчётом.

[PDF и архив релиза](https://github.com/timurzakirov239/T-Lab/releases/tag/submission-v1.0) · [Obsidian: карта проекта](СТАРТ.md) · [подтверждение авторской проверки](task_01_auto_features/report/AUTHOR_REVIEW_CONFIRMATION.json).

## Проверка сохранённых результатов

Требуются Python3.12 и NumPy. Из корня этой ветки или распакованного архива:

```text
python -m pip install numpy==2.5.4
python task_01_auto_features/src/verify_submission.py
```

Команда проверяет компактные материалы P001/P006/P007/P008, пересчитывает основные сохранённые метрики, не запускает обучение или оценку исходных событий. [Область проверки](task_01_auto_features/research/SUBMISSION_VERIFICATION.md), [успешная квитанция](task_01_auto_features/report/submission_verification.json).

## Структура

| Папка | Содержимое |
| --- | --- |
| task_01_auto_features/src | Расчёт признаков, поиск, эксперименты, программные проверки |
| task_01_auto_features/configs | Протоколы и конфигурации, включая фиксацию до test |
| task_01_auto_features/report | Отчёты P001-P008, ранги, метрики, контрольные суммы |
| task_01_auto_features/research | Анализ семантики и программные рецензии |
| notes | Связанные исследовательские заметки Obsidian |
| task_02_rl_recommendations | Только заготовка второй работы; в сдачу не входит |

[Полная воспроизводимость](task_01_auto_features/report/REPRODUCIBILITY.md), [P007](task_01_auto_features/report/P007_REPRODUCIBILITY.md), [P008](task_01_auto_features/report/P008_REPRODUCIBILITY.md), [версии среды](task_01_auto_features/requirements-pilot.lock.txt).

Датасеты, локальное окружение, модели и большие промежуточные матрицы вне Git. Их источники и версии описаны в [DATASETS.md](DATASETS.md). Одного клонирования недостаточно для повторения обучения замороженного P006: требуются точные входы согласно инструкции. Исходный test повторно не используется для подбора.

Основной test P006: NDCG@20 ManualUIDC0,073912 / AutoA2 0,073679; прирост не подтверждён. P007/P008 - дополнительные исследования на прежней валидации. Их нельзя выдавать за независимый test. Пилот P001 и отрицательные результаты сохранены; [исторический отчёт первой серии](task_01_auto_features/report/FINAL_REPORT.md) не заменён.

Применение ИИ раскрыто в финальном отчёте. Автор 25.10.2026 подтвердил личную проверку источников и ключевых утверждений работы1; прежние статусы в исторических материалах отражают состояние соответствующих этапов.
