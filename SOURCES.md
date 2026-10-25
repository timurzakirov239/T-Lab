# Источники

Полный список использованных методов, пояснения к собственным адаптациям и ссылки приведены в [разделе 12 итогового отчёта](README.md#12-источники-и-применённые-методы).

| Материал | Назначение |
| --- | --- |
| [Yambda: закреплённая карточка датасета](https://huggingface.co/datasets/yandex/yambda/blob/dd6f3a19eef5866e346c3270e098baa641a44948/README.md) | Описание исходных данных |
| [CatBoost: обучение](https://catboost.ai/docs/en/concepts/algorithm-main-stages) и [важность признаков](https://catboost.ai/docs/en/concepts/fstr) | Модель и исходный отбор A1/A2 |
| [scikit-learn: случайный поиск](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization), [перестановочная важность](https://scikit-learn.org/stable/modules/permutation_importance.html), [последовательный отбор](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) | Общие принципы, адаптированные в собственных процедурах |
| [NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) и [bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | Метрика и оценка неопределённости |

Определения грамматики, Stable206, Union и обучающих конкурентов заданы в коде исследования. Документация общих методов не заменяет описание этих адаптаций; их различия разобраны в отчёте.

Численные результаты подтверждаются конфигурациями, компактными снимками рангов и проверками в `task_01_auto_features/report/`. Единая команда проверки — `python task_01_auto_features/src/verify_submission.py`. Она пересчитывает сохранённые метрики и не обучает модели заново.
