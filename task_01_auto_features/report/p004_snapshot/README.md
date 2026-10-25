# Контрольная точка P004

Сохранены точные копии метрик, split, specs, выбора, reviews и результатов.
query_ranks.csv позволяет независимо пересчитать inner/final метрики после clone.
Rank table производна от сохранённых scores; верификация самих scores была
выполнена локально, её evidence и хеши приложены. Это не повтор обучения.
Матрицы, модели, predictions и исходные данные остаются локально;
их хеши записаны в оригинальном metrics.json/output_sha256.

Проверка: python task_01_auto_features/src/verify_auto_snapshot.py

[Отчёт P004](../P004_AUTO_FEATURES.md). Test не оценивался.
