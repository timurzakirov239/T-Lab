# P007 — расширенный отбор признаков и полный доступный каталог

Статус до запусков: exploratory development protocol. P001–P006, их outputs и72 frozen inputs не меняются. P006 test повторно не оценивается. Этот протокол закрепляется коммитом до новых fits/scoring.

## Вопросы

1. Как меняется качество при случайном отборе, перестановочной важности, устойчивом importance-отборе и пошаговом выборе групп?
2. Как те же обученные модели ранжируют полный доступный каталог вместо исходных100 кандидатов?

## Данные и знаменатели

Три прежние непересекающиеся когорты64. Только development t<23380165. Повторно используем исходные queries/matrices206 P004/P005 read-only, проверяя SHA из P006 manifest. Оценка: все прежние outer-validation queries499/514/518 (1531 в сумме), без выбора наиболее удачных запросов. 'Вся выборка' здесь означает все эти запросы и все доступные объекты каждой когорты, не все10 000UID/50M событий Yambda.

Полный доступный каталог C(t) — sorted distinct item IDs, уже встречавшиеся в событиях соответствующей когорты строго<t. События с равным timestamp глобально невидимы. Повторы разрешены. Fixed mapping каждой когорты прежний; историческая доступность mapping остаётся допущением. Target не добавляется; отсутствующие targets остаются в знаменателе с нулевыми метриками. C(t) — наблюдаемый cohort catalog, не доказанный полный каталог сервиса.

## Обучение и методы

Чтобы изолировать selector и evaluation universe, все методы обучаются по прежнему одинаковому K100 training design: до30 train queries на UID, полный исходный блок100 при наличии target. Остальные запросы исключены из fit, но misses сохранены в оценке. Это ограниченная conditional training выборка с generator-biased unmarked negatives; новые методы не получают другой sampling. Full-catalog scoring — проверка переноса этих же моделей на более широкий universe, не claim о моделях, специально обученных на всём каталоге.

CatBoost и seed прежние:120/depth5/.07/l2=5/Balanced/2threads/modelseed20261025. Manual I12 и Manual UIDC66 фиксированы. Original A1/A2 используют прежние masks без донастройки. Популярность — прошлое listen count.

Новые selectors используют только inner fit/selection из train: cutoff70th percentile lower; ties не разделяются. Константы/дубли удаляются по inner fit. Внешняя validation и full catalog не участвуют в выборе masks.

- Random206:10 уникальных случайных masks, размеры8/16/32/64/128 циклически, заранее фиксированный cohort seed.
- Permutation206: обучить общий full-inner model; переставлять каждый признак между целыми query blocks одного размера на inner selection, сохраняя порядок candidates внутри блока. Важность — падение user-macro NDCG20. Проверить full/top8/top16/top32/top64/balanced32.
- Stable206: две importance-модели на early50% train-query timestamp и inner-fit70%. Нормировать L1 importance отдельно; score=mean−populationSD. Проверить те же masks на inner selection. Это устойчивость importance к двум обучающим префиксам, а не полноценная многократная temporal-CV.
- ForwardGroups66: последовательный выбор групп U/I/D/C; на каждом шаге проверить все одногрупповые расширения выбранного набора, до4+3+2+1=10fits; итоговый выбор лучший из посещённых masks по inner score.

У каждого нового метода потолок10 search fits; общий full-inner fit разделён и его стоимость раскрывается. Stable имеет один дополнительный early-prefix fit. Количество реальных fits и predictions сообщается, равенство фактических вычислений не предполагается. Выбор: inner NDCG20, затем меньше признаков, lex spec IDs. Оценки importance не являются причинным доказательством полезности признака.

## Порядок оценки

После поиска записать immutable selection_frozen.json со всеми masks и SHA, затем обучить итоговые модели на всех исходных train queries и оценить K100. Одинаковые masks alias к одному fit. Затем в свежем процессе выполнить full-catalog replay для всех тех же validation queries и всех моделей, разделяя feature chunks. Не менять модели после K100/full результатов.

До full scoring: синтетический gate transform float32 byte equality с исходным AutoFeatureEngine; реальный gate на всех исходных100 кандидатах по всем validation queries против frozen matrix. Для ускорения разрешена vectorization с exact value gate; семантика не меняется. Full иK100 score для исходных candidates должны совпасть; rank в full не может быть лучше K100 для присутствовавшего target при идентичных scores и вложенных каталогах. Отдельно сообщить recovered targets и catalog sizes.

NDCG/Recall@10/@20, query-micro и user-macro, repeat/new и warm/cold. Pooled равновесен представленным пользователям, не среднему по номинальным192UID. Парные user deltas, descriptive bootstrap2000 seed20261102. Эти development-данные уже использованы; никакой независимой итоговой значимости/победы не заявлять.

## Ресурсы и сохранение

Последовательные дочерние процессы, лимит RSS1100MiB/reserve150MiB. Full evaluation потоковый, chunk2048, без хранения всей query×catalog×206 матрицы. Сохранять progress, runtime, peak, модели, masks, compact target ranks/top20 score evidence и hashes. Частичный full-run не выдавать за полную оценку; при необходимости продолжать с новым run ID/сохранённой границей без перезаписи.

P007 не заменяет исходный итоговый отчёт P006: отдельный отчёт/Obsidian карточка/commits. После результатов определить следующий этап совместно с пользователем. Работа2 не начинается. Human verification pending.
