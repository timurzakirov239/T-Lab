# P004 — автоматическое построение и отбор признаков, когорта64

Протокол до новых feature builds, отбора и обучения P004. Уточняет проект
`research/P004_AUTO_FEATURE_PLAN.md`. Главный вопрос: что даёт программная
генерация выражений и ограниченный отбор при одном протоколе рекомендаций?
Нулевой или отрицательный результат сохраняется. Никакого обещания превосходства.

## Данные и неизменные условия

Те же 64 пользователя, 1848 query/candidate lists P001 (1349 train/499 validation),
K=100 и исходные квоты25/25/35/15. P002 repeat-heavy в P004 не подмешивается:
это изолированное сравнение признаков на фиксированных запросах. Проверять
SHA256 P001 queries, prepared data и P003 baseline matrix/schema/structural input.
P003 scale256 failed не используется; успешная equivalence64 является гейтом.

Границы RAW: p30=9752280, p70=20125775, p85=23380165; short_window и half_life
670390. Дополнительные окна201117 и2011171 закреплены по P001. Календарная
интерпретация не вводится. Arrow фильтрует `timestamp < p85` до Python.
История query строго `< t`; все события того же timestamp ещё недоступны.
Features получают user/time/candidates; target, labels и strata не передаются.
Mapping snapshot — прежнее явно оговорённое допущение; будущие события не
задают пользовательские агрегаты. Не читать payload p85+ в Python.

Inner split: нижняя p70 квантиль распределения timestamp **train queries**,
`numpy.quantile(..., .70, method="lower")`. Fit t<cutoff, selection t>=cutoff;
равные timestamps не разделять. Сохранить cutoff, query IDs и counts до моделей.
Внешняя validation499 не участвует в генерации DSL, pruning, importance,
выборе family/поднабора/Union. Она уже использована в P001/P002: это повторное
development-сравнение; итоговый test остаётся закрытым. Это один seed/малая когорта.

## Грамматика и материализация

206 детерминированных спецификаций, максимум256; полная provenance каждой:
id/group/family/op/path/signal/window/reducer/formula. Грамматика спроектирована
человеком; автоматически выполняются её развёртка, материализация и отбор.
Это ограниченный поиск в заданном языке, не автономное открытие отношений.

A1: автоматически порождённые66 выражений той же семантики, что ручные UIDC:
U/I/D — counts events/listen/like/dislike, all и670390, age/known; item degree
artist/album; user-artist и user-album — max/mean этих counts и knownshare.
Сверить **каждую** A1 float32 ячейку с соответствующей колонкой P003 matrix
до обучения. A1 отделяет программный отбор от расширения пространства.

A2 pool включает A1 и140 новых выражений:

- 56 count-выражений с двумя дополнительными окнами на всех пяти путях;
- 42 smoothed share: listen/like/dislike, all/670390,
  `(signal_count + 0.5) / (event_count + 1)`;
- 42 trend: events/listen на трёх окнах,
  `current=recent(w)`, `previous=max(0,recent(2w)-recent(w))`,
  `(current-previous)/(previous+1)`.

Для многозначной связи событие имеет дробный вес1/k по уникальным entity;
share/trend считаются отдельно на entity, затем max/mean. Для отсутствующей
истории share=0.5; пустая relation агрегируется в0. Current интервал `(t-w,t)`,
previous `(t-2w,t-w]`. Значения считаются из prefix sums в Python float;
окончательный dtype float32. Raw IDs, X decay, unique counts и новые отдельно
учитываемые unlike/undislike signals в этот ограниченный DSL не включены.
Они по-прежнему входят в общий events-count; пропуски не являются dislike.

Один causal replay в новую memmap206. После него сравнить A1 с manual66,
записать raw/file hashes и освободить состояние engine до CatBoost.
Материализация внешних feature rows допустима без просмотра их labels/значений
при отборе; pruning получает только inner fit rows, включая все кандидаты.
Удалять константы и точные дубликаты только по inner fit; сохранять причину
и stable representative в manifest. Даже если совпадение не сохранится на
будущих данных, представители не менять. Union сохраняет все66 manual колонки;
новые automatic добавляются только если не совпадают ни с одной manual или
уже добавленной automatic колонкой на inner fit. Это не меняет сам manual
control и даёт Union=Manual, когда все выбранные выражения уже присутствуют.

## Ограниченный поиск и конечные сравнения

Для A1 и A2 независимо: первая inner модель использует все оставшиеся колонки.
Importance из неё вычисляется только на **fit**, без eval_set/early stopping.
Далее до пяти proposals: top8,16,32,64 и balanced32 (лучший признак каждой
непустой U/I/D/C группы, затем лучшие остальные до32). Порядок importance DESC,
feature ID ASC; mask хранить в каноническом порядке колонок. Одинаковые маски
переиспользовать; число реальных inner fits <=6 на family, суммарно<=12.
Каждая попытка, включая failed, считается. Качество отбора — inner user-macro
NDCG@20; ничьи: меньше колонок, затем lexicographic feature IDs. На selection
учитываются и queries без retrieved target с нулевым качеством.

До внешних predictions/оценки сохранить `selection_frozen.json`: оба winners,
их expression IDs, выбор лучшей family по inner и все конечные masks.
Для внешнего сравнения до16 финальных моделей:

1. Девять manual controls: I, UI, UID, UIC, UIDC, IDC, UDC,
   UID+C_artist, UID+C_album. Primary comparator фиксирован M_UIDC;
   I — дополнительный фиксированный сильный пилотный control.
2. AutoA1, AutoA2; обе оцениваются независимо от того, какая лучше внутри.
3. Union = manualUIDC + auto winner, выбранный только по inner score.
4. Union без каждой из U/I/D/C; отбор повторно не запускается.

Повторяющиеся masks переиспользуют model/predictions. Общий лимит **28 fits**,
не более12inner+16final; совпадающие/пустые варианты не компенсировать новыми.
Все модели CatBoost120/depth5/lr.07, l2=5, Balanced, два потока, один seed;
без model hyperparameter tuning. Auto расходует больше поискового бюджета,
чем фиксированный Manual: это отражать в стоимости, не называть бюджет равным.

Train только на запросах, где target уже есть в fixed candidate list; полный
список unmarked кандидатов сохраняется, все variants имеют одни train labels.
Финальные модели учатся на полном train P001. Evaluation включает все queries
со всеми пропусками, tie score DESC/item ASC. Primary user-macro NDCG@20,
Recall@20; @10, candidate coverage, repeat/new и warm/cold дополнительно.
Сохранить paired per-query/per-user deltas Auto/Union относительно Manual;
без заявления статистической устойчивости и без выбора по внешней validation.

## Ресурсы, ошибки и сохранение

Один тяжёлый процесс. RSS1100МиБ, reserve150МиБ; RAM limit CatBoost512mb мягкий.
Новый output ID `p004_auto64`, завершённые/partial каталоги не переиспользовать.
При ошибке сохранять failed status, attempts, прогресс и partial memmap;
такой результат не считать завершённым. Не расширять бюджет после результатов.
Сохранять source/config/protocol/input hashes, specs, split IDs, pruning,
importance, все inner scores, frozen masks, модели, predictions, ресурсы,
таблицы и независимую проверку. Коммиты: протокол до запуска, реализация,
проверенные результаты. P001/P002/P003 не менять. Работа2 пока не начинается.
