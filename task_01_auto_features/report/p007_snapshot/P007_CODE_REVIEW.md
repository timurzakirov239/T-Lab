# P007 — read-only code review поисковых и catalog runners

Повторно проверены текущие версии `src/run_expanded_search.py`,
`src/run_expanded_catalog.py`, helper-модули `expanded_search.py`,
`expanded_catalog.py`, `expanded_vector_features.py`, метрики и закреплённый
`configs/P007_PROTOCOL.md`. Модели не обучались, outer metrics для изменения
кода не использовались. P006 test runner не вызывался.

## Обновлённые находки

### [closed] Полная per-query feature matrix заменена потоковой обработкой chunks

В актуальном `run_expanded_catalog.py` больше нет `all_features` для всего
`C(t)`. Для каждой validation query сначала сверяются исходные K100 признаки
и scores с frozen matrix/predictions. Затем `evaluate_full_catalog` передаёт
каталог порциями `catalog_chunk` (2048 по конфигурации); callback преобразует
только текущий item chunk и сразу сверяет строки, относящиеся к исходным K100
кандидатам, по float32 bytes. `matched` проверяет, что gate встретил каждый
original candidate. Поэтому прежний P1 о полной матрице `|C(t)|×206` закрыт.

Трансформер сохраняет per-query кэшированные item/pair counts и reduced
relation columns, а не полную query-by-catalog-by-206 матрицу. На уровне
`evaluate_full_catalog` scores всё ещё накапливаются по каждому каталогу для
точного target rank и top-20; это O(|C(t)| × число моделей) score values плюс
сортировка. Это ожидаемая стоимость текущего exact ranking алгоритма и не та
ошибка полной feature matrix, на которой основывался прежний blocker. Ресурсный
лимит и реальные peak RSS остаются обязательной проверкой каждого полного run.

Runtime gate сохраняет hash `expanded_vector_features.py` и full run откажется
работать, если файл изменится после успешного gate. Это pin для transformer,
который порождает full-catalog строки. `run_expanded_search.source_hashes`
также содержит `expanded_vector_features.py`, хотя search его не использует;
это исторический fingerprint, а не зависимость обучения от vectorizer. Для
интерпретации полного результата существенны search model/input hashes плюс
runtime source hashes и успешный exact gate.

### ForwardGroups66 — зафиксированная ручная групповая абляция

В search runner группы ForwardGroups66 формируются из исходных 66 A1 колонок,
не из `available` после pruning 206-пула. С учётом определения
`ForwardGroups66` в P007 это намеренно: метод сравнивает предопределённые
ручные группы U/I/D/C и сохраняет исходный состав каждой группы, как в manual
ablation; это не отбор subset из 206-кандидатного пула. Поэтому это не
корректностная ошибка и код после просмотра результатов менять не следует.

Нужно читать протокол точно: pruning констант/дубликатов применяется к
206-пулам новых Random/Permutation/Stable selectors; ForwardGroups66 оставляет
исходные manual group memberships. Общая фраза протокола «selectors use only
inner fit» описывает момент отбора, но фразу про pruning следует понимать
только для 206-пулов. Это заранее определённое методическое различие и
ограничение сопоставимости состава, которое следует раскрыть в итоговом
описании P007.

## Остальные проверенные участки

- Внутренний split строится только из train queries; inner fit используется для
  pruning/importance, inner selection — для выбора по NDCG@20. Внешние
  validation queries в proposal/selection код не передаются. Query loader
  ограничивает timestamp development cutoff, test-флаг остаётся false.
- Random206 генерирует уникальные masks фиксированным seed; permutation
  переставляет строки только между одинаковыми candidate-block sizes, сохраняя
  позицию кандидата; Stable206 комбинирует normalized importance двух
  префиксов; ForwardGroups выполняет до 4+3+2+1 предложений. Результаты
  остаются exploratory: inner selection используется для поиска и оценки
  кандидатов, а development validation ранее многократно просмотрена.
- A1/A2 masks извлекаются из frozen selection files, schema/order и query ↔
  feature-input row structure проверяются перед fit. Outer masks frozen до
  scoring; совпадающие итоговые masks alias к общему fit/model/prediction.
- Full-catalog replay продвигает event history строго до `timestamp < query`
  и определяет каталог из наблюдавшихся до query `engine._items`; target не
  добавляется. Для K100 проверяются exact float32 features и exact transferred
  scores; при nested catalog rank исходной цели не может улучшиться при
  идентичных scores/tie-break.
- `load_development_events` фильтрует события до Python conversion по
  `timestamp < validation_end`. `check_inputs()` дополнительно хеширует
  целый history-файл для проверки целостности; это byte-level integrity read,
  не использование поздних строк для model features или scoring.

## Знаменатели

Закреплённые outer validation counts — P001 499, P005 seed 20261026 — 514,
P005 seed 20261027 — 518; всего 1 531 query. Train counts P005 соответственно
1 517 и 1 397. Не менять их местами при сводке.

## Source SHA256 этой recheck

| Файл | SHA256 |
| --- | --- |
| `src/run_expanded_catalog.py` | `395F2E061D0988C4544E9D58431E170B6352F4A374058C3D11B00E001DD23617` |
| `src/expanded_catalog.py` | `4E6E16C2CD32DE80AE328C7C1A9038671C720714A9FD86AF6E12EBBBCB54354C` |
| `src/expanded_vector_features.py` | `1B917FCFDFB5C9517B39C990F50F080A7BAD48DC8618C22B562DD7B97A8C2534` |

## Вывод

Первая версия P1 закрыта в проверенном текущем коде: full feature matrix больше
не создаётся, K100 gate выполняется при chunk transform. Для полного запуска
остаётся наблюдать фактический RSS и завершённость всех chunks; новых
leakage/target/test корректностных ошибок в указанном review scope не выявлено.
