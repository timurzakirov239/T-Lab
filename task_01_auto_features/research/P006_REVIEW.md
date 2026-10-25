# P006 — protocol preflight до test opening

Read-only review `configs/P006_PROTOCOL.md`,
`configs/p006_final_test.json`, P001/P005 checkpoint path inventories and
preceding development reviews. Никакие reserved event values/payload, test
queries, features или scores не читались; runner ещё не оценивался.

## Методические решения

Основной вопрос и контроль зафиксированы: Auto_A2 против Manual UIDC; P004/P005
cohort-specific selection masks переносятся без выбора по test. Модели обучаются
на собственных замороженных development blocks каждой когорты, с одинаковыми
параметрами и модельным seed. Candidate strategy, strict past-only replay,
equal-timestamp handling, target separation, fixed mapping snapshots, limits по
fits/resources, one-shot marker и запрет повторного test прописаны.

План bootstrap корректно трактует CI как условное описание тестовых пользователей,
а не вывод о популяционной значимости; pooled macro должен давать одинаковый
вес каждому UID с хотя бы одним test query, при отдельном показе фактического N
и числа выбранных UID без подходящих запросов. При online replay строки должны
быть упорядочены глобально по timestamp; все события timestamp `t` должны
оставаться невидимы всем запросам этого времени. Query target необходимо
извлекать и хранить отдельно; candidate/feature API принимает только UID,
timestamp и уже закреплённые candidate IDs. После обучения на test разрешён
только заранее заданный набор масок, без последующего выбора победителя или
изменения кода/порогов.

## Уточнение P001 history artifact

Первоначальное замечание об отсутствии P001 full history было ошибочным и
отозвано. Проверен `src/pilot_data.py`: выбранные UID строки целиком
записываются через `writer.write_batch(chosen)` без timestamp-фильтра. Порог
`timestamp < validation_end` в `run_retrieval.load_development_events`
применяется при загрузке для development replay, а не при создании сохранённого
Parquet. Footer `artifacts/pilot_001/data/events.parquet` показывает 318 157
строк и колонки событий, что соответствует frozen metadata row count. История
из config пригодна для test replay. Parquet footer проверен без чтения payload.

## Runner preflight и one-shot guard

`run_final_test.py` правильно требует committed freeze manifest до загрузки
истории, сверяет config/protocol/source/input hashes, проверяет сохранённые
development matrices, использует только cohort-local mapping snapshot, глобально
сортирует историю по timestamp через `load_published_history`, выбирает test
queries после p85 и не передаёт target в candidate generator или feature engine.
Online replay использует только `timestamp < query timestamp`; A1/manual
float32 gate и structural order check выполняются до fit. Финальное обучение
включает прежние train и validation blocks без мутации сохранённых query
records; блоки с пропущенной retrieval целью остаются в знаменателе, но не
входят в классификаторное обучение.

Ранее найденный one-shot обход закрыт: child разрешается только в точном
`output/<cohort_id>` пути и эксклюзивно создаёт отдельный
`<cohort_id>.test_opened.json` claim до чтения истории. Claim сохраняется при
ошибке. Глобальный lock запрещает повторный parent run. Это предотвращает
повторное открытие test через другой sibling output.

Текущий static runner review не выявил оставшегося методического blocker.
Открытие test всё ещё должно ждать зафиксированного, закоммиченного freeze
manifest и проверки его source/input hashes. Manifest будет добавлен после
окончательной версии runner/helpers и содержит проверяемые fingerprints;
перед первым чтением история, source или test outputs не открывались.

Отдельный post-run gate: независимая проверка всех predictions/metrics, hashes
и portable rank snapshot, затем cohort/user-level анализа и отчёта. Один
`final_test_analysis.py` не следует считать заменой protocol-required verifier.

## Обязательные проверки нового runner

До test read fail-close закрепить hashes config/protocol, P001/P005
development artifacts, cohort memberships, queries, feature masks/specs, model
sources и raw source manifest. Проверить, что P001 cohort membership и P005
memberships попарно disjoint; P005 memberships/exclusions совпадают с verified
P005 checkpoint. Удостовериться, что developer inputs 206 matrix и original
candidate blocks идут в одном row/query order; A1/manual66 exact bytes gate
срабатывает на test matrix до первого fit/scoring.

На synthetic fixture проверить test cutoff inclusion (`>= p85`), до-конца
истории query eligibility, до 10 uniform unambiguous listens на UID без
досэмплирования, same-time non-observation, отсутствие target в candidate и
feature inputs, после-query test events не влияют на прошлые признаки,
CandidateGenerator P001 quotas и tie-break popularity, правильное population
bootstrap с user-level paired deltas. Проверить failed child сохраняет статус и
следующие предвыбранные когорты исполняются без redrawing.
