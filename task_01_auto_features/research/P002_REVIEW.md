# P002 — протокольная проверка до результатов

Статус: предварительная независимая проверка фиксируемого протокола; никаких P002 metrics не просмотрено или рассчитано. Runner и config JSON ещё не проверялись — этот этап будет отдельно после их готовности.

Подтверждаю по текущему `src/run_pilot.py`, что P001 related sources используют top 5 artist/album профилей, а source candidate lists обрезаются до `max_candidates=100`. Протокол P002 сохраняет эти определения для exact equality anchor и не расширяет baseline через диагностическую глубину 200; глубина 200 полностью используется только RRF. Для Affinity32 явно отделены top32 профилей от первых 100 item в выдаче квот.

Протокол задаёт те же 1 349 train и 499 validation queries, проверку candidate IDs в точном порядке для всех 1 848 запросов P001, user-macro candidate recall как главный критерий, query-micro/counts и repeat/new, warm/cold с исходным знаменателем. Target-derived значения и P001 candidates закрыты от генераторов до завершения выдачи. Заданы Arrow-фильтрация до Python-level преобразований, strict prefix `< t`, source contribution/overlap отчёт, запрет test-оценки и обучения моделей, resource limits, failed-run handling и хеширование входов/кода/config.

Критических неясностей в зафиксированном protocol.md не осталось. При проверке P002 runner-а особенно проверить, что Arrow-фильтр по `timestamp < p85` выполнен до `.to_pydict()` и что target/target strata/P001 `candidate_items` физически не передаются стратегии до фиксации её candidate IDs. Все утверждения о результатах остаются пустыми до отдельного review завершённого run.
