# P003 — проект оптимизированного генератора признаков

Статус: read-only дизайн до получения baseline cProfile. Код и тесты не добавлялись; `pilot_features.py`, P001 runner, данные, outputs и Git не менялись. Реализацию выбирать после профиля: если replay тратит существенную долю времени вне `FeatureEngine.transform`, этот локальный micro-optimization пакет может не оправдать внедрение.

## Интерфейс и границы

Создать отдельный модуль `src/optimized_features.py` с `class OptimizedFeatureEngine(FeatureEngine)`, импортируя базовый `FeatureEngine` из `pilot_features.py`. Переиспользовать его `__init__`, `observe`, mapping normalization, `_Stats` state, `feature_names`, `feature_groups` и причинные проверки. Переопределить только `transform` и приватные сборщики/вспомогательные функции. Конструктор сохранить ровно как у базового класса:

```python
OptimizedFeatureEngine(
    artist_map: dict[int, tuple[int, ...]],
    album_map: dict[int, tuple[int, ...]],
    short_window: int,
    half_life: int,
)
```

API: наследуемый `observe(event)` и `transform(user_id, item_ids, timestamp) -> numpy.float32[len(item_ids), len(feature_names)]`. Для прямого A/B в runner достаточно подставлять класс-фабрику; P001 `FeatureEngine` и его source hash остаются нетронутыми. Не добавлять тайминг/strategy flags к публичному constructor. Внутренние query-кэши создаются заново в каждом `transform`; состояние наблюдённой истории и порядок событий никогда не меняются кэшированием.

Порядок колонок задаётся унаследованными `feature_names` и должен полностью совпасть с базовым. Обрабатывать пустой `item_ids` тем же `(0, n_features) float32` результатом. Ошибки для запроса, где последнее observed event timestamp не строго меньше query timestamp, оставить базовому `_assert_query_time`.

## Кандидаты оптимизаций и профильный gate

Baseline cProfile нужен по функциям и call counts прежде чем выбирать реализацию. По статическому коду видно следующее, но это ещё не измеренные bottlenecks:

- **U повторяется на каждом кандидате.** `transform` получает `user_stats` один раз, но вызывает `_append_counts` для каждой строки. `_append_counts` вызывает `stats.count` восемь раз (2 окна × 4 сигнала), хотя U-значения одинаковы в одном запросе.
- **All/short чтение повторяется.** Каждый `stats.count(signal, t, window)` уже возвращает `(all_count, short_count)`, однако вызывается отдельно для окна `all` и `short`. Это происходит в U/I/D `_append_counts` и по каждому user-entity внутри `_append_entity_features`.
- **C/X может повторять один entity между кандидатами.** Для каждой категории candidate artist/album делаются lookup `(uid, entity)`, четыре `count` на каждый из двух окон и X-decay lookups. Одинаковый user-entity может повторяться у нескольких кандидатов в одном `transform`. Известен max mapping degree 241 для album, но неизвестно, часто ли такие item попали в query candidates; это само по себе не доказательство дорогого профиля.
- **Mapping normalization не повторяется.** `FeatureEngine.__init__` один раз делает dedup/sort и хранит tuple; `transform` только берёт их через `.get`. Оптимизировать повторную normalization бессмысленно, поскольку такого повтора нет.

Порядок проверки:

1. Если профиль показывает заметное время в `_append_counts`/Stats/series lookup, сначала сделать U reuse и single count retrieval вместе или по отдельности и сравнить exact output.
2. Если после них C/X helpers остаются значимыми и query содержит повторяющиеся entity, добавить локальный cache для artist/album. Если доля повторов или время мала, оставить базовый прямой обход.
3. Если `np.asarray`/Python row creation заметны либо RSS является ограничением, только отдельным экспериментом проверить preallocated output. Не менять `replay_features` или его `np.concatenate` без отдельного профиля и плана памяти.

## Безопасная форма переиспользования

### U и all/short counts

На старте transform вычислить `user_values` один раз из унаследованного `user_stats`. Для каждого сигнала вызвать ровно один раз `stats.count(signal, timestamp, short_window)`, сохранить pair, а затем составить десять U-значений. Добавлять неизменённый набор к каждой candidate row. Случай `user_stats is None` оставить нулями, age `-1`, known `0` как в baseline.

Для U/I/D count helper вызывать один `count` на каждый сигнал, но обязательно сохранить layout: сначала all values по signals в порядке `events/listen/like/dislike`, затем short values в том же порядке, потом recency и known; у I затем degrees. Нельзя добавлять `(all, short)` попеременно по сигналу — это изменит связь имён и колонок.

В C аналогично один раз получить `count` для каждой signal/category pair, а агрегировать сначала для окна all, потом для short. Если конкретные функции наследуются, можно реализовать переиспользование поверх `_Stats.count`; `_Stats` и prefix-series не менять.

### C/X query-local entity cache

В начале `transform` один раз пройти candidate items и собрать mapping category tuples, сохраняя исходный кандидатный порядок и унаследованные sorted/deduplicated tuples. Допустимо хранить словарь cache по ключу `(relation, entity_id)` только на время данного вызова. На первом появлении entity взять `stats = _user_artists.get((uid,e))` или `_user_albums.get((uid,e))` и посчитать для него exact payload:

- все/short пары по каждому из четырёх signals через `_Stats.count`;
- четыре X-decay через `_Stats.decay`;
- `has_activity()` для обычного C knownshare;
- `decay(0, timestamp, half_life) > 0.0` отдельно для X knownshare.

Для кандидата затем материализовать values в порядке его уникального `entity_ids` tuple и применить исходную редукцию в том же порядке: built-in `max(values, default=0.0)`, `sum(values) / len(values)` или 0 при пустом наборе. C knownshare остаётся `sum(bool(stats and stats.has_activity()) for stats in related) / len(entity_ids)`. X knownshare остаётся долей entity, у которых decayed activity strictly `> 0.0`. Кэш не должен объединять artist и album namespaces либо разных пользователей/запросы. Не модифицировать stats.

Кэшировать можно `_Stats` object reference до сериализации, но надёжнее сразу хранить только численные tuples/флаги: это ограничивает lifetime памяти и не скрывает состояние. Кэш не должен заменять category list set-ом: одинаковая категория в разных candidate items допустима, но внутри одной entity tuple duplicates уже удалены в базовом init.

## Числовые и causality invariants

- Не менять алгоритм `_Series.total/recent`: recent использует существующую float64 cumulative-prefix разность и `bisect_right(timestamp-short_window)`, то есть open-left границу `(t-window, t)`. Не пересчитывать short count проходом по timestamps.
- Не пересчитывать half-life из event times через `np.sum` или иной порядок; использовать только унаследованный `_Stats.decay`, который отражает online accumulation и query-time factor.
- Не переводить cached counts/decays в float32 до финального `np.asarray(rows, dtype=np.float32)`. Исторические `_Series.prefix` и Python float промежуточные значения остаются двойной точности.
- Для C/X не заменять текущие reducers на `np.mean`, векторные reductions, сортировку score или set iteration. Даже при float64 типе другой порядок суммирования может изменить последнее значение и финальные float32 bits.
- Сохранять feature append order из базового `transform`: U → I → D → C artist → C album → X item → X pair → X artist → X album. Не менять семантику missing history, knownshare и empty mapping.
- Равные timestamps остаются orchestration обязанностью: engine получает только события строго раньше query; `transform` не добавляет историю и не кэширует результат между timestamp.

## Exact-value validation plan

После выбора оптимизаций создать `tests/test_optimized_features.py`, не меняя существующие P001 tests. Для каждой synthetic fixture два engine — базовый и optimized — получают один и тот же causal event sequence и queries. Сравнивать `feature_names` list и `feature_groups` dict на равенство; результат матрицы — равные shape/dtype и `np.array_equal` (дополнительно сравнить `.tobytes()` для строгого float32 bitwise equivalence). Не использовать `allclose` как критерий приёма.

Наборы случаев:

- пустая и частично пустая история; item без mapping и candidate list пуст;
- несколько кандидатов, общий user, одинаковые artist/album у разных item;
- candidate item с множественными artist и альбомом с заметной кардинальностью;
- counts на обоих sides границы short window и много одинаковых timestamps;
- listen/like/dislike/unlike/undislike и несколько записей одной user-item пары;
- несколько transform queries одной timestamp без нового observe; после observe более позднего события следующий query видит обновлённый prefix, потому что кэш не живёт между вызовами;
- сохранение ValueError, если последнее событие не раньше query timestamp.

Только после synthetic equivalence A/B может пройти один не-тренировочный P001 development replay на тех же events/query IDs/mappings/window/half-life. Сначала проверить полное совпадение matrix values/hash и порядка кандидатов, потом измерять unprofiled wall time и peak RSS. Положительный speedup считается измеренным только при повторяемом замере без profiler; cProfile использовать только для локализации вызовов. Не читать test period и не выбирать модель по этому микропрофилю.
