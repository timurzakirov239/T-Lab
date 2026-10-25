# P008 — read-only pre-fit review протокола

До fits прочитаны только `configs/P008_PROTOCOL.md` и
`configs/p008_sampled.json`. Реальные fits, P008 outputs, raw event payload и
P006/P007 results не читались. Это оценка только схемы до запуска.

## Проверки дизайна

- **Запросы и scope:** protocol фиксирует прежние 1 531 outer-validation
  queries из трёх непересекающихся когорт и 158 представленных пользователей.
  Evaluation остаётся в development до `validation_end`; P006 test выключен.
  C(t) — наблюдавшиеся в когорте items с первым/предыдущим событием строго
  до query, равный timestamp не виден. Target не инжектируется в evaluation.
- **Warm eligibility:** текущая версия протокола явно определяет warm/cold как
  известность в cohort-wide C(t), в согласии с `target_known_before` в
  предыдущих runners. New-to-user item допускается, если ранее наблюдался
  другим пользователем этой когорты. `target_is_repeat` отделён. В отчётах
  P008 нужно сохранить эту семантику и вывести отдельно cohort-catalog
  availability, user-known warm/cold и repeat counts.
- **Изоляция arms:** Baseline переиспользует исходные P007 K100 models.
  UniformHit и HybridHit имеют тот же query-ID набор, что baseline hit queries;
  различается состав отрицательных объектов. Важно, чтобы первые 50
  uniform draws HybridHit точно совпадали с теми же draws в UniformHit, а
  общие hit-query blocks в HybridWarm были byte-identical к HybridHit. Эти
  отношения хорошо заданы текстом и должны быть assert-нуты в sampling
  verifier.
- **Тепловое расширение:** HybridWarm добавляет запросы с target в C(t),
  включая новые пользователю items, а общие hit queries сохраняют HybridHit
  sampling. Поэтому HybridWarm−HybridHit оценивает совместный эффект
  дополнительного warm-to-cohort query coverage и увеличения training data
  при неизменном sampling на пересекающихся запросах; это не изолированный
  эффект только негативов на расширенной выборке.
- **Основной контраст:** primary — Full user-macro NDCG@20
  HybridHit_UIDC против Baseline_UIDC. При совпадающем hit-query наборе и
  неизменных UIDC66 mask это направленный тест замены shortlist-derived
  negatives на 50 uniform +49 teacher-hard. UniformHit и сравнения с A1
  дополнительные; masks фиксированы, нового поиска/tuning нет. Cohort-specific
  A1 masks 16/62/8 допустимы как заранее обозначенное продолжение
  exploratory P007; не трактовать их как независимо выбранный результат.
- **Teacher и mining:** teacher — прежний P007 M_UIDC, обученный на полном
  train portion и используемый для in-sample hard mining. Протокол явно
  признаёт эту зависимость/смещение, отсутствие validation/test labels и
  offline-training interpretation. Пул исключает target и Uniform50,
  дубликаты/пересечения запрещены; deterministic seeds, teacher-score
  tie-break и item-first-seen inventory создают проверяемую основу.
  Unmarked negatives корректно не объявлены реальными отрицательными
  реакциями.
- **Masks:** UIDC66 постоянен; Original_A1 masks берутся per cohort из P007
  (16/62/8), в пределах первых66 features. Для запуска закрепить точные
  spec IDs и SHA каждого cohort mask/teacher, чтобы исключить ошибочное
  использование A2 columns или переотбор после результатов.
- **Fit budget:** config `maximum_total_fits=6` следует понимать как
  per-cohort limit: три новых arms × две masks = 6 на cohort; по трём
  cohorts получается ровно **18 новых fits**. Baseline models переиспользуются
  и в эти 18 не входят. Gate/full scoring не являются CatBoost fits.
  Runner/status должны явно показывать per-cohort 6 и project total 18; не
  пытаться применить cap6 к общему parent-run.

## Существенная реализационная формулировка до кодирования

В protocol сказано, что подготовятся матрицы
`UniformHit (hit_queries × 100 × 66)` и
`HybridWarm (warm_queries × 100 × 66)`. Нужно уточнить в runner manifest, что
эти 100 rows — именно sampled training block `[target + 99 sampled negatives]`
на query. Для новых item IDs их признаки должны заново материализоваться из
строго до-query cohort history. P007 frozen feature matrix содержит лишь
прежние candidate rows; она подходит для exact gate исходных K100 inputs, но
не заменяет features для равномерно выбранных объектов C(t), отсутствующих в
shortlist.

Аналогично teacher hard pool требует score/features для union прежних K100
IDs и до256 дополнительных C(t) draws, затем исключения target и Uniform50,
а не только для финальных49 negatives. Эти временные pool features могут
считаться потоково и не обязаны сохраняться целой большой матрицей; достаточно
сохранить проверяемые IDs/scores и links к prefix/input hashes по протоколу.
До fit следует явно отделить: исходную K100 feature gate, sampled 100-row
student training matrix и временный teacher-pool scoring. Это предотвращает
случайное использование shortlist matrix для вне-shortlist IDs.

## Ограничения, которые корректно раскрыты

P008 остаётся exploratory: cohort-local каталог и один временной горизонт,
user/cold distinctions, conditional K100 baseline, sampled unmarked negatives,
in-sample teacher mining, post-P007 A1 mask history и повторно использованная
validation ограничивают причинную/популяционную интерпретацию. Full models
не обучаются на полном каталоге. Human source review pending; работа2 не
начинается.

**Итог:** сравнения, eligibility, primary contrast и aggregate budget
методически согласованы, если cap6 исполняется per cohort и shared sampled
blocks проверяются. Единственная существенная ясность для реализации —
явно описать/сохранить feature rows для sampled candidates и полного teacher
pool; frozen P007 K100 matrix является gate/control, не источником всех
новых sampled-item features.
