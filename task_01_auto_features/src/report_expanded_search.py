"""Build a portable P007 rank checkpoint and comparison report after verification."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import shutil

from expanded_metrics import summarize_ranks, paired_user_delta

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK/'artifacts/p007_expanded_search'
REPORT=TASK/'report'
SNAPSHOT=REPORT/'p007_snapshot'
LABELS=['Popularity','M_I','M_UIDC','Original_A1','Original_A2','Random206',
        'Permutation206','Stable206','ForwardGroups66']
COHORTS=['p001','p005_20261026','p005_20261027']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_ranks(folder,mode,label):
    path=folder/('ranks100' if mode=='K100' else 'full_catalog')/f'{label}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def catalog_bridge(rows):
    """Pair identical queries and count target outcomes, never catalog item overlap."""
    result={}
    for label in LABELS:
        small=[r for r in rows if r['mode']=='K100' and r['model']==label]
        full=[r for r in rows if r['mode']=='Full' and r['model']==label]
        paired=paired_user_delta(small,full,allow_catalog_change=True)
        by_key={(r['cohort'],r['query_id']):r for r in full}
        pairs=[(r,by_key[(r['cohort'],r['query_id'])]) for r in small]
        groups={}
        for name,predicate in [('overall',lambda r:True),
                               ('repeat',lambda r:r['target_is_repeat']),
                               ('new',lambda r:not r['target_is_repeat'])]:
            subset=[(a,b) for a,b in pairs if predicate(a)]
            hit=lambda r:r['target_rank'] is not None and r['target_rank']<=20
            recovered=lambda a,b:a['target_rank'] is None and b['target_rank'] is not None
            groups[name]={
                'queries':len(subset),
                'targets_recovered':sum(recovered(a,b) for a,b in subset),
                'recovered_targets_in_top20':sum(recovered(a,b) and hit(b) for a,b in subset),
                'top20_hits_retained':sum(hit(a) and hit(b) for a,b in subset),
                'top20_hits_lost':sum(hit(a) and not hit(b) for a,b in subset),
                'top20_hits_gained':sum(not hit(a) and hit(b) for a,b in subset),
                'targets_absent_from_full':sum(b['target_rank'] is None for a,b in subset)}
        result[label]={'target_transitions':groups,'paired_full_minus_k100':paired}
    return result


def verify_snapshot(snapshot=SNAPSHOT):
    manifest=read(snapshot/'checkpoint_manifest.json')
    for relative,expected in manifest.items():
        if sha(snapshot/relative)!=expected:raise AssertionError(f'Checkpoint SHA drift: {relative}')
    rows=[]
    with (snapshot/'ranks.csv').open(encoding='utf-8',newline='') as stream:
        for raw in csv.DictReader(stream):
            row=dict(raw)
            for field in ('query_id','user_id','timestamp','candidate_count'):
                row[field]=int(row[field])
            row['target_rank']=int(row['target_rank']) if row['target_rank'] else None
            for field in ('target_is_repeat','target_known_before'):
                if row[field] not in ('True','False'):raise ValueError('Invalid bool in rank evidence')
                row[field]=row[field]=='True'
            rows.append(row)
    stored=read(snapshot/'analysis.json');metrics={};paired={}
    for mode in ('K100','Full'):
        metrics[mode]={}
        baseline=[r for r in rows if r['mode']==mode and r['model']=='M_UIDC']
        paired[mode]={}
        for label in LABELS:
            group=[r for r in rows if r['mode']==mode and r['model']==label]
            metrics[mode][label]=summarize_ranks(group)
            if label not in ('Popularity','M_UIDC'):
                paired[mode][label]=paired_user_delta(baseline,group)
    cohort_metrics={c:{mode:{label:summarize_ranks([r for r in rows if r['cohort']==c and r['mode']==mode and r['model']==label])
                            for label in LABELS} for mode in ('K100','Full')} for c in COHORTS}
    if (metrics!=stored['metrics'] or paired!=stored['paired_vs_manual'] or
            cohort_metrics!=stored['cohort_metrics'] or catalog_bridge(rows)!=stored['catalog_bridge']):
        raise AssertionError('Portable recomputation differs')
    return {'status':'PASS','tracked_files':len(manifest),'rank_rows':len(rows),
            'pooled_groups':18,'cohort_groups':54,'paired_user_comparisons':23,'datasets_or_models_read':False}


def create_snapshot():
    if SNAPSHOT.exists():raise FileExistsError(SNAPSHOT)
    verification=read(ROOT/'verification_independent.json')
    if verification.get('status') not in ('verified','PASS','complete'):
        raise ValueError('P007 requires independent complete verification')
    SNAPSHOT.mkdir()
    for source in [TASK/'configs/P007_PROTOCOL.md',TASK/'configs/p007_expanded.json',
                   TASK/'research/P007_CODE_REVIEW.md',TASK/'research/P007_K100_RESULT_REVIEW.md',
                   TASK/'research/P007_RUNTIME_RECHECK.md',REPORT/'p007_frozen_input_verification.json',
                   ROOT/'verification_independent.json']:
        shutil.copyfile(source,SNAPSHOT/source.name)
    rows=[];cohort_status={}
    for cohort in COHORTS:
        folder=ROOT/cohort;destination=SNAPSHOT/cohort;destination.mkdir()
        cohort_status[cohort]={}
        for mode in ('K100','Full'):
            status=read(folder/('run_status.json' if mode=='K100' else 'full_catalog/run_status.json'))
            if status['status']!='complete':raise ValueError(f'Incomplete {cohort}/{mode}')
            cohort_status[cohort][mode]=status
            shutil.copyfile(folder/('metrics100.json' if mode=='K100' else 'full_catalog/metrics.json'),destination/f'metrics_{mode}.json')
            shutil.copyfile(folder/('run_status.json' if mode=='K100' else 'full_catalog/run_status.json'),destination/f'status_{mode}.json')
            for label in LABELS:
                rows.extend(dict(r,mode=mode,model=label) for r in load_ranks(folder,mode,label))
        for name in ('selection_frozen.json','importance_diagnostics.json'):
            shutil.copyfile(folder/name,destination/name)
        gate_directory=cohort_status[cohort]['Full'].get('gate_directory','catalog_gate')
        shutil.copyfile(folder/gate_directory/'run_status.json',destination/'catalog_gate.json')
        for failed in folder.glob('catalog_gate*/run_status.json'):
            if read(failed)['status']!='complete':
                shutil.copyfile(failed,destination/f'{failed.parent.name}_incomplete.json')
            if (failed.parent/'interruption.json').exists():
                shutil.copyfile(failed.parent/'interruption.json',destination/f'{failed.parent.name}_interruption.json')
    fields=['cohort','mode','model','query_id','user_id','timestamp','candidate_count','target_rank','target_is_repeat','target_known_before']
    with (SNAPSHOT/'ranks.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    metrics={};paired={}
    for mode in ('K100','Full'):
        metrics[mode]={label:summarize_ranks([r for r in rows if r['mode']==mode and r['model']==label]) for label in LABELS}
        baseline=[r for r in rows if r['mode']==mode and r['model']=='M_UIDC']
        paired[mode]={label:paired_user_delta(baseline,[r for r in rows if r['mode']==mode and r['model']==label])
                      for label in LABELS if label not in ('Popularity','M_UIDC')}
    cohort_metrics={c:{mode:{label:summarize_ranks([r for r in rows if r['cohort']==c and r['mode']==mode and r['model']==label])
                            for label in LABELS} for mode in ('K100','Full')} for c in COHORTS}
    analysis={'experiment':'P007','status':'exploratory_complete','metrics':metrics,'paired_vs_manual':paired,
              'cohort_metrics':cohort_metrics,'catalog_bridge':catalog_bridge(rows),
              'full_catalog_not_global_dataset':True,'final_test_evaluated':False}
    (SNAPSHOT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    (SNAPSHOT/'README.md').write_text('# P007 — compact rank checkpoint\n\n'
        'Three disjoint 64-user development cohorts, all 1531 original validation queries, K100 vs complete prior-observed cohort catalog. '
        'No P006 test replay. Models trained on identical original conditional K100 rows; full evaluation is a fixed-model catalog bridge.\n\n'
        'The CSV contains one-based target ranks for nine methods and two catalog modes. Missing ranks remain zero in metric denominators. '
        'Raw histories, feature matrices, predictions and models are omitted. Local independent verification checked original prediction ranks, '
        'top20 score evidence and catalog-superset invariants; portable verification checks hashes and recomputes pooled/cohort metrics, target transitions and descriptive bootstrap.\n\n'
        'Run `python task_01_auto_features/src/report_expanded_search.py --verify` with NumPy only. '
        'Human source verification remains pending.\n',encoding='utf-8')
    manifest={p.relative_to(SNAPSHOT).as_posix():sha(p) for p in sorted(SNAPSHOT.rglob('*')) if p.is_file()}
    (SNAPSHOT/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return analysis,cohort_status


def generate(analysis,statuses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='t-lab-p007'
    metrics=analysis['metrics']
    lines=['# P007 — новые способы отбора признаков и оценка полного каталога','',
        '**Дополнение к работе1; исследовательская development-оценка.** '
        'Все 1 531 прежних validation-запроса трёх непересекающихся когорт по 64 пользователя; '
        'полный каталог — все наблюдавшиеся до запроса объекты соответствующей когорты. '
        'Этот этап не охватывает всех 10 000 пользователей/50M событий Yambda и не является новым независимым test. P006 сохранён без изменений.','',
        '## Что сравнивается','',
        '| Метод | Принцип |','| --- | --- |',
        '| Popularity | Прошлое число прослушиваний трека |',
        '| M_I / M_UIDC | Фиксированные12 признаков трека /66 ручных U/I/D/C |',
        '| Original_A1 / Original_A2 | Прежние masks P004/P005 из66/206 выражений, без переотбора |',
        '| Random206 | Десять случайных поднаборов8/16/32/64/128, выбор по inner NDCG20 |',
        '| Permutation206 | Падение inner NDCG20 при перестановке признака целыми query blocks; full/top8/16/32/64/balanced |',
        '| Stable206 | Importance двух обучающих временных префиксов, L1-нормировка, mean−SD; те же размеры поднаборов |',
        '| ForwardGroups66 | Пошаговое добавление ручных групп U/I/D/C, лучший из посещённых наборов |','',
        'Все selectors выбираются только внутри train, затем masks фиксируются. '
        'Константы/дубли удаляются у новых206-pool selectors; ForwardGroups66 сохраняет состав ручных групп, '
        'как фиксированные manual controls. Stable проверяет два обучающих префикса, а не несколько независимых временных validation folds. '
        'Перестановочная важность при коррелирующих признаках не является причинным вкладом. '
        'Перестановка целых query blocks сохраняет позиции кандидатов внутри блока; '
        'это конкретная схема perturbation, а не универсальная оценка важности.','',
        '## Одинаковое обучение, два каталога оценки','',
        'Все методы используют те же ограниченные train-query blocks K100, только когда target найден. '
        'CatBoost: 120 итераций, глубина 5, learning_rate=0.07, l2_leaf_reg=5, Balanced, два потока, seed=20261025. '
        'Новые выборки отрицательных объектов и переобучение специально под полный каталог в этот этап не входят. '
        'Модель каждого метода одна и та же в K100 и Full. Следовательно, это проверка переноса на более широкий каталог '
        'с distribution shift относительно генератора обучающих кандидатов.','',
        'Каталог Full=C(t): distinct item IDs из событий когорты строго<t. '
        'Повторы разрешены, target не добавляется. События одного timestamp невидимы всем запросам этого времени. '
        'Прежний fixed mapping остаётся допущением. Target, ещё не наблюдавшийся в когорте, сохраняется как промах.','',
        '## Общее сравнение','',
        '| Метод | K100 NDCG20 | K100 Recall20 | Full NDCG20 | Full Recall20 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        a=metrics['K100'][label]['overall']['user_macro'];b=metrics['Full'][label]['overall']['user_macro']
        lines.append(f'| {label} | {a["ndcg@20"]:.6f} | {a["recall@20"]:.6f} | {b["ndcg@20"]:.6f} | {b["recall@20"]:.6f} |')
    lines+=['','Все значения user-macro: сначала среднее запросов пользователя, затем среднее пользователей. '
            'В объединённых метриках представлены 158 пользователей из 192 выбранных. Полные query-micro, strata и NDCG/Recall@10 '
            'сохранены в [analysis.json](p007_snapshot/analysis.json).','',
            '![Сравнение каталогов](p007_comparison.png)','',
            '## Что изменилось при полном каталоге','',
            '| Режим | Целей в каталоге / запросов | Новые пользователю: найдено / запросов |','| --- | ---: | ---: |']
    for mode in ('K100','Full'):
        m=metrics[mode]['M_UIDC'];o=m['overall'];n=m['strata']['new']
        lines.append(f'| {mode} | {round(o["candidate_recall"]*o["query_count"])} / {o["query_count"]} '
                     f'({o["candidate_recall"]:.1%}) | {round(n["candidate_recall"]*n["query_count"])} / {n["query_count"]} |')
    lines+=['','Повышение покрытия не гарантирует рост top20 качества: новые конкурирующие объекты могут опустить '
            'целевой трек в ранге. Для каждого target, уже присутствовавшего в K100, сохранённые scores совпали, '
            'а его Full rank не улучшился при расширении множества конкурентов. Отдельно возникают recovered targets, '
            'отсутствовавшие в K100. Не смешивать coverage query-micro с пользовательским Recall20.','',
            '| Метод | Добавлено targets в каталог | Из добавленных в top20 | Прежних top20 попаданий вытеснено | Попаданий сохранено |',
            '| --- | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        t=analysis['catalog_bridge'][label]['target_transitions']['overall']
        lines.append(f'| {label} | {t["targets_recovered"]} | {t["recovered_targets_in_top20"]} | {t["top20_hits_lost"]} | {t["top20_hits_retained"]} |')
    lines+=['','Здесь считаются переходы **целевого трека** каждого запроса; это не пересечение списков рекомендованных объектов. '
            'Из-за вложенности каталогов и точных scores прежний target вне top20 не может улучшить ранг; '
            'все новые top20 попадания относятся к targets, отсутствовавшим в K100. '
            'Это query counts. Разбиение repeat/new и парные user-bootstrap Full−K100 сохранены в `analysis.json`.','',
            '## Различия между когортами','',
            '| Метод | P001 K100 → Full NDCG20 | Seed26 K100 → Full | Seed27 K100 → Full |',
            '| --- | ---: | ---: | ---: |']
    for label in LABELS:
        values=[]
        for c in COHORTS:
            m=analysis['cohort_metrics'][c]
            values.append(f'{m["K100"][label]["overall"]["user_macro"]["ndcg@20"]:.6f} → {m["Full"][label]["overall"]["user_macro"]["ndcg@20"]:.6f}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','Полные cohort-specific Recall/NDCG@10/@20 и strata доступны в snapshot; '
            'pooled результат взвешивает представленных пользователей, а не три когорты поровну.','',
            '## Какие признаки выбраны','',
            '| Метод | Исходная когорта | Seed26 | Seed27 |','| --- | ---: | ---: | ---: |']
    selections={c:read(SNAPSHOT/c/'selection_frozen.json') for c in COHORTS}
    for label in LABELS[1:]:
        lines.append('| '+label+' | '+' | '.join(str(len(selections[c]['final_masks'][label])) for c in COHORTS)+' |')
    lines+=['','Точные имена выбранных выражений и все inner trials — в cohort-specific '
            '`selection_frozen.json` внутри snapshot. Выбор зависит от когорты; число выбранных признаков '
            'не доказывает их индивидуальную полезность.','',
            '| Метод | P001 U/I/D/C | Seed26 U/I/D/C | Seed27 U/I/D/C |',
            '| --- | --- | --- | --- |']
    for label in LABELS[1:]:
        counts=[]
        for c in COHORTS:
            ids=selections[c]['spec_ids'][label]
            counts.append('/'.join(str(sum(x.startswith(prefix) for x in ids)) for prefix in ('u_','i_','d_','c_')))
        lines.append('| '+label+' | '+' | '.join(counts)+' |')
    lines+=['','U — активность пользователя, I — характеристики/статистика объекта, '
            'D — прямые взаимодействия user–item, C — связь пользователя с исполнителями/альбомами. '
            'Статистики объекта через его категории (i_artist/i_album) относятся к I.','',
            '## Стоимость и проверка','',
            '| Когорта | Search+K100, с / MiB / fits | Exact gate, с / cells | Full, с / MiB | Catalog min–max |',
            '| --- | --- | --- | --- | --- |']
    for c in COHORTS:
        s=statuses[c]['K100'];f=statuses[c]['Full'];g=read(SNAPSHOT/c/'catalog_gate.json')
        lines.append(f'| {c} | {s["elapsed_seconds"]:.2f} / {s["resources"]["peak_process_rss_mb"]:.2f} / {s["fits_actual"]} '
            f'| {g["elapsed_seconds"]:.2f} / {g["exact_gate"]["cells"]} '
            f'| {f["elapsed_seconds"]:.2f} / {f["resources"]["peak_process_rss_mb"]:.2f} '
            f'| {f["catalog_size"]["min"]}–{f["catalog_size"]["max"]} |')
    lines+=['','| Когорта | Общий inner fit | Random subset fits | Stable prefix + subset fits | Permutation subset fits / perturbations | Forward fits | Final refits / aliases |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in COHORTS:
        status=statuses[c]['K100'];attempts=status['attempts']
        inner=[r for r in attempts if r['stage']=='inner']
        count=lambda prefix:sum(r['label'].startswith(prefix) for r in inner)
        perturbations=len(read(SNAPSHOT/c/'importance_diagnostics.json')['permutation'])
        final=[r for r in attempts if r['stage']=='outer_validation']
        aliases=sum('alias_of' in r for r in status['final_models'])
        lines.append(f'| {c} | {count("common_full_inner")} | {count("Random206_")} '
                     f'| {count("stable_early_prefix")} + {count("Stable206_")} '
                     f'| {count("Permutation206_")} / {perturbations} | {count("Forward_")} | {len(final)} / {aliases} |')
    lines+=['','Stable/Permutation дополнительно рассматривают общий full-inner mask без повторного fit; '
        'одинаковые proposals дедуплицированы до обучения. Число perturbations — отдельные вызовы predict '
        'на inner selection, плюс один baseline prediction. Оно не включено в число fits. '
        'Полный журнал каждого fit, conditional train counts и его длительность сохранены в cohort status_K100.json.','',
        'Все206 float32 значения исходных100 validation candidates проверены побайтно против frozen matrices '
        'до Full; при Full проверены те же строки внутри чанков и точные scores исходных моделей. '
        'Счётчики/связи кэшируются по префиксу/пользователю, output feature matrix ограничена2048строками; '
        'полной query×catalog×206 матрицы нет. Histories проверялись на SHA как байты, event payload для models '
        'фильтровался по development-boundary до Python-конвертации.','',
        'Сохранены диагностические попытки перед Full: первый exact gate остановился на двух mean-trend '
        'ячейках из-за различий CPython3.12 sum и NumPy reduction при компенсации дробных значений. '
        'Исправление сохраняет порядок и семантику исходного суммирования; регрессионный тест воспроизводит случай. '
        'Следующая gate-попытка прервана ради оптимизации повторных вычислений. Эти попытки не обучали модели '
        'и не оценивали полный каталог; успешная попытка с новым ID сохранена отдельно.','',
        'Потолок search fits10 на новый метод, но фактические вычислительные бюджеты не равны: '
        'Permutation делает отдельные predictions для каждого признака; Stable использует дополнительный prefix fit; '
        'fixed manual не получает столько же попыток. Общий full-inner fit разделён между процедурами. '
        'Время включает fit/prediction/I/O данной реализации и не является универсальным benchmark.','',
        '## Интерпретация и ограничения','']
    for mode in ('K100','Full'):
        best=max(LABELS,key=lambda x:metrics[mode][x]['overall']['user_macro']['ndcg@20'])
        lines.append(f'Наблюдаемый максимум {mode}: **{best}**, NDCG20={metrics[mode][best]["overall"]["user_macro"]["ndcg@20"]:.6f}. '
                     'Это описательный максимум среди проверенных вариантов, не независимо подтверждённый победитель.')
    lines+=['','| Метод | K100 ΔNDCG20 к M_UIDC [95% CI] | Full ΔNDCG20 к M_UIDC [95% CI] |',
            '| --- | ---: | ---: |']
    for label in LABELS:
        if label in ('Popularity','M_UIDC'):continue
        values=[]
        for mode in ('K100','Full'):
            b=analysis['paired_vs_manual'][mode][label]['bootstrap']['ndcg20']
            values.append(f'{b["mean"]:+.6f} [{b["ci_lower"]:+.6f}; {b["ci_upper"]:+.6f}]')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','Парные user-bootstrap интервалы против Manual UIDC (2000 resamples,seed20261102) '
            'даны в `analysis.json`. Они условны на этих данных и не корректируют множественный поиск; '
            'все development queries уже использовались ранее. Интервал, включающий0, не доказывает эквивалентность.','',
            'Малые cohort-local каталоги, conditional training на retrieval-hit queries, fixed mapping и один временной горизонт '
            'ограничивают обобщение. Full catalog здесь снимает только ограничение shortlist100 среди ранее наблюдавшихся '
            'объектов когорты; cold targets и реальная доступность всего каталога сервиса остаются нерешёнными. '
            'Наблюдаемый target — один logged listen; остальные объекты не являются доказанно нежелательными.','',
            'Следующий шаг обсуждается после результатов: общее sampled training с uniform+hard negatives, '
            'новые источники кандидатов, действительно несколько временных folds, либо новый независимый holdout. '
            'Эти эксперименты ещё не выполнены. Работа2 не начата.','',
            '## Материалы','',
            '- [Протокол](../configs/P007_PROTOCOL.md), [историческая рецензия кода](../research/P007_CODE_REVIEW.md), [финальная runtime-рецензия](../research/P007_RUNTIME_RECHECK.md).',
        '- [Компактные доказательства](p007_snapshot/README.md), [независимая проверка](p007_snapshot/verification_independent.json).',
        '- Ранги за пределами20 записаны evaluator и защищены SHA; независимо восстановлены top20-метрики по score evidence. Полные векторы scores Full не сохранялись.',
            '- [Команды проверки и воспроизведения](P007_REPRODUCIBILITY.md), [сохранность72 frozen inputs P006](p007_frozen_input_verification.json).',
            '- [Предыдущий итоговый отчёт P001–P006](FINAL_REPORT.md) сохранён как первая завершённая серия.',
            '- ИИ и Luna субагенты использованы для реализации/проверок; человеческая проверка источников по [чек-листу P007](P007_HUMAN_REVIEW.md) и [первой серии](HUMAN_REVIEW.md) остаётся ожидаемой.']
    lines=[re.sub(r'(?<=[А-Яа-яЁё])(?=\d)|(?<=\d)(?=[А-Яа-яЁё])', ' ', line) for line in lines]
    (REPORT/'P007_EXPANDED_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    import numpy as np
    for axis,key in zip(axes,['ndcg@20','recall@20']):
        y=np.arange(len(LABELS));height=.36
        axis.barh(y-height/2,[metrics['K100'][l]['overall']['user_macro'][key] for l in LABELS],height,label='K100',color='#427b99')
        axis.barh(y+height/2,[metrics['Full'][l]['overall']['user_macro'][key] for l in LABELS],height,label='Full available cohort catalog',color='#d38b4c')
        axis.set_yticks(y,LABELS);axis.invert_yaxis();axis.set_title(f'Development · user-macro {key.upper()}')
        axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True);axis.legend(fontsize=8)
    fig.savefig(REPORT/'p007_comparison.png',dpi=160)
    fig.savefig(REPORT/'p007_comparison.svg',metadata={'Date':None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        print(json.dumps(verify_snapshot(),indent=2));return
    analysis,statuses=create_snapshot();generate(analysis,statuses)
    verified=verify_snapshot()
    (REPORT/'p007_portable_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
