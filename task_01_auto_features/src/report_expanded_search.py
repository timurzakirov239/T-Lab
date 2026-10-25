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
    if metrics!=stored['metrics'] or paired!=stored['paired_vs_manual']:
        raise AssertionError('Portable recomputation differs')
    return {'status':'PASS','tracked_files':len(manifest),'rank_rows':len(rows),
            'pooled_groups':18,'paired_user_comparisons':14,'datasets_or_models_read':False}


def create_snapshot():
    if SNAPSHOT.exists():raise FileExistsError(SNAPSHOT)
    verification=read(ROOT/'verification_independent.json')
    if verification.get('status') not in ('verified','PASS','complete'):
        raise ValueError('P007 requires independent complete verification')
    SNAPSHOT.mkdir()
    for source in [TASK/'configs/P007_PROTOCOL.md',TASK/'configs/p007_expanded.json',
                   TASK/'research/P007_CODE_REVIEW.md',ROOT/'verification_independent.json']:
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
    analysis={'experiment':'P007','status':'exploratory_complete','metrics':metrics,'paired_vs_manual':paired,
              'full_catalog_not_global_dataset':True,'final_test_evaluated':False}
    (SNAPSHOT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    (SNAPSHOT/'README.md').write_text('# P007 — compact rank checkpoint\n\n'
        'Three disjoint64-user development cohorts, all1531 original validation queries, K100 vs complete prior-observed cohort catalog. '
        'No P006 test replay. Models trained on identical original conditional K100 rows; full evaluation is a fixed-model catalog bridge.\n\n'
        'The CSV contains one-based target ranks for nine methods and two catalog modes. Missing ranks remain zero in metric denominators. '
        'Raw histories, feature matrices, predictions and models are omitted. Local independent verification checked original prediction ranks, '
        'top20 score evidence and catalog-superset invariants; portable verification checks hashes and recomputes metrics and descriptive bootstrap.\n\n'
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
        'Все1531 прежних validation-запроса трёх непересекающихся когорт64; '
        'полный каталог — все наблюдавшиеся до запроса объекты соответствующей когорты. '
        'Это не все10 000UID/50M событий Yambda и не новый независимый test. P006 сохранён без изменений.','',
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
        'Перестановочная важность при коррелирующих признаках не является причинным вкладом.','',
        '## Одинаковое обучение, два каталога оценки','',
        'Все методы используют те же ограниченные train-query blocks K100, только когда target найден. '
        'CatBoost120/depth5/.07/l2=5/Balanced/2threads,seed20261025. '
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
            'В pooled представлены158 пользователей, не номинальные192UID. Полные query-micro, strata и NDCG/Recall@10 '
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
            '## Какие признаки выбраны','',
            '| Метод | Исходная когорта | Seed26 | Seed27 |','| --- | ---: | ---: | ---: |']
    selections={c:read(SNAPSHOT/c/'selection_frozen.json') for c in COHORTS}
    for label in LABELS[1:]:
        lines.append('| '+label+' | '+' | '.join(str(len(selections[c]['final_masks'][label])) for c in COHORTS)+' |')
    lines+=['','Точные имена выбранных выражений и все inner trials — в cohort-specific '
            '`selection_frozen.json` внутри snapshot. Выбор зависит от когорты; число выбранных признаков '
            'не доказывает их индивидуальную полезность.','',
            '## Стоимость и проверка','',
            '| Когорта | Search+K100, с / MiB / fits | Exact gate, с / cells | Full, с / MiB | Catalog min–max |',
            '| --- | --- | --- | --- | --- |']
    for c in COHORTS:
        s=statuses[c]['K100'];f=statuses[c]['Full'];g=read(SNAPSHOT/c/'catalog_gate.json')
        lines.append(f'| {c} | {s["elapsed_seconds"]:.2f} / {s["resources"]["peak_process_rss_mb"]:.2f} / {s["fits_actual"]} '
            f'| {g["elapsed_seconds"]:.2f} / {g["exact_gate"]["cells"]} '
            f'| {f["elapsed_seconds"]:.2f} / {f["resources"]["peak_process_rss_mb"]:.2f} '
            f'| {f["catalog_size"]["min"]}–{f["catalog_size"]["max"]} |')
    lines+=['','Все206 float32 значения исходных100 validation candidates проверены побайтно против frozen matrices '
        'до Full; при Full проверены те же строки внутри чанков и точные scores исходных моделей. '
        'Счётчики/связи кэшируются по префиксу/пользователю, output feature matrix ограничена2048строками; '
        'полной query×catalog×206 матрицы нет. Histories проверялись на SHA как байты, event payload для models '
        'фильтровался по development-boundary до Python-конвертации.','',
        'Потолок search fits10 на новый метод, но фактические вычислительные бюджеты не равны: '
        'Permutation делает отдельные predictions для каждого признака; Stable использует дополнительный prefix fit; '
        'fixed manual не получает столько же попыток. Общий full-inner fit разделён между процедурами. '
        'Время включает fit/prediction/I/O данной реализации и не является универсальным benchmark.','',
        '## Интерпретация и ограничения','']
    for mode in ('K100','Full'):
        best=max(LABELS,key=lambda x:metrics[mode][x]['overall']['user_macro']['ndcg@20'])
        lines.append(f'Наблюдаемый максимум {mode}: **{best}**, NDCG20={metrics[mode][best]["overall"]["user_macro"]["ndcg@20"]:.6f}. '
                     'Это описательный максимум среди проверенных вариантов, не независимо подтверждённый победитель.')
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
            '- [Протокол](../configs/P007_PROTOCOL.md), [рецензия кода](../research/P007_CODE_REVIEW.md).',
            '- [Компактные доказательства](p007_snapshot/README.md), [независимая проверка](p007_snapshot/verification_independent.json).',
            '- [Предыдущий итоговый отчёт P001–P006](FINAL_REPORT.md) сохранён как первая завершённая серия.',
            '- ИИ и Luna субагенты использованы для реализации/проверок; человеческая проверка источников по [чек-листу](HUMAN_REVIEW.md) остаётся ожидаемой.']
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
