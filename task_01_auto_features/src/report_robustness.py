"""Report P005 development replications from independently verified predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .report_auto_features import query_contributions, write_csv
else:
    from report_auto_features import query_contributions, write_csv

TASK = Path(__file__).resolve().parents[1]
CORE = ['M_I', 'M_UIDC', 'Auto_A1', 'Auto_A2', 'Union']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def generate(root):
    summary = read(root/'summary.json')
    verification = read(root/'verification_independent.json')
    if verification['status'] != 'verified' or summary['status'] not in ('complete','complete_with_failures'):
        raise ValueError('P005 report requires independently verified finished parent')
    rows, paired, users, runs, selections = [], [], [], [], []
    lines = ['# P005 — устойчивость автоматического отбора на новых когортах', '',
             'Две заранее выбранные непересекающиеся development-когорты; результаты и неудачи сохраняются без повторного отбора. Итоговый test не оценивался.', '',
             '## Метод и границы интерпретации', '',
             'До обработки событий обе membership заморожены после UID-only прохода опубликованного Yambda-50M. '
             'Исключены все64 пользователя P001/P004; затем uniform random.sample выбрал64 UID с seed20261026 и64 с seed20261027 из оставшихся UID. '
             'Нет фильтра по активности или качеству. Это две условные выборки из одной конечной популяции, а не независимые наборы данных. '
             'Полные опубликованные истории сохранены; Arrow удаляет timestamp>=23380165 до Python payload/statistics. '
             'Полнота50M относительно5B и временная доступность mapping не доказаны.', '',
             'RAW границы P001 сохранены: train queries от9752280 до20125775, внешняя validation до23380165. '
             'До30 train/10 validation запросов на пользователя; неоднозначные группы listen исключаются. '
             'История строго раньше запроса, включая обработку ties. Original P001 candidates K100 '
             '(popular25/repeat25/artist35/album15), target не вставляется. Retrieval P002 не переносился. '
             'Каждая когорта имеет свой query sampling seed; общий model seed20261025.', '',
             'Грамматика206 и алгоритм отбора P004 закреплены source hashes: A1=66 ручных UIDC, '
             'A2 ищет во всех206 выражениях, включая140 дополнительных окон/долей/трендов. '
             'Inner split — train timestamp p70 lower, ties не разделяются; pruning/importance используют fit, '
             'поднабор выбирается на inner selection. Full/top8/16/32/64/balanced32 с dedup; максимум6 fits на семейство,28 total. '
             'Все финальные masks замораживаются до внешнего scoring. Manual I — дополнительный фиксированный контроль, '
             'primary comparator — Manual UIDC. CatBoost120/depth5/lr.07/l2=5/Balanced/2threads. '
             'Автоматический метод расходует дополнительный бюджет поиска; Manual не получает равного числа попыток.', '',
             '## Результаты отдельно по когортам', '']
    for entry in summary['cohorts']:
        folder = root/entry['run_dir']
        status = read(folder/'metrics.json')
        runs.append((entry,status))
        seed = entry['seed']
        lines += [f'### Когорта {seed}', '']
        if status['status'] != 'complete':
            lines += [f"Запуск failed: `{status.get('error')}`. ПикRSS {status['resources']['peak_process_rss_mb']} МиБ. "
                      'Частичные outputs не используются как результат; пользователей не заменяли и бюджет не повышали.', '']
            continue
        models = [m for m in status['final_models'] if 'metrics' in m]
        by_label = {m['label']:m for m in models}
        queries = [json.loads(line) for line in (folder/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
        by_id = {q['query_id']:q for q in queries}
        split = read(folder/'split.json')
        selected = read(folder/'selection_frozen.json')
        specs = read(folder/'specs.json')
        metadata = read(folder/'data/metadata.json')
        coverage = defaultdict(list)
        for q in queries:
            if q['split'] == 'validation': coverage[q['user_id']].append(q['target_item'] in q['candidate_items'])
        macro_coverage = float(np.mean([np.mean(values) for values in coverage.values()]))
        measured = by_label['M_UIDC']['metrics']
        overall = measured['overall']
        lines += [f"Полная история {metadata['full_history_retention']['stored_full_rows_arrow_count']:,} строк; "
                  f"development {metadata['development']['rows']:,}. Запросов train {sum(q['split']=='train' for q in queries)}, "
                  f"validation {overall['query_count']} от {overall['user_count']} пользователей. "
                  f"Inner cutoff {split['cutoff']}: {len(split['inner_fit'])} fit / {len(split['inner_selection'])} selection. "
                  f"A1 побайтно совпал с manual во всех {status['a1_manual_equivalence']['cells_compared']:,} float32 ячейках до обучения.", '',
                  '| Набор | Столбцов | NDCG@20 user-macro | Recall@20 user-macro |',
                  '| --- | ---: | ---: | ---: |']
        for model in models:
            m = model['metrics']
            row = {'cohort_seed':seed,'model':model['label'],'columns':model['feature_count'],
                   'ndcg20':m['overall']['user_macro']['ndcg@20'],'recall20':m['overall']['user_macro']['recall@20'],
                   'ndcg10':m['overall']['user_macro']['ndcg@10'],'recall10':m['overall']['user_macro']['recall@10'],
                   'repeat_ndcg20':m['strata']['repeat']['user_macro'].get('ndcg@20'),
                   'new_ndcg20':m['strata']['new']['user_macro'].get('ndcg@20'),
                   'reused_from':model.get('reused_from','')}
            rows.append(row)
            lines.append(f"| {row['model']} | {row['columns']} | {row['ndcg20']:.6f} | {row['recall20']:.6f} |")
        lines += ['', f"Retrieval: {measured['counts']['targets_in_candidates']}/{measured['counts']['queries']} целей в кандидатах; "
                  f"candidate recall user-macro {macro_coverage:.6f}. "
                  f"Repeat/new/cold запросов {measured['strata']['repeat']['query_count']}/"
                  f"{measured['strata']['new']['query_count']}/{measured['strata']['cold']['query_count']}. "
                  'Во всех оценках пропуски retrieval остаются нулями. Обучение условно на найденных целях; остальные кандидаты — unmarked, не доказанные отрицательные реакции.', '']
        for family in ['A1','A2']:
            winner = selected['winners'][family]
            lines.append(f"- {family}: выбран `{winner['label']}`, {len(winner['columns'])} столбцов, inner NDCG20={winner['inner_score']:.6f}.")
            for column in winner['columns']:
                spec = specs[column]
                selections.append({'cohort_seed':seed,'family':family,'feature_id':spec['feature_id'],
                                   'group':spec['group'],'op':spec['op'],'window_raw':spec['window']})
        lines += ['', f"Union использует `{selected['chosen_auto_by_inner']['label']}` по внутреннему критерию; "
                  f"{len(selected['final_masks']['Union'])} столбцов после добавления к manual66 и удаления fit-дублей.", '']
        contributions = {}
        for label in CORE:
            model = by_label[label]
            with np.load(folder/model['prediction_file']) as pred:
                contributions[label] = {int(qid):query_contributions(by_id[int(qid)],pred['scores'][pred['offsets'][i]:pred['offsets'][i+1]])
                                        for i,qid in enumerate(pred['query_ids'])}
        for label in ['Auto_A1','Auto_A2','Union']:
            by_user = defaultdict(list)
            for qid,value in contributions[label].items():
                base = contributions['M_UIDC'][qid]
                row = {'cohort_seed':seed,'model':label,'query_id':qid,'user_id':by_id[qid]['user_id'],
                       'manual_rank':base['rank'],'auto_rank':value['rank'],
                       'ndcg20_delta':value['ndcg20']-base['ndcg20'],'recall20_delta':value['recall20']-base['recall20']}
                paired.append(row)
                by_user[row['user_id']].append(row)
            local = []
            for uid, values in sorted(by_user.items()):
                row = {'cohort_seed':seed,'model':label,'user_id':uid,'queries':len(values),
                       'ndcg20_delta':float(np.mean([v['ndcg20_delta'] for v in values])),
                       'recall20_delta':float(np.mean([v['recall20_delta'] for v in values]))}
                users.append(row); local.append(row)
            deltas = [r['ndcg20_delta'] for r in local]
            lines.append(f"- {label} против ManualUIDC: ΔNDCG20={np.mean(deltas):+.6f}, "
                         f"ΔRecall20={np.mean([r['recall20_delta'] for r in local]):+.6f}; "
                         f"пользователей рост/снижение/равенство {sum(d>0 for d in deltas)}/{sum(d<0 for d in deltas)}/{sum(d==0 for d in deltas)}.")
        lines += ['', f"Стоимость child вместе с подготовкой: {status['elapsed_seconds']:.3f} с; "
                  f"baseline replay {status['baseline_build']['replay_seconds']:.3f} с, auto replay {status['build']['replay_seconds']:.3f} с; "
                  f"{status['fits_actual']}/28 фактических fits, peakRSS {status['resources']['peak_process_rss_mb']:.2f} МиБ, "
                  f"минимум свободной RAM {status['resources']['minimum_available_system_mb']:.2f} МиБ. "
                  f"Матрица {status['build']['shape'][0]}×206, {status['build']['matrix_values_bytes']/2**20:.2f} МиБ.", '']
    report = TASK/'report'
    for name,values in [('metrics',rows),('paired_queries',paired),('paired_users',users),('selected_expressions',selections)]:
        if values: write_csv(report/f'p005_{name}.csv',values)
    complete = [entry for entry,status in runs if status['status']=='complete']
    if complete:
        plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'T-Lab-P005'})
        fig,axes=plt.subplots(len(complete),2,figsize=(11,4*len(complete)),squeeze=False,layout='constrained')
        for index,entry in enumerate(complete):
            core={r['model']:r for r in rows if r['cohort_seed']==entry['seed']}
            for axis,key,title in zip(axes[index],['ndcg20','recall20'],['NDCG@20','Recall@20']):
                values=[core[label][key] for label in CORE]
                bars=axis.bar(CORE,values,color=['#718596','#93a4b2','#26799b','#dc8a3e','#43856e'])
                axis.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
                axis.set_title(f"Cohort {entry['seed']} · {title} · user-macro")
                axis.set_ylim(0,max(values)*1.24)
                axis.grid(axis='y',alpha=.2); axis.set_axisbelow(True)
                axis.tick_params(axis='x',labelrotation=14)
        fig.savefig(report/'p005_comparison.png',dpi=170)
        fig.savefig(report/'p005_comparison.svg',metadata={'Date':None})
        svg=report/'p005_comparison.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)
        lines += ['![Две репликации](p005_comparison.png)', '']
    lines += ['## Устойчивость и выбранные выражения', '']
    for label in ['Auto_A1','Auto_A2','Union']:
        local=[(entry['seed'],next(r for r in rows if r['cohort_seed']==entry['seed'] and r['model']==label)['ndcg20']-
                next(r for r in rows if r['cohort_seed']==entry['seed'] and r['model']=='M_UIDC')['ndcg20']) for entry in complete]
        lines.append(f"- {label}: ΔNDCG20 к ManualUIDC по когортам: "+', '.join(f'{seed}: {value:+.6f}' for seed,value in local)+'.')
    for family in ['A1','A2']:
        sets=[{r['feature_id'] for r in selections if r['cohort_seed']==entry['seed'] and r['family']==family} for entry in complete]
        if len(sets)==2:
            lines.append(f"- {family}: пересечение {len(sets[0]&sets[1])}, объединение {len(sets[0]|sets[1])} выбранных expressions; "
                         f"Jaccard {len(sets[0]&sets[1])/len(sets[0]|sets[1]):.4f}.")
    lines += ['', 'Частота выбора не доказывает причинный вклад признака. Состав и размер набора могут меняться вместе с когортой; '
              'малые разности на двух development-выборках не устанавливают популяционное превосходство. '
              'A2 против A1 одновременно меняет язык и выбранный поднабор; вклад отдельных операций здесь не изолирован. '
              'Общие ограничения: небольшой локальный каталог, отсутствие независимого календарного периода, неизвестная полнота историй, '
              'фиксированный mapping snapshot и ограниченное покрытие retrieval.', '',
              'P004 остаётся исторической development reference: ManualUIDC NDCG20=0.075126, A1=0.078346, A2=0.078076; '
              'Union=Manual. P005 показывает, повторяется ли направление этих различий при новом UID sampling, '
              'без переопределения метода после внешних метрик. P001 обязательно войдёт в итоговый отчёт; '
              'P003 scale256 остаётся failed и не заменяется успешными64.', '',
              '## Проверка и воспроизведение', '',
              f"Независимая проверка parent: `{verification['status']}`, когорт {verification['cohort_count']}. "
              'Verifier читает predictions/queries и пересчитывает метрики с item-ID tie break, проверяет временные split, '
              'selection, aliases, budgets и SHA256 inputs/sources/outputs. Компактный snapshot сохраняет rank evidence; '
              'проверка его rank table не означает повторного обучения или повторной проверки исходных scores.', '',
              '```powershell', '.venv\\Scripts\\python.exe task_01_auto_features/src/run_robustness.py --output task_01_auto_features/artifacts/p005_new_run',
              '.venv\\Scripts\\python.exe task_01_auto_features/src/verify_robustness_checkpoint.py --run artifacts/p005_new_run',
              '.venv\\Scripts\\python.exe task_01_auto_features/src/report_robustness.py --run task_01_auto_features/artifacts/p005_new_run',
              '.venv\\Scripts\\python.exe task_01_auto_features/src/verify_robustness_snapshot.py', '```', '',
              'Конфигурация: [p005_robustness.json](../configs/p005_robustness.json); '
              '[протокол](../configs/P005_PROTOCOL.md); [preflight](../research/P005_REVIEW.md); '
              '[компактная контрольная точка](p005_snapshot/README.md). '
              'CSV: [все варианты](p005_metrics.csv), [парные пользователи](p005_paired_users.csv), '
              '[парные запросы](p005_paired_queries.csv), [выражения](p005_selected_expressions.csv).', '',
              'Следующий этап — зафиксировать итоговый метод и отдельный протокол test с учётом всех development-результатов; '
              'не выбирать новые параметры по тесту. Работа2 начнётся после завершения работы1.']
    (report/'P005_ROBUSTNESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=TASK/'artifacts/p005_robustness')
    args=parser.parse_args()
    generate(args.run.resolve())


if __name__=='__main__': main()
