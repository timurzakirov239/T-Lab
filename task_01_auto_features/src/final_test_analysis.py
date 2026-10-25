"""Frozen user-level analysis for P006; no model or feature selection."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np

if __package__:
    from .verify_auto_checkpoint import recompute
else:
    from verify_auto_checkpoint import recompute


def user_bootstrap(deltas,resamples=2000,seed=20261101,confidence=.95):
    values=np.asarray(deltas,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires finite per-user deltas')
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),size=(resamples,len(values)))].mean(axis=1)
    alpha=(1-confidence)/2
    lo,hi=np.quantile(draws,[alpha,1-alpha],method='linear')
    return {'users':len(values),'mean':float(values.mean()),'ci_lower':float(lo),'ci_upper':float(hi),
            'confidence':confidence,'resamples':resamples,'seed':seed,'quantile_method':'linear',
            'positive_users':int(sum(values>0)),'negative_users':int(sum(values<0)),
            'equal_users':int(sum(values==0))}


def contribution(query,scores):
    order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),query['candidate_items'][i]))
    target=query['target_item']
    rank=next((position+1 for position,index in enumerate(order) if query['candidate_items'][index]==target),None)
    hit=rank is not None and rank<=20
    return {'rank':rank,'ndcg20':1/math.log2(rank+1) if hit else 0.,'recall20':float(hit)}


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(root,config):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] not in ['complete','complete_with_failures']:
        raise ValueError('Final analysis requires finished test parent')
    labels=['Popularity',*config['model_labels']]
    pooled_queries={label:[] for label in labels};pooled_scores={label:[] for label in labels}
    metric_rows=[];user_rows=[];query_rows=[];cohorts=[]
    for entry in summary['cohorts']:
        folder=root/entry['run_dir']
        status=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if status['status']!='complete':
            cohorts.append({'id':entry['id'],'status':'failed','error':status.get('error')});continue
        queries=[json.loads(line) for line in (folder/'queries.jsonl').read_text(encoding='utf-8').splitlines()]
        values={}
        for model in status['final_models']:
            label=model['label']
            with np.load(folder/model['prediction_file']) as pred:
                if pred['query_ids'].tolist()!=[q['query_id'] for q in queries]: raise ValueError('Prediction query order drift')
                scores=[pred['scores'][pred['offsets'][i]:pred['offsets'][i+1]].copy() for i in range(len(queries))]
            pooled_queries[label].extend(queries);pooled_scores[label].extend(scores)
            values[label]={q['query_id']:contribution(q,score) for q,score in zip(queries,scores)}
            metric_rows.append({'cohort':entry['id'],'model':label,'features':model['feature_count'],
                                'users':model['metrics']['overall']['user_count'],'queries':len(queries),
                                'ndcg20':model['metrics']['overall']['user_macro']['ndcg@20'],
                                'recall20':model['metrics']['overall']['user_macro']['recall@20']})
        for label in ['Auto_A1','Auto_A2','Union']:
            grouped=defaultdict(list)
            for query in queries:
                qid=query['query_id'];base=values['M_UIDC'][qid];other=values[label][qid]
                row={'cohort':entry['id'],'model':label,'query_id':qid,'user_id':query['user_id'],
                     'manual_rank':base['rank'],'auto_rank':other['rank'],
                     'ndcg20_delta':other['ndcg20']-base['ndcg20'],
                     'recall20_delta':other['recall20']-base['recall20']}
                query_rows.append(row);grouped[query['user_id']].append(row)
            for uid,rows in sorted(grouped.items()):
                user_rows.append({'cohort':entry['id'],'model':label,'user_id':uid,'queries':len(rows),
                                  'ndcg20_delta':float(np.mean([r['ndcg20_delta'] for r in rows])),
                                  'recall20_delta':float(np.mean([r['recall20_delta'] for r in rows]))})
        cohorts.append({'id':entry['id'],'status':'complete','query_selection':status['query_selection'],
                        'resources':status['resources'],'elapsed_seconds':status['elapsed_seconds'],
                        'fits_actual':status['fits_actual']})
    pooled={}
    if metric_rows:
        for label in labels:
            # Query IDs are local; global IDs are assigned solely for the scorer.
            queries=[dict(q,query_id=i) for i,q in enumerate(pooled_queries[label])]
            pooled[label]=recompute(queries,pooled_scores[label])
            metric_rows.append({'cohort':'pooled','model':label,'features':'cohort-specific',
                                'users':pooled[label]['overall']['user_count'],'queries':len(queries),
                                'ndcg20':pooled[label]['overall']['user_macro']['ndcg@20'],
                                'recall20':pooled[label]['overall']['user_macro']['recall@20']})
    intervals={}
    for context in [c['id'] for c in cohorts if c['status']=='complete']+(['pooled'] if metric_rows else []):
        local=sorted([r for r in user_rows if r['model']=='Auto_A2' and (context=='pooled' or r['cohort']==context)],key=lambda r:r['user_id'])
        intervals[context]={metric:user_bootstrap([r[f'{metric}_delta'] for r in local],**config['bootstrap'])
                            for metric in ['ndcg20','recall20']}
    result={'status':summary['status'],'primary_comparison':config['primary_comparison'],'cohorts':cohorts,
            'pooled_metrics':pooled,'primary_paired_user_bootstrap':intervals,
            'interval_scope':'Conditional descriptive paired-user bootstrap; no refitting, no cohort/model uncertainty',
            'final_test_evaluated':summary['final_test_evaluated']}
    (root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,rows in [('metrics',metric_rows),('paired_users',user_rows),('paired_queries',query_rows)]:
        if rows: write_csv(root/f'{name}.csv',rows)
    return result
