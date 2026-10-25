"""P007 exploratory selectors; frozen P001-P006 inputs are read-only."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from auto_features import AutoFeatureEngine, generate_specs
from auto_search import temporal_split, offsets_sizes, prune_columns, pick_best
from expanded_search import (importance_proposals, random_proposals,
    forward_group_proposals, query_group_permutation, stability_importance)
from expanded_metrics import summarize_ranks
from run_auto_features import candidate_rows, fit_and_predict
from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
from run_retrieval import load_development_events

TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK / 'configs/p007_expanded.json'
PROTOCOL = TASK / 'configs/P007_PROTOCOL.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_queries(cohort, config):
    queries = [json.loads(line) for line in (TASK/cohort['development_queries']).read_text(encoding='utf-8').splitlines() if line]
    end = config['boundaries_raw']['validation_end']
    if any(q['timestamp'] >= end or q['split'] not in ('train', 'validation') for q in queries):
        raise ValueError('P007 queries must remain strictly inside development')
    if queries != sorted(queries, key=lambda q: (q['timestamp'], q['user_id'], q['target_item'])):
        raise ValueError('Frozen query order changed')
    return queries


def check_inputs(cohort):
    manifest = read(TASK/'configs/p006_freeze_manifest.json')['files']
    run = cohort['development_run']
    paths = [f'{run}/features.npy', f'{run}/feature_inputs.jsonl',
             f'{run}/feature_schema.json', f'{run}/selection_frozen.json',
             cohort['development_queries'], cohort['history']]
    for relative in paths:
        if relative not in manifest or sha256(TASK/relative) != manifest[relative]:
            raise ValueError(f'Frozen input mismatch: {relative}')
    return {p: manifest[p] for p in paths}


def quick_ndcg(queries, scores):
    """Inner-only user-macro NDCG20 without repeated full metric trees."""
    from collections import defaultdict
    by_user = defaultdict(list)
    offset = 0
    for q in queries:
        items = np.asarray(q['candidate_items'])
        values = scores[offset:offset+len(items)]
        offset += len(items)
        positions = np.flatnonzero(items == q['target_item'])
        score = 0.0
        if len(positions):
            p = positions[0]
            rank = 1 + int(np.sum((values > values[p]) | ((values == values[p]) & (items < items[p]))))
            if rank <= 20:
                score = float(1/np.log2(rank+1))
        by_user[q['user_id']].append(score)
    if offset != len(scores):
        raise ValueError('Inner score row mismatch')
    return float(np.mean([np.mean(v) for v in by_user.values()]))


def ranking_rows(cohort, queries, offsets, scores):
    rows = []
    for j, q in enumerate(queries):
        items = np.asarray(q['candidate_items'])
        values = scores[offsets[j]:offsets[j+1]]
        order = np.lexsort((items, -values))
        positions = np.flatnonzero(items[order] == q['target_item'])
        rank = int(positions[0])+1 if len(positions) else None
        target_idx = q['candidate_items'].index(q['target_item']) if rank is not None else None
        rows.append({k: q[k] for k in ('query_id','user_id','timestamp','target_is_repeat','target_known_before')} |
            {'cohort':cohort, 'candidate_count':len(items), 'target_rank':rank,
             'target_score':float(values[target_idx]) if target_idx is not None else None,
             'top_items':[{'item_id':int(items[i]), 'score':float(values[i])} for i in order[:20]]})
    return rows


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',',':'))+'\n')


def source_hashes():
    names = ['run_expanded_search.py','expanded_search.py','expanded_metrics.py',
             'expanded_catalog.py','expanded_vector_features.py','auto_features.py',
             'pilot_features.py','auto_search.py','run_auto_features.py']
    return {name:sha256(TASK/'src'/name) for name in names if (TASK/'src'/name).is_file()}


def run_search(config, cohort, output):
    from catboost import CatBoostClassifier
    if output.exists():
        raise FileExistsError(output)
    for folder in [output, output/'models', output/'predictions', output/'results', output/'ranks100']:
        folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    monitor = ResourceMonitor(config['resource_limits']); monitor.start()
    status = {'experiment':'P007', 'cohort':cohort['id'], 'status':'running', 'attempts':[],
              'config':config, 'config_sha256':sha256(CONFIG), 'protocol_sha256':sha256(PROTOCOL),
              'source_sha256':source_hashes(), 'final_test_evaluated':False,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()}
    try:
        status['inputs_sha256'] = check_inputs(cohort)
        queries = load_queries(cohort, config)
        specs = generate_specs(config['windows_raw'], config['short_window_raw'])
        matrix = np.load(TASK/cohort['development_run']/'features.npy', mmap_mode='r')
        schema = read(TASK/cohort['development_run']/'feature_schema.json')
        if matrix.shape != (sum(len(q['candidate_items']) for q in queries),206):
            raise ValueError('Frozen feature matrix shape mismatch')
        if schema['feature_names'] != [s['feature_id'] for s in specs]:
            raise ValueError('Frozen feature order mismatch')
        inputs = [json.loads(line) for line in (TASK/cohort['development_run']/'feature_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
        for q, saved in zip(queries, inputs, strict=True):
            if any(q[k] != saved[k] for k in ('query_id','user_id','timestamp','candidate_items')):
                raise ValueError('Query/matrix structural mismatch')
        del inputs
        offsets, sizes = offsets_sizes(queries)
        split = temporal_split(queries)
        fit_rows = candidate_rows(split['inner_fit'], offsets, sizes)
        available, removed = prune_columns(matrix, fit_rows, list(range(206)))
        write_json(output/'pruning.json', {'available':available,'removed':removed})
        groups = {i:s['group'] for i,s in enumerate(specs)}
        def fit(label, cols, positions=split['inner_fit'], evaluate=split['inner_selection'], stage='inner'):
            return fit_and_predict(label, cols, positions, evaluate, matrix, queries, offsets, sizes,
                                   specs, config, output, monitor, status, stage)
        def choose(results):
            return dict(pick_best([{'label':r['label'],'columns':r['columns'],'spec_ids':r['spec_ids'],
                'inner_score':r['metrics']['overall']['user_macro']['ndcg@20']} for r in results]))
        full = fit('common_full_inner', available)
        full_importance = dict(zip(available, full['importance_fit_only']))
        masks = {'M_I':[i for i,s in enumerate(specs[:66]) if s['group']=='I'], 'M_UIDC':list(range(66))}
        original = read(TASK/cohort['development_run']/'selection_frozen.json')['final_masks']
        masks.update(Original_A1=original['Auto_A1'], Original_A2=original['Auto_A2'])
        searches = {}
        def trials(method, proposals, include_common=False):
            results = [full] if include_common else []
            for label, cols in proposals.items():
                if include_common and cols == available:
                    continue
                results.append(fit(f'{method}_{label}', cols))
            winner = choose(results)
            masks[method] = winner['columns']
            searches[method] = {'winner':winner,'trials':[
                {'label':r['label'],'columns':r['columns'],'inner_ndcg20':r['metrics']['overall']['user_macro']['ndcg@20']} for r in results]}
            if len(results)>config['maximum_search_fits_per_method']:
                raise ValueError('Per-method fit budget exceeded')
            write_json(output/'search_progress.json', searches)
        seed = config['search_seed'] + [c['id'] for c in config['cohorts']].index(cohort['id'])
        trials('Random206', random_proposals(available,n_proposals=config['random_trials'],seed=seed))
        # An importance fit on a shorter prefix, never on outer validation.
        train = [i for i,q in enumerate(queries) if q['split']=='train']
        cutoff = int(np.quantile([queries[i]['timestamp'] for i in train],config['stability_prefix_quantile'],method='lower'))
        early = np.asarray([i for i in train if queries[i]['timestamp']<cutoff])
        earlier = fit('stable_early_prefix', available, positions=early)
        stable = stability_importance(dict(zip(available,earlier['importance_fit_only'])),full_importance,available)
        trials('Stable206', importance_proposals(available,stable,groups),include_common=True)
        # Query-block permutations preserve the candidate-rank position inside a block.
        common_model = CatBoostClassifier(); common_model.load_model(str(output/full['model_file']))
        sel_positions = split['inner_selection']; sel_queries = [queries[int(i)] for i in sel_positions]
        selected_rows = candidate_rows(sel_positions,offsets,sizes)
        x = np.asarray(matrix[np.ix_(selected_rows,available)]).copy()
        local_offsets = np.cumsum([0]+[len(q['candidate_items']) for q in sel_queries])
        baseline_score = quick_ndcg(sel_queries,common_model.predict_proba(x,thread_count=2)[:,1])
        permutation_scores = {}
        for local_col,col in enumerate(available):
            saved = x[:,local_col].copy()
            permutation = query_group_permutation(local_offsets,seed=seed+col)
            x[:,local_col] = saved[permutation]
            permutation_scores[col] = baseline_score-quick_ndcg(sel_queries,common_model.predict_proba(x,thread_count=2)[:,1])
            x[:,local_col] = saved
            monitor.check()
            if local_col%32==0:
                print(f'P007 {cohort["id"]}: permutation {local_col+1}/{len(available)}',flush=True)
        del x, selected_rows, common_model
        gc.collect()
        write_json(output/'importance_diagnostics.json',{'permutation':permutation_scores,'stability':stable,
                   'permutation_baseline_inner_ndcg20':baseline_score,'early_cutoff':cutoff})
        trials('Permutation206',importance_proposals(available,permutation_scores,groups),include_common=True)
        group_columns = {g:[i for i,s in enumerate(specs[:66]) if s['group']==g] for g in 'UIDC'}
        selected_groups=[]; visited=[]; steps=[]
        while len(selected_groups)<4:
            candidates=forward_group_proposals(group_columns,selected_groups)
            results=[fit(f'Forward_s{len(selected_groups)}_{name}',cols) for name,cols in candidates.items()]
            winner=choose(results); visited.extend(results)
            chosen_group=winner['label'].split('_')[-1]
            selected_groups.append(chosen_group)
            steps.append({'added':chosen_group,'winner':winner})
        chosen=choose(visited);masks['ForwardGroups66']=chosen['columns']
        searches['ForwardGroups66']={'winner':chosen,'steps':steps,'fits':len(visited)}
        frozen={'experiment':'P007','cohort':cohort['id'],'final_masks':masks,'searches':searches,
                'spec_ids':{label:[specs[i]['feature_id'] for i in cols] for label,cols in masks.items()},
                'outer_evaluation_started':False,'full_catalog_started':False}
        write_json(output/'selection_frozen.json',frozen)
        frozen_sha=sha256(output/'selection_frozen.json')
        status['selection_frozen_sha256']=frozen_sha
        evaluation=split['external_validation']; eval_queries=[queries[int(i)] for i in evaluation]
        final_results=[]; cache={}
        for label, cols in masks.items():
            if sha256(output/'selection_frozen.json')!=frozen_sha:
                raise ValueError('Frozen selection changed')
            signature=tuple(cols)
            if signature in cache:
                result=dict(cache[signature],label=label,alias_of=cache[signature]['label'])
            else:
                result=fit(label,cols,positions=train,evaluate=evaluation,stage='outer_validation')
                cache[signature]=result
            final_results.append(result)
            with np.load(output/result['prediction_file']) as pred:
                rows=ranking_rows(cohort['id'],eval_queries,pred['offsets'],pred['scores'])
            recomputed=summarize_ranks(rows)
            for metric in ('ndcg@20','recall@20'):
                if abs(recomputed['overall']['user_macro'][metric]-result['metrics']['overall']['user_macro'][metric])>1e-12:
                    raise AssertionError('Independent rank metric mismatch')
            write_rows(output/'ranks100'/f'{label}.jsonl',rows)
        # Popularity uses the stored exact causal listen-count feature.
        pop_col=[s['feature_id'] for s in specs].index('i_listen_all')
        eval_rows=candidate_rows(evaluation,offsets,sizes)
        pop_scores=np.asarray(matrix[eval_rows,pop_col],dtype=float)
        pop_rows=ranking_rows(cohort['id'],eval_queries,np.cumsum([0]+[len(q['candidate_items']) for q in eval_queries]),pop_scores)
        write_rows(output/'ranks100/Popularity.jsonl',pop_rows)
        write_json(output/'metrics100.json',{label:summarize_ranks([json.loads(line) for line in (output/'ranks100'/f'{label}.jsonl').read_text().splitlines()]) for label in [*masks,'Popularity']})
        status.update(status='complete',final_models=final_results,searches=searches,
                      query_counts={'train':len(train),'validation':len(evaluation)},fits_actual=len(status['attempts']))
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['elapsed_seconds']=round(time.perf_counter()-started,6)
        status['resources']=monitor.finish()
        write_json(output/'run_status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True)
    parser.add_argument('--output',type=Path,default=TASK/'artifacts/p007_expanded_search')
    args=parser.parse_args()
    config=read(CONFIG)
    cohort=next(c for c in config['cohorts'] if c['id']==args.cohort)
    run_search(config,cohort,args.output/cohort['id'])


if __name__=='__main__':main()
