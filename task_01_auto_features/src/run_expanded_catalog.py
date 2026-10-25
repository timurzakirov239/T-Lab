"""P007: exact-gated full available catalog scoring of fixed trained models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from auto_features import AutoFeatureEngine, generate_specs
from auto_search import offsets_sizes
from expanded_catalog import evaluate_full_catalog
from expanded_metrics import summarize_ranks
from expanded_vector_features import FullCatalogFeatures
from run_expanded_search import CONFIG, PROTOCOL, TASK, read, load_queries, check_inputs, write_rows
from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
from run_retrieval import load_development_events


def replay(engine, events, position, timestamp, monitor):
    while position<len(events['timestamp']) and int(events['timestamp'][position])<timestamp:
        engine.observe({'uid':int(events['uid'][position]),'item_id':int(events['item_id'][position]),
            'timestamp':int(events['timestamp'][position]),
            'event_type':events['event_names'][int(events['event_code'][position])]})
        position+=1
        if position%5000==0:monitor.check()
    return position


def run(config, cohort, search, gate_only=False):
    from catboost import CatBoostClassifier
    search_status=read(search/'run_status.json')
    if search_status['status']!='complete':raise ValueError('Search/training incomplete')
    frozen_sha=sha256(search/'selection_frozen.json')
    if frozen_sha!=search_status['selection_frozen_sha256']:raise ValueError('Mask freeze drift')
    if sha256(CONFIG)!=search_status['config_sha256']:raise ValueError('Config drift')
    if sha256(PROTOCOL)!=search_status['protocol_sha256']:raise ValueError('Protocol drift')
    output=search/('catalog_gate' if gate_only else 'full_catalog')
    if output.exists():raise FileExistsError(output)
    if not gate_only:
        gate=read(search/'catalog_gate/run_status.json')
        if gate['status']!='complete' or gate['source_sha256']['expanded_vector_features.py']!=sha256(TASK/'src/expanded_vector_features.py'):
            raise ValueError('Full catalog requires successful exact gate with unchanged transformer')
    output.mkdir()
    started=time.perf_counter()
    monitor=ResourceMonitor(config['resource_limits']);monitor.start()
    status={'experiment':'P007','cohort':cohort['id'],'stage':'gate' if gate_only else 'full_catalog',
            'status':'running','final_test_evaluated':False,'selection_frozen_sha256':frozen_sha,
            'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),
            'source_sha256':{name:sha256(TASK/'src'/name) for name in
                ['expanded_vector_features.py','expanded_catalog.py','run_expanded_catalog.py','expanded_metrics.py']},
            'progress':{'queries_completed':0,'cells_compared':0,'catalog_rows_scored':0}}
    streams={}
    try:
        status['inputs_sha256']=check_inputs(cohort)
        queries=load_queries(cohort,config)
        offsets,sizes=offsets_sizes(queries)
        validation=[(i,q) for i,q in enumerate(queries) if q['split']=='validation']
        matrix=np.load(TASK/cohort['development_run']/'features.npy',mmap_mode='r')
        events=load_development_events(TASK/cohort['history'],config['boundaries_raw']['validation_end'])
        artists=load_mapping(TASK/cohort['mapping_dir']/'artist_mapping.parquet','artist_id')
        albums=load_mapping(TASK/cohort['mapping_dir']/'album_mapping.parquet','album_id')
        specs=generate_specs(config['windows_raw'],config['short_window_raw'])
        engine=AutoFeatureEngine(artists,albums,config['short_window_raw'],config['half_life_raw'],specs=specs)
        vector=FullCatalogFeatures(engine,events)
        del artists,albums
        models={};scorers={};columns={};old_rows={};old_scores={}
        masks=read(search/'selection_frozen.json')['final_masks']
        if not gate_only:
            for result in search_status['final_models']:
                label=result['label']
                if sha256(search/result['model_file'])!=result['model_sha256']:
                    raise ValueError('Trained model drift')
                key=result['model_file']
                if key not in models:
                    model=CatBoostClassifier();model.load_model(str(search/key));models[key]=model
                model=models[key]
                scorers[label]=lambda x,model=model:model.predict_proba(x,thread_count=2)[:,1]
                columns[label]=masks[label]
                with np.load(search/result['prediction_file']) as saved:
                    old_scores[label]=[saved['scores'][saved['offsets'][j]:saved['offsets'][j+1]].copy() for j in range(len(validation))]
                old_rows[label]=[json.loads(line) for line in (search/'ranks100'/f'{label}.jsonl').read_text().splitlines()]
            scorers['Popularity']=lambda x:x[:,0]
            columns['Popularity']=[[s['feature_id'] for s in specs].index('i_listen_all')]
            old_rows['Popularity']=[json.loads(line) for line in (search/'ranks100/Popularity.jsonl').read_text().splitlines()]
            for label in scorers:streams[label]=(output/f'{label}.jsonl').open('x',encoding='utf-8')
        pointer=0;digest=hashlib.sha256();catalog_sizes=[];next_log=started+15
        for j,(query_pos,q) in enumerate(validation):
            pointer=replay(engine,events,pointer,q['timestamp'],monitor)
            expected=np.asarray(matrix[offsets[query_pos]:offsets[query_pos]+sizes[query_pos]])
            actual=vector.transform(q['user_id'],q['candidate_items'],q['timestamp'])
            if actual.dtype!=np.float32 or actual.tobytes()!=expected.tobytes():
                mismatch=np.argwhere(actual!=expected)
                raise AssertionError(f'Exact frozen206 gate failed at query {q["query_id"]}: {mismatch[:5].tolist()}')
            digest.update(actual.tobytes());status['progress']['cells_compared']+=int(actual.size)
            if not gate_only:
                catalog=np.asarray(sorted(engine._items),dtype=np.int64)
                if not set(q['candidate_items']).issubset(set(catalog.tolist())):
                    raise AssertionError('Original candidates outside full causal catalog')
                all_features=vector.transform(q['user_id'],catalog.tolist(),q['timestamp'])
                # Recomputed scores for every original candidate must be identical.
                at=np.searchsorted(catalog,q['candidate_items'])
                if all_features[at].tobytes()!=expected.tobytes():
                    raise AssertionError('Full vs K100 feature gate failed')
                for label in old_scores:
                    predicted=scorers[label](all_features[np.ix_(at,columns[label])])
                    if not np.array_equal(predicted,old_scores[label][j]):
                        raise AssertionError(f'Score transfer drift: {label}/{q["query_id"]}')
                def transform(_engine,uid,item_ids,timestamp):
                    return all_features[np.searchsorted(catalog,item_ids)]
                evaluated=evaluate_full_catalog(engine,q['user_id'],q['timestamp'],q['target_item'],scorers,
                    columns=columns,chunk_size=config['catalog_chunk'],transform_fn=transform,monitor=monitor)
                catalog_sizes.append(evaluated['catalog_size'])
                for label,ranked in evaluated['models'].items():
                    before=old_rows[label][j]['target_rank']
                    if before is not None and (ranked['target_rank'] is None or ranked['target_rank']<before):
                        raise AssertionError('Full-catalog rank improved within strict candidate superset')
                    row={k:q[k] for k in ('query_id','user_id','timestamp','target_is_repeat','target_known_before')} | {
                        'cohort':cohort['id'],'candidate_count':evaluated['catalog_size'],**ranked}
                    streams[label].write(json.dumps(row,separators=(',',':'))+'\n')
                status['progress']['catalog_rows_scored']+=evaluated['catalog_size']
                del all_features,evaluated
            status['progress']['queries_completed']=j+1
            monitor.check()
            if time.perf_counter()>next_log:
                print(f'P007 {cohort["id"]} {status["stage"]}: {j+1}/{len(validation)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
                write_json(output/'run_status.json',status);next_log=time.perf_counter()+15
        for stream in streams.values():stream.close()
        streams={}
        status['exact_gate']={'identical_float32_bytes':True,'cells':status['progress']['cells_compared'],
                              'values_sha256':digest.hexdigest(),'queries':len(validation)}
        if not gate_only:
            metrics={label:summarize_ranks([json.loads(line) for line in (output/f'{label}.jsonl').read_text().splitlines()]) for label in scorers}
            write_json(output/'metrics.json',metrics)
            status['catalog_size']={'min':min(catalog_sizes),'max':max(catalog_sizes),'mean':float(np.mean(catalog_sizes))}
            status['output_sha256']={p.name:sha256(p) for p in output.glob('*.jsonl')}
            status['all_k100_scores_exact']=True
        status['status']='complete'
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for stream in streams.values():stream.close()
        status['elapsed_seconds']=round(time.perf_counter()-started,6)
        status['resources']=monitor.finish()
        write_json(output/'run_status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True)
    parser.add_argument('--root',type=Path,default=TASK/'artifacts/p007_expanded_search')
    parser.add_argument('--gate-only',action='store_true')
    args=parser.parse_args();config=read(CONFIG)
    cohort=next(c for c in config['cohorts'] if c['id']==args.cohort)
    run(config,cohort,args.root/cohort['id'],args.gate_only)


if __name__=='__main__':main()
