"""P008 fixed-feature training with uniform and teacher-mined negatives."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from auto_features import AutoFeatureEngine, generate_specs
from auto_search import offsets_sizes
from expanded_catalog import evaluate_full_catalog
from expanded_metrics import summarize_ranks
from expanded_vector_features import FullCatalogFeatures
from run_auto_features import MemoryCallback
from run_expanded_catalog import replay
from run_expanded_search import load_queries, check_inputs, ranking_rows
from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
from run_retrieval import load_development_events

TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK/'configs/p008_sampled.json'
PROTOCOL = TASK/'configs/P008_PROTOCOL.md'
LABELS = ['Baseline_UIDC','Baseline_A1','Uniform_UIDC','Uniform_A1',
          'HybridHit_UIDC','HybridHit_A1','HybridWarm_UIDC','HybridWarm_A1','Popularity']
REFERENCE = {'Baseline_UIDC':'M_UIDC','Baseline_A1':'Original_A1','Popularity':'Popularity'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def sources():
    names=['run_sampled_training.py','prepare_sampled_training.py','sampled_negatives.py',
           'auto_features.py','pilot_features.py','expanded_vector_features.py',
           'expanded_catalog.py','expanded_metrics.py']
    return {name:sha256(TASK/'src'/name) for name in names}


def begin(output,cohort,stage,config):
    output.mkdir()
    status={'experiment':'P008','cohort':cohort['id'],'stage':stage,'status':'running',
            'final_test_evaluated':False,'config_sha256':sha256(CONFIG),
            'protocol_sha256':sha256(PROTOCOL),'source_sha256':sources(),
            'progress':{},'models':[]}
    started=time.perf_counter()
    monitor=ResourceMonitor(config['resource_limits']);monitor.start()
    write_json(output/'run_status.json',status)
    return status,started,monitor


def finish(output,status,started,monitor):
    status['elapsed_seconds']=round(time.perf_counter()-started,6)
    status['resources']=monitor.finish()
    write_json(output/'run_status.json',status)


def require_complete(folder):
    status=read(folder/'run_status.json')
    if status['status']!='complete':raise ValueError(f'Incomplete prerequisite: {folder}')
    if status['config_sha256']!=sha256(CONFIG) or status['protocol_sha256']!=sha256(PROTOCOL):
        raise ValueError(f'Config/protocol drift: {folder}')
    for name,expected in status['source_sha256'].items():
        source=TASK/name if '/' in name else TASK/'src'/name
        if sha256(source)!=expected:raise ValueError(f'Source drift: {name}')
    for relative,expected in status.get('outputs_sha256',{}).items():
        if sha256(folder/relative)!=expected:raise ValueError(f'Prerequisite output drift: {relative}')
    return status


def reference_models(config,cohort):
    folder=TASK/config['p007_root']/cohort['id']
    saved=read(folder/'run_status.json')
    if saved['status']!='complete':raise ValueError('P007 incomplete')
    masks=read(folder/'selection_frozen.json')['final_masks']
    results={row['label']:row for row in saved['final_models']}
    models=[]
    for label,old in [('Baseline_UIDC','M_UIDC'),('Baseline_A1','Original_A1')]:
        record=results[old];path=folder/record['model_file']
        if sha256(path)!=record['model_sha256']:raise ValueError('P007 model drift')
        cols=masks[old]
        if not cols or max(cols)>=66:raise ValueError('P008 fixed masks must be within66')
        models.append({'label':label,'arm':'Baseline','columns':cols,'model_origin':'p007',
                       'model_file':path.relative_to(TASK).as_posix(),'model_sha256':sha256(path),
                       'training':record['training']})
    return models


def require_all_fits(config,root):
    """Freeze all eighteen fits before any new validation feature/score pass."""
    ledger={'experiment':'P008','frozen_before_evaluation':True,
            'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),
            'source_sha256':sources(),'cohort_fit_status_sha256':{}}
    for cohort in config['cohorts']:
        folder=root/cohort['id']/'fit'
        fitted=require_complete(folder)
        models=fitted.get('models',[])
        if len(models)!=8 or sum(r['model_origin']=='p008' for r in models)!=6:
            raise ValueError(f'All six new fits required before validation: {cohort["id"]}')
        if sha256(folder/'training_frozen.json')!=fitted['training_frozen_sha256']:
            raise ValueError('Training freeze changed before evaluation')
        ledger['cohort_fit_status_sha256'][cohort['id']]=sha256(folder/'run_status.json')
    path=root/'all_fits_frozen.json'
    if path.exists():
        if read(path)!=ledger:raise ValueError('All-fit freeze drift')
    else:
        write_json(path,ledger)
    return sha256(path)


def fit(config,cohort,folder):
    from catboost import CatBoostClassifier
    prepared=require_complete(folder/'prepare')
    records=jsonl(folder/'prepare/records.jsonl')
    warm=[r for r in records if r['warm']]
    hits=[r for r in warm if r['old_hit']]
    if len(hits)!=prepared['query_counts']['hit'] or len(warm)!=prepared['query_counts']['warm']:
        raise ValueError('Eligibility count drift')
    baseline=reference_models(config,cohort)
    masks={'UIDC':baseline[0]['columns'],'A1':baseline[1]['columns']}
    if masks!=prepared['masks']:raise ValueError('Fixed mask drift')
    output=folder/'fit'
    status,started,monitor=begin(output,cohort,'fit',config)
    (output/'models').mkdir()
    status['inputs_sha256']=check_inputs(cohort)
    status['prepare_status_sha256']=sha256(folder/'prepare/run_status.json')
    uniform=np.load(folder/'prepare/uniform.npy',mmap_mode='r')
    hybrid=np.load(folder/'prepare/hybrid.npy',mmap_mode='r')
    status['models']=baseline
    frozen={'experiment':'P008','cohort':cohort['id'],'masks':masks,
            'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),
            'prepare_status_sha256':status['prepare_status_sha256'],
            'arms':{'Uniform':[r['query_id'] for r in hits],
                    'HybridHit':[r['query_id'] for r in hits],
                    'HybridWarm':[r['query_id'] for r in warm]}}
    write_json(output/'training_frozen.json',frozen)
    status['training_frozen_sha256']=sha256(output/'training_frozen.json')
    try:
        for arm,chosen,matrix,row_key,item_key in [
                ('Uniform',hits,uniform,'uniform_row_start','uniform_items'),
                ('HybridHit',hits,hybrid,'hybrid_row_start','hybrid_items'),
                ('HybridWarm',warm,hybrid,'hybrid_row_start','hybrid_items')]:
            rows=np.concatenate([np.arange(r[row_key],r[row_key]+100) for r in chosen])
            labels=np.concatenate([np.asarray(r[item_key])==r['target_item'] for r in chosen]).astype(np.int8)
            if int(labels.sum())!=len(chosen) or len(labels)!=100*len(chosen):
                raise ValueError('Training positives/rows invalid')
            for name,columns in masks.items():
                if sha256(output/'training_frozen.json')!=status['training_frozen_sha256']:
                    raise ValueError('Training specification drift')
                label=f'{arm}_{name}';tick=time.perf_counter();monitor.check()
                model=CatBoostClassifier(**config['model'],random_seed=config['seed'])
                x=np.asarray(matrix[np.ix_(rows,columns)])
                model.fit(x,labels,callbacks=[MemoryCallback(monitor)]);monitor.check()
                path=output/'models'/f'{label}.cbm';model.save_model(str(path))
                record={'label':label,'arm':arm,'columns':columns,'model_origin':'p008',
                        'model_file':path.relative_to(output).as_posix(),'model_sha256':sha256(path),
                        'training_query_ids':[r['query_id'] for r in chosen],
                        'training':{'queries':len(chosen),'rows':len(labels),'positives':int(labels.sum())},
                        'fit_seconds':round(time.perf_counter()-tick,6)}
                status['models'].append(record)
                status['progress']['fits_completed']=len(status['models'])-2
                write_json(output/'run_status.json',status)
                print(f'P008 {cohort["id"]} fit {label}: {len(chosen)} queries, {record["fit_seconds"]:.2f}s',flush=True)
                del model,x;gc.collect()
        status['status']='complete'
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        finish(output,status,started,monitor)
    return status


def evaluate(config,cohort,folder,gate_only=False):
    from catboost import CatBoostClassifier
    all_fit_sha=require_all_fits(config,folder.parent)
    fitted=require_complete(folder/'fit')
    stage='gate' if gate_only else 'full'
    if not gate_only:require_complete(folder/'gate')
    output=folder/stage
    status,started,monitor=begin(output,cohort,stage,config)
    status['all_fits_frozen_sha256']=all_fit_sha
    streams={};buffers={};models={}
    try:
        status['inputs_sha256']=check_inputs(cohort)
        status['fit_status_sha256']=sha256(folder/'fit/run_status.json')
        queries=load_queries(cohort,config);offsets,sizes=offsets_sizes(queries)
        validation=[(i,q) for i,q in enumerate(queries) if q['split']=='validation']
        matrix=np.load(TASK/cohort['development_run']/'features.npy',mmap_mode='r')
        events=load_development_events(TASK/cohort['history'],config['boundaries_raw']['validation_end'])
        artists=load_mapping(TASK/cohort['mapping_dir']/'artist_mapping.parquet','artist_id')
        albums=load_mapping(TASK/cohort['mapping_dir']/'album_mapping.parquet','album_id')
        specs=generate_specs(config['windows_raw'],config['short_window_raw'])[:66]
        engine=AutoFeatureEngine(artists,albums,config['short_window_raw'],config['half_life_raw'],specs=specs)
        vector=FullCatalogFeatures(engine,events)
        del artists,albums
        scorers={};columns={};reference={}
        if not gate_only:
            for sub in ('ranks100','ranksfull','predictions'):(output/sub).mkdir()
            for record in fitted['models']:
                path=(TASK if record['model_origin']=='p007' else folder/'fit')/record['model_file']
                if sha256(path)!=record['model_sha256']:raise ValueError('Fitted model drift')
                model=CatBoostClassifier();model.load_model(str(path));models[record['label']]=model
                scorers[record['label']]=lambda x,model=model:model.predict_proba(x,thread_count=2)[:,1]
                columns[record['label']]=record['columns']
            columns['Popularity']=[engine.feature_names.index('i_listen_all')]
            scorers['Popularity']=lambda x:x[:,0].astype(float)
            for label in LABELS:
                buffers[label]=[]
                for mode in ('ranks100','ranksfull'):
                    streams[(mode,label)]=(output/mode/f'{label}.jsonl').open('x',encoding='utf-8')
            for label,old in REFERENCE.items():
                base=TASK/config['p007_root']/cohort['id']
                for mode,path in [('ranks100',base/'ranks100'/f'{old}.jsonl'),
                                  ('ranksfull',base/'full_catalog'/f'{old}.jsonl')]:
                    reference[(mode,label)]=jsonl(path)
        pointer=0;cells=0;catalog_rows=0;catalog_sizes=[];next_log=time.perf_counter()+15
        import hashlib
        digest=hashlib.sha256()
        for j,(position,q) in enumerate(validation):
            pointer=replay(engine,events,pointer,q['timestamp'],monitor)
            actual=vector.transform(q['user_id'],q['candidate_items'],q['timestamp'])
            expected=np.asarray(matrix[offsets[position]:offsets[position]+sizes[position],:66])
            if actual.dtype!=np.float32 or actual.tobytes()!=expected.tobytes():
                raise AssertionError(f'P008 validation byte gate: {q["query_id"]}')
            cells+=actual.size;digest.update(actual.tobytes())
            if not gate_only:
                small={}
                for label in LABELS:
                    scores=np.asarray(scorers[label](actual[:,columns[label]]),dtype=float)
                    buffers[label].append(scores)
                    row=ranking_rows(cohort['id'],[q],np.array([0,len(scores)]),scores)[0]
                    if (('ranks100',label) in reference and row!=reference[('ranks100',label)][j]):
                        raise AssertionError(f'P007 K100 reference drift: {label}/{q["query_id"]}')
                    small[label]=row
                    streams[('ranks100',label)].write(json.dumps(row,separators=(',',':'))+'\n')
                original={item:i for i,item in enumerate(q['candidate_items'])};matched=set()
                def transform(_engine,uid,items,timestamp):
                    values=vector.transform(uid,items,timestamp)
                    for local,item in enumerate(items):
                        if item in original:
                            if values[local].tobytes()!=expected[original[item]].tobytes():
                                raise AssertionError('P008 chunk feature mismatch')
                            matched.add(item)
                    return values
                result=evaluate_full_catalog(engine,q['user_id'],q['timestamp'],q['target_item'],scorers,
                    columns=columns,chunk_size=config['catalog_chunk'],transform_fn=transform,monitor=monitor)
                if matched!=set(original):raise AssertionError('Original K100 items outside full catalog')
                catalog_rows+=result['catalog_size'];catalog_sizes.append(result['catalog_size'])
                for label,ranked in result['models'].items():
                    row={k:q[k] for k in ('query_id','user_id','timestamp','target_is_repeat','target_known_before')}
                    row.update(cohort=cohort['id'],candidate_count=result['catalog_size'],**ranked)
                    before=small[label]
                    if before['target_rank'] is not None:
                        if row['target_rank'] is None or row['target_rank']<before['target_rank']:
                            raise AssertionError('Nested rank improved')
                        if row['target_score']!=before['target_score']:raise AssertionError('Target score drift')
                    if ('ranksfull',label) in reference and row!=reference[('ranksfull',label)][j]:
                        raise AssertionError(f'P007 full reference drift: {label}/{q["query_id"]}')
                    streams[('ranksfull',label)].write(json.dumps(row,separators=(',',':'))+'\n')
                del result
            status['progress']={'queries_completed':j+1,'cells_compared':int(cells),'catalog_rows_scored':catalog_rows}
            monitor.check()
            if time.perf_counter()>next_log:
                print(f'P008 {cohort["id"]} {stage}: {j+1}/{len(validation)}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
                write_json(output/'run_status.json',status);next_log=time.perf_counter()+15
        for stream in streams.values():stream.close()
        streams={}
        status['exact_gate']={'identical_float32_bytes':True,'cells':int(cells),'queries':len(validation),
                              'values_sha256':digest.hexdigest()}
        if not gate_only:
            for label in LABELS:
                np.savez_compressed(output/'predictions'/f'{label}.npz',
                    query_ids=[q['query_id'] for _,q in validation],offsets=np.cumsum([0]+[len(s) for s in buffers[label]]),
                    scores=np.concatenate(buffers[label]))
            for mode,name in [('ranks100','metrics100.json'),('ranksfull','metrics_full.json')]:
                write_json(output/name,{label:summarize_ranks(jsonl(output/mode/f'{label}.jsonl')) for label in LABELS})
            status['catalog_size']={'min':min(catalog_sizes),'max':max(catalog_sizes),'mean':float(np.mean(catalog_sizes))}
            status['reference_parity_exact']=True
            status['outputs_sha256']={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='run_status.json'}
        status['status']='complete'
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        for stream in streams.values():stream.close()
        finish(output,status,started,monitor)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True);parser.add_argument('--stage',choices=['prepare','fit','gate','full'],required=True)
    parser.add_argument('--root',type=Path,default=TASK/'artifacts/p008_sampled_training')
    args=parser.parse_args();config=read(CONFIG)
    cohort=next(c for c in config['cohorts'] if c['id']==args.cohort)
    folder=args.root/cohort['id'];folder.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':
        from prepare_sampled_training import prepare
        prepare(config,cohort,folder/'prepare')
    elif args.stage=='fit':fit(config,cohort,folder)
    else:evaluate(config,cohort,folder,args.stage=='gate')


if __name__=='__main__':main()
