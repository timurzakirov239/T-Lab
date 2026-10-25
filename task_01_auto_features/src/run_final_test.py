"""P006: one reserved-period evaluation of development-frozen feature masks."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

if __package__:
    from .auto_features import AutoFeatureEngine, generate_specs
    from .auto_search import offsets_sizes, eligible_training_rows
    from .final_test_data import load_published_history, select_test_queries
    from .pilot_evaluation import evaluate_ranking
    from .pilot_features import FeatureEngine
    from .run_auto_features import MemoryCallback, candidate_rows
    from .run_feature_benchmark import feature_build
    from .run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from .run_robustness import assert_manual_equivalence
else:
    from auto_features import AutoFeatureEngine, generate_specs
    from auto_search import offsets_sizes, eligible_training_rows
    from final_test_data import load_published_history, select_test_queries
    from pilot_evaluation import evaluate_ranking
    from pilot_features import FeatureEngine
    from run_auto_features import MemoryCallback, candidate_rows
    from run_feature_benchmark import feature_build
    from run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from run_robustness import assert_manual_equivalence

TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK/'configs/p006_final_test.json'
PROTOCOL = TASK/'configs/P006_PROTOCOL.md'
MANIFEST = TASK/'configs/p006_freeze_manifest.json'
LOCK = TASK/'artifacts/p006_test_opened.json'
CONFIG_SHA = '4bb55f5ca215dbd1ee4563dbfa41397e52d310863f70dc2d2dfad9f777df9b23'
PROTOCOL_SHA = '2787a6928096e8cec13c37353f3f35b8269c7eb11d8977e4fa57a96430861724'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_inputs():
    if sha256(CONFIG)!=CONFIG_SHA or sha256(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError('P006 frozen config/protocol drift')
    # The manifest must already be committed; it pins runner/helper/input bytes.
    committed=subprocess.check_output(['git','show','HEAD:task_01_auto_features/configs/p006_freeze_manifest.json'],cwd=TASK)
    if committed != MANIFEST.read_bytes(): raise ValueError('Freeze manifest must be committed unchanged')
    manifest=read(MANIFEST)
    for name,digest in manifest['files'].items():
        if sha256(TASK/name)!=digest: raise ValueError(f'P006 input/source drift: {name}')
    config=read(CONFIG)
    seen=set()
    for cohort in config['cohorts']:
        members=read(TASK/cohort['membership_metadata'])['selected_users']
        if len(members)!=64 or len(set(members))!=64 or seen.intersection(members):
            raise ValueError('P006 membership overlap or count mismatch')
        seen.update(members)
        status=read(TASK/cohort['development_run']/'metrics.json')
        if status['status']!='complete' or status['final_test_evaluated']:
            raise ValueError('P006 requires completed development matrix')
        saved=read(TASK/cohort['snapshot']/'metrics.json')
        if status!=saved: raise ValueError('Development metrics differ from frozen snapshot')
    return config,manifest


def seal_candidates(events, queries, artists, albums, config, output, monitor):
    """The generator sees only uid/time; labels are attached after candidate freeze."""
    generator=CandidateGenerator(artists,albums,config)
    index=0
    for position,q in enumerate(queries):
        uid,t=q['user_id'],q['timestamp']
        while index<len(events['timestamp']) and int(events['timestamp'][index])<t:
            generator.observe({'uid':int(events['uid'][index]),'item_id':int(events['item_id'][index]),
                               'timestamp':int(events['timestamp'][index]),
                               'event_type':events['event_names'][int(events['event_code'][index])]})
            index+=1
            if index%5000==0: monitor.check()
        candidates=generator.candidates(uid,t)
        if len(set(candidates))!=len(candidates) or any(i not in generator.catalog for i in candidates):
            raise AssertionError('Invalid test candidates')
        q['candidate_items']=candidates
        q['popularity_scores']=[float(generator.counts[i]) for i in candidates]
        target=q['target_item']
        q['target_is_repeat']=target in generator.last_listen.get(uid,{})
        q['target_known_before']=target in generator.catalog
        if position%20==0: monitor.check()
    with (output/'queries.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for q in queries: stream.write(json.dumps(q,separators=(',',':'))+'\n')
    return {'strategy':'P001_original','quotas':config['candidate_quotas'],
            'candidate_rows':sum(len(q['candidate_items']) for q in queries),
            'past_events_observed':index,'queries_sha256':sha256(output/'queries.jsonl')}


def save_predictions(output,label,queries,scores):
    path=output/'predictions'/f'{label}.npz'
    np.savez_compressed(path,query_ids=np.array([q['query_id'] for q in queries]),
                        offsets=np.cumsum([0]+[len(s) for s in scores]),scores=np.concatenate(scores))
    return {'prediction_file':str(path.relative_to(output)),'prediction_sha256':sha256(path)}


def development_training_rows(queries,start):
    if any(q['split'] not in ['train','validation'] or q['timestamp']>=start for q in queries):
        raise ValueError('Final fit accepts only pre-test development blocks')
    offsets,sizes=offsets_sizes(queries)
    # The frozen helper accepts train labels only. This local view reclassifies
    # former validation blocks as fit data without mutating the saved queries.
    training_view=[dict(q,split='train') for q in queries]
    return eligible_training_rows(training_view,range(len(queries)),offsets,sizes)


def final_fit(label,columns,dev_matrix,dev_queries,test_matrix,test_queries,config,output,monitor,status):
    from catboost import CatBoostClassifier
    if len(status['attempts'])>=config['maximum_total_fits']: raise ValueError('Final fit budget exhausted')
    rows,labels,counts=development_training_rows(dev_queries,config['test_start_raw'])
    if len(np.unique(labels))!=2: raise ValueError('Final training lacks two classes')
    attempt={'label':label,'stage':'final_test_fit','status':'started','columns':columns,'training':counts}
    status['attempts'].append(attempt)
    write_json(output/'run_status.json',status)
    model=CatBoostClassifier(**config['model'],random_seed=config['seed'])
    tick=time.perf_counter()
    try:
        monitor.check()
        training=dev_matrix[np.ix_(rows,columns)]
        model.fit(training,labels,callbacks=[MemoryCallback(monitor)])
        monitor.check()
        fit_seconds=time.perf_counter()-tick
        del training,rows,labels
        gc.collect()
        test_offsets,test_sizes=offsets_sizes(test_queries)
        scores=[]
        for start in range(0,len(test_queries),64):
            positions=range(start,min(start+64,len(test_queries)))
            flat=candidate_rows(positions,test_offsets,test_sizes)
            predicted=model.predict_proba(test_matrix[np.ix_(flat,columns)],thread_count=2)[:,1]
            local=0
            for position in positions:
                size=int(test_sizes[position]);scores.append(predicted[local:local+size].copy());local+=size
            monitor.check()
        model_path=output/'models'/f'{label}.cbm'
        model.save_model(str(model_path))
        result={'label':label,'stage':'final_test','evaluation_period':'test','columns':columns,
                'feature_count':len(columns),'training':counts,'fit_seconds':round(fit_seconds,6),
                'model_file':str(model_path.relative_to(output)),'model_sha256':sha256(model_path),
                **save_predictions(output,label,test_queries,scores),
                'metrics':evaluate_ranking(test_queries,scores,ks=tuple(config['metrics_k']))}
        attempt.update(status='complete',fit_seconds=result['fit_seconds'])
        write_json(output/'results'/f'{label}.json',result)
        print(f"P006 {label}: features={len(columns)} NDCG20={result['metrics']['overall']['user_macro']['ndcg@20']:.6f}",flush=True)
        return result
    except Exception as error:
        attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        del model
        gc.collect()


def child(output,cohort_id):
    config,manifest=validate_inputs()
    lock=read(LOCK)
    if lock['output']!=str(output.parent) or lock['manifest_sha256']!=sha256(MANIFEST):
        raise ValueError('Child is outside one-time final-test reservation')
    cohort=next(c for c in config['cohorts'] if c['id']==cohort_id)
    if output.name != cohort_id: raise ValueError('Child path must match its frozen cohort ID')
    if output.exists(): raise FileExistsError('Final test child needs fresh directory')
    claim=output.parent/f'{cohort_id}.test_opened.json'
    with claim.open('x',encoding='utf-8') as stream:
        json.dump({'cohort':cohort_id,'output':str(output),'manifest_sha256':sha256(MANIFEST),
                   'created_at_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
    output.mkdir(parents=True)
    for name in ['baseline','models','predictions','results']: (output/name).mkdir()
    monitor=ResourceMonitor(config['resource_limits']);monitor.start()
    tick=time.perf_counter()
    status={'run_id':output.name,'status':'running','config':config,'cohort':cohort,
            'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),'manifest_sha256':sha256(MANIFEST),
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'attempts':[],'final_models':[],
            'final_test_evaluated':False,'test_period_opened':True,'source_and_input_sha256':manifest['files']}
    write_json(output/'run_status.json',status)
    try:
        monitor.check()
        members=read(TASK/cohort['membership_metadata'])['selected_users']
        events=load_published_history(TASK/cohort['history'],monitor)
        if set(map(int,np.unique(events['uid'])))!=set(members): raise ValueError('History membership drift')
        queries,info=select_test_queries(events,cohort['query_seed'],config['test_start_raw'],config['test_queries_per_user'])
        if not queries: raise ValueError('Frozen cohort has no eligible test queries')
        status['query_selection']=dict(info,selected_users=64,users_without_test_queries=64-len({q['user_id'] for q in queries}))
        artists=load_mapping(TASK/cohort['mapping_dir']/'artist_mapping.parquet','artist_id')
        albums=load_mapping(TASK/cohort['mapping_dir']/'album_mapping.parquet','album_id')
        status['candidates']=seal_candidates(events,queries,artists,albums,config,output,monitor)
        selected=read(TASK/cohort['development_run']/'selection_frozen.json')
        specs=generate_specs(config['windows_raw'],config['short_window_raw'])
        write_json(output/'specs.json',specs)
        masks={label:selected['final_masks'][label] for label in config['model_labels']}
        write_json(output/'masks_frozen.json',{'final_masks':masks,'development_selection_sha256':sha256(TASK/cohort['development_run']/'selection_frozen.json'),
                                            'test_scores_started':False})
        masks_sha=sha256(output/'masks_frozen.json');status['masks_frozen_sha256']=masks_sha
        status['baseline_build']=feature_build(FeatureEngine,events,queries,artists,albums,dict(config,feature_count=92),output/'baseline',monitor)
        def automatic_engine(a,b,w,h): return AutoFeatureEngine(a,b,w,h,specs=specs)
        status['build']=feature_build(automatic_engine,events,queries,artists,albums,config,output,monitor)
        del events,artists,albums
        gc.collect()
        matrix=np.load(output/'features.npy',mmap_mode='r')
        baseline=np.load(output/'baseline/features.npy',mmap_mode='r')
        schema=read(output/'baseline/feature_schema.json')
        status['a1_manual_equivalence']=assert_manual_equivalence(matrix,baseline,specs,schema,output/'a1_equivalence.json')
        if sha256(output/'feature_inputs.jsonl')!=sha256(output/'baseline/feature_inputs.jsonl'):
            raise ValueError('Test candidate input order mismatch')
        del baseline;gc.collect()
        dev_queries=[json.loads(line) for line in (TASK/cohort['development_queries']).read_text(encoding='utf-8').splitlines()]
        if any(q['timestamp']>=config['test_start_raw'] for q in dev_queries): raise ValueError('Test labels in development queries')
        dev_matrix=np.load(TASK/cohort['development_run']/'features.npy',mmap_mode='r')
        if dev_matrix.shape!=(sum(len(q['candidate_items']) for q in dev_queries),206): raise ValueError('Development shape drift')
        status['final_test_evaluated']=True
        popularity={'label':'Popularity','stage':'final_test','evaluation_period':'test','columns':[],'feature_count':0,
                    **save_predictions(output,'Popularity',queries,[np.asarray(q['popularity_scores']) for q in queries]),
                    'metrics':evaluate_ranking(queries,[q['popularity_scores'] for q in queries],ks=tuple(config['metrics_k']))}
        write_json(output/'results/Popularity.json',popularity);status['final_models'].append(popularity)
        cache={}
        for label,columns in masks.items():
            if sha256(output/'masks_frozen.json')!=masks_sha: raise ValueError('Final masks changed after test opened')
            signature=tuple(columns)
            if signature in cache:
                result=dict(cache[signature],label=label,reused_from=cache[signature]['label'])
                write_json(output/'results'/f'{label}.json',result)
            else:
                result=final_fit(label,columns,dev_matrix,dev_queries,matrix,queries,config,output,monitor,status)
                cache[signature]=result
            status['final_models'].append(result)
            write_json(output/'run_status.json',status)
        status.update(status='complete',fits_actual=len(status['attempts']))
        monitor.check()
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['elapsed_seconds']=round(time.perf_counter()-tick,6);status['resources']=monitor.finish()
        if status['resources']['limit_exceeded']: status.update(status='failed',error=status['resources']['limit_exceeded'])
        status['output_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ['run_status.json','metrics.json']}
        write_json(output/'run_status.json',status);write_json(output/'metrics.json',status)
    if status['status']!='complete': raise MemoryError(status['error'])


def parent(output):
    config,manifest=validate_inputs()
    if LOCK.exists(): raise FileExistsError('P006 test has already been opened; no retuning/rerun permitted')
    if output.exists(): raise FileExistsError('P006 parent needs fresh output')
    output.mkdir(parents=True)
    reservation={'output':str(output),'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),
                 'manifest_sha256':sha256(MANIFEST),'created_at_utc':datetime.now(timezone.utc).isoformat(),
                 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()}
    with LOCK.open('x',encoding='utf-8') as stream: json.dump(reservation,stream,indent=2)
    summary={'status':'running','cohorts':[],'reservation':reservation,'final_test_evaluated':False,
             'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),'manifest_sha256':sha256(MANIFEST)}
    write_json(output/'summary.json',summary)
    for cohort in config['cohorts']:
        name=cohort['id']
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(output/name),'--child',name],stdout=log,stderr=subprocess.STDOUT)
        state=read(output/name/'metrics.json') if (output/name/'metrics.json').exists() else {}
        complete=process.returncode==0 and state.get('status')=='complete'
        summary['cohorts'].append({'id':name,'run_dir':name,'status':'complete' if complete else 'failed','exit_code':process.returncode})
        summary['final_test_evaluated'] |= state.get('final_test_evaluated',False)
        write_json(output/'summary.json',summary)
        print(f"P006 {name}: {summary['cohorts'][-1]['status']}",flush=True)
    summary['status']='complete' if all(c['status']=='complete' for c in summary['cohorts']) else 'complete_with_failures'
    write_json(output/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=TASK/'artifacts/p006_final_test')
    parser.add_argument('--child')
    args=parser.parse_args()
    if args.child: child(args.output.resolve(),args.child)
    else: parent(args.output.resolve())


if __name__=='__main__': main()
