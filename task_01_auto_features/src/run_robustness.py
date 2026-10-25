"""P005: serial disjoint-cohort replication of frozen P004 feature selection."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
if __package__:
    from . import run_auto_features as frozen
    from .auto_features import AutoFeatureEngine, generate_specs
    from .auto_search import temporal_split, offsets_sizes, prune_columns, select_trial_columns, pick_best
    from .pilot_features import FeatureEngine
    from .run_feature_benchmark import feature_build
    from .run_feature_scale import select_fixed_queries
    from .run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from .run_retrieval import load_development_events
    from .robustness_data import prepare_memberships, prepare_cohort
    from .scale_data import EVENT_SOURCE, ARTIST_SOURCE, ALBUM_SOURCE
    from .pilot_data import _manifest_hashes
else:
    import run_auto_features as frozen
    from auto_features import AutoFeatureEngine, generate_specs
    from auto_search import temporal_split, offsets_sizes, prune_columns, select_trial_columns, pick_best
    from pilot_features import FeatureEngine
    from run_feature_benchmark import feature_build
    from run_feature_scale import select_fixed_queries
    from run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from run_retrieval import load_development_events
    from robustness_data import prepare_memberships, prepare_cohort
    from scale_data import EVENT_SOURCE, ARTIST_SOURCE, ALBUM_SOURCE
    from pilot_data import _manifest_hashes
TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK / "configs/p005_robustness.json"
PROTOCOL = TASK / "configs/P005_PROTOCOL.md"
CONFIG_SHA = "835928602418aca0a4b11683ea4f080556a874908d02115d5c0f256b69e7bbf6"
PROTOCOL_SHA = "908b10b2a12e150a1a01e25a5bac9e14b377c074f6d1e6a1f7b20a8ffdc686e0"
EXCLUSION_SHA = "52a44696948ea7ebcf285c8a80b56429a19a6a4f61eb30e2132be63b76103699"
P004_METRICS_SHA = "dfd2abf3c7ac5bfc35fff49566162847dce43251b3efafdc48e851654949b2a3"
candidate_rows = frozen.candidate_rows
union_columns = frozen.union_columns
final_masks = frozen.final_masks
fit_and_predict = frozen.fit_and_predict


def validate_config(path=CONFIG):
    if sha256(path) != CONFIG_SHA or sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P005 frozen config/protocol drift")
    config = json.loads(path.read_text(encoding="utf-8"))
    if sha256(TASK / config['exclusion_metadata']) != EXCLUSION_SHA:
        raise ValueError('Frozen P001 exclusion metadata drift')
    reference_path = TASK / config['frozen_p004_metrics']
    if sha256(reference_path) != P004_METRICS_SHA: raise ValueError('P004 checkpoint authority drift')
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    for name, expected in reference["source_code_sha256"].items():
        if sha256(TASK / "src" / name) != expected:
            raise ValueError(f"Frozen P004 source drift: {name}")
    return config


def validate_memberships(memberships, excluded, config):
    if (memberships.get('status') != 'frozen' or
            memberships.get('seeds') != config['cohort_seeds'] or
            memberships.get('users_per_cohort') != config['users_per_cohort'] or
            memberships.get('excluded_users') != sorted(excluded) or
            len(memberships.get('cohorts', [])) != 2):
        raise ValueError('Membership protocol mismatch')
    seen = set(excluded)
    for cohort, seed in zip(memberships['cohorts'], config['cohort_seeds']):
        selected = cohort['selected_users']
        if (cohort['seed'] != seed or len(selected) != 64 or len(set(selected)) != 64 or
                seen.intersection(selected)):
            raise ValueError('Membership overlap/count/seed mismatch')
        digest = hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode('ascii')).hexdigest()
        if digest != cohort['selected_users_sha256']: raise ValueError('Membership UID hash mismatch')
        seen.update(selected)


def assert_manual_equivalence(matrix, baseline, specs, schema, output):
    names = [s["feature_id"] for s in specs if s["family"] == "A1"]
    expected = [n for n in schema["feature_names"] if schema["feature_groups"][n] in "UIDC"]
    if names != expected or len(names) != 66 or matrix.shape[1] != 206 or baseline.shape != (len(matrix),92):
        raise ValueError("A1/manual schema or row mismatch")
    ac = [i for i,s in enumerate(specs) if s["family"] == "A1"]
    bc = [schema["feature_names"].index(n) for n in names]
    digest=hashlib.sha256()
    for start in range(0,len(matrix),2048):
        a,b=matrix[start:start+2048,ac],baseline[start:start+2048,bc]
        if a.dtype != np.float32 or b.dtype != np.float32 or a.tobytes()!=b.tobytes():
            raise AssertionError("A1 byte equivalence failed; training forbidden")
        digest.update(a.tobytes())
    result={"status":"pass","cells_compared":len(matrix)*66,"feature_ids":names,
            "identical_float32_bytes":True,"values_sha256":digest.hexdigest()}
    write_json(output,result)
    return result


def seal_candidates(events, queries, artists, albums, config, output, monitor):
    generator=CandidateGenerator(artists,albums,config)
    index=0
    started=time.perf_counter()
    with (output / "queries.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        for position,q in enumerate(queries):
            while index<len(events["timestamp"]) and int(events["timestamp"][index])<q["timestamp"]:
                generator.observe({"uid":int(events["uid"][index]),"item_id":int(events["item_id"][index]),
                                   "timestamp":int(events["timestamp"][index]),
                                   "event_type":events["event_names"][int(events["event_code"][index])]})
                index+=1
                if index%5000==0: monitor.check()
            candidates=generator.candidates(q["user_id"],q["timestamp"])
            if len(candidates)!=len(set(candidates)) or any(i not in generator.catalog for i in candidates):
                raise AssertionError("Invalid candidates")
            q["candidate_items"]=candidates
            target=q["target_item"]
            q.update(target_is_repeat=target in generator.last_listen.get(q["user_id"],{}),
                     target_known_before=target in generator.catalog)
            stream.write(json.dumps(q,separators=(",",":"))+"\n")
            if position%20==0: monitor.check()
    return {"strategy":"P001_original","quotas":config["candidate_quotas"],
            "queries_sha256":sha256(output/"queries.jsonl"),"candidate_rows":sum(len(q["candidate_items"]) for q in queries),
            "elapsed_seconds":round(time.perf_counter()-started,6)}


def child(output,config_path,memberships_path,seed):
    config=validate_config(config_path)
    membership_hash=sha256(memberships_path)
    memberships=json.loads(memberships_path.read_text(encoding="utf-8"))
    parent_status=json.loads((memberships_path.parent.parent/'summary.json').read_text(encoding='utf-8'))
    if membership_hash != parent_status['memberships_sha256']: raise ValueError('Frozen membership hash drift')
    excluded=json.loads((TASK/config['exclusion_metadata']).read_text(encoding='utf-8'))['selected_users']
    validate_memberships(memberships, excluded, config)
    cohort=next(c for c in memberships["cohorts"] if c["seed"]==seed)
    config=dict(config,query_sampling_seed=seed)
    if output.exists(): raise FileExistsError("P005 needs fresh child directory")
    output.mkdir(parents=True)
    for folder in ("baseline","models","results","predictions"): (output/folder).mkdir()
    monitor=ResourceMonitor(config["resource_limits"])
    monitor.start()
    started=time.perf_counter()
    sources=list(json.loads((TASK/config["frozen_p004_metrics"]).read_text())["source_code_sha256"])
    sources+=['run_robustness.py','robustness_data.py','run_feature_scale.py','scale_data.py','pilot_data.py','pilot_storage.py']
    status={"run_id":output.name,"status":"running","config":config,"query_sampling_seed":seed,
            "config_path":str(config_path.relative_to(TASK)),"protocol_path":"configs/P005_PROTOCOL.md",
            "config_sha256":sha256(config_path),"protocol_sha256":sha256(PROTOCOL),
            "source_code_sha256":{n:sha256(TASK/'src'/n) for n in sources},
            "git_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"attempts":[],"build_progress":{},
            "final_models":[],"final_test_evaluated":False,
            "source_data_sha256":parent_status['source_data_sha256'],
            "input_sha256":{str(memberships_path.relative_to(TASK)):membership_hash}}
    write_json(output/'run_status.json',status)
    try:
        monitor.check()
        tick=time.perf_counter()
        metadata=prepare_cohort(output/'data',cohort['selected_users'],seed,config['boundaries_raw']['validation_end'],monitor=monitor)
        status['data_preparation_seconds']=round(time.perf_counter()-tick,6)
        if metadata['development']['chronology']['violations']: raise ValueError('Chronology violations')
        for path in (output/'data').iterdir(): status['input_sha256'][str(path.relative_to(TASK))]=sha256(path)
        events=load_development_events(output/'data/events.parquet',config['boundaries_raw']['validation_end'])
        artists=load_mapping(output/'data/artist_mapping.parquet','artist_id')
        albums=load_mapping(output/'data/album_mapping.parquet','album_id')
        queries,query_info=select_fixed_queries(events,dict(config,seed=seed))
        status['query_selection']=query_info
        status['candidates']=seal_candidates(events,queries,artists,albums,config,output,monitor)
        offsets,sizes=offsets_sizes(queries)
        split=temporal_split(queries)
        write_json(output/'split.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in split.items()})
        status['query_counts']={k:len(v) for k,v in split.items() if k!='cutoff'}
        specs=generate_specs(config['windows_raw'],config['short_window_raw'])
        reference=json.loads((TASK/'report/p004_snapshot/specs.json').read_text(encoding='utf-8'))
        if specs!=reference: raise ValueError('P004 grammar drift')
        write_json(output/'specs.json',specs)
        status['specs_sha256']=sha256(output/'specs.json')
        status['baseline_build']=feature_build(FeatureEngine,events,queries,artists,albums,dict(config,feature_count=92),output/'baseline',monitor)
        def automatic_engine(a,b,w,h): return AutoFeatureEngine(a,b,w,h,specs=specs)
        status['build']=feature_build(automatic_engine,events,queries,artists,albums,config,output,monitor,progress=status['build_progress'])
        del events,artists,albums
        gc.collect()
        matrix=np.load(output/'features.npy',mmap_mode='r')
        baseline=np.load(output/'baseline/features.npy',mmap_mode='r')
        schema=json.loads((output/'baseline/feature_schema.json').read_text())
        status['a1_manual_equivalence']=assert_manual_equivalence(matrix,baseline,specs,schema,output/'a1_equivalence.json')
        if sha256(output/'feature_inputs.jsonl')!=sha256(output/'baseline/feature_inputs.jsonl'):
            raise ValueError('Candidate input order mismatch')
        del baseline
        gc.collect()
        fit_rows = candidate_rows(split["inner_fit"], offsets, sizes)
        winners = {}
        pruning = {}
        inner_results = []
        selection_started = time.perf_counter()
        for family in ("A1", "A2"):
            candidates = [i for i, spec in enumerate(specs) if family == "A2" or spec["family"] == "A1"]
            available, removed = prune_columns(matrix, fit_rows, candidates)
            pruning[family] = {"available": available, "removed": removed}
            write_json(output / "pruning.json", pruning)
            full = fit_and_predict(f"inner_{family}_full", available, split["inner_fit"], split["inner_selection"], matrix,
                                   queries, offsets, sizes, specs, config, output, monitor, status, "inner")
            results = [full]
            importances = {column: importance for column, importance in zip(available, full["importance_fit_only"])}
            proposals = select_trial_columns(available, specs, importances)
            if len(proposals) > config["maximum_inner_fits_per_family"]:
                raise ValueError("Family search budget exceeded")
            for name, columns in proposals.items():
                if columns == available:
                    continue
                results.append(fit_and_predict(f"inner_{family}_{name}", columns, split["inner_fit"], split["inner_selection"],
                                              matrix, queries, offsets, sizes, specs, config, output, monitor, status, "inner"))
            choices = [{"label": row["label"], "columns": row["columns"], "spec_ids": row["spec_ids"],
                        "inner_score": row["metrics"]["overall"]["user_macro"]["ndcg@20"]} for row in results]
            winners[family] = dict(pick_best(choices))
            inner_results.extend(results)
            write_json(output / "inner_results.json", inner_results)
        best = dict(pick_best(list(winners.values())))
        manual = [i for i, spec in enumerate(specs) if spec["family"] == "A1"]
        union, union_removed = union_columns(matrix, fit_rows, manual, best["columns"])
        masks = final_masks(specs, winners["A1"]["columns"], winners["A2"]["columns"], union)
        frozen = {"winners": winners, "chosen_auto_by_inner": best, "final_masks": masks,
                  "union_removed": union_removed, "specs_sha256": status["specs_sha256"],
                  "external_predictions_started": False}
        write_json(output / "selection_frozen.json", frozen)
        frozen_hash = sha256(output / "selection_frozen.json")
        status["selection_frozen_sha256"] = frozen_hash
        status["selection_seconds"] = round(time.perf_counter() - selection_started, 6)
        write_json(output / "run_status.json", status)
        train_positions = [i for i, q in enumerate(queries) if q["split"] == "train"]
        status["external_evaluation_started"] = True
        final_cache = {}
        for label, columns in masks.items():
            if sha256(output / "selection_frozen.json") != frozen_hash:
                raise ValueError("Frozen selection changed during external evaluation")
            if not columns:
                status["final_models"].append({"label": label, "status": "skipped_empty_group_ablation"})
                continue
            signature = tuple(columns)
            if signature in final_cache:
                result = dict(final_cache[signature], label=label, reused_from=final_cache[signature]["label"])
                write_json(output / "results" / f"{label}.json", result)
            else:
                result = fit_and_predict(label, columns, train_positions, split["external_validation"], matrix, queries,
                                         offsets, sizes, specs, config, output, monitor, status, "final")
                final_cache[signature] = result
            status["final_models"].append(result)
            write_json(output / "run_status.json", status)
        status["fits_actual"] = len(status["attempts"])
        status["status"] = "complete"
        status["output_sha256"] = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*")
                                   if path.is_file() and path.name not in {"run_status.json", "metrics.json"}}
        monitor.check()
    except Exception as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status['elapsed_seconds']=round(time.perf_counter()-started,6)
        status['resources']=monitor.finish()
        if status['resources']['limit_exceeded']: status.update(status='failed',error=status['resources']['limit_exceeded'])
        write_json(output/'run_status.json',status)
        write_json(output/'metrics.json',status)
    if status['status']!='complete': raise MemoryError(status['error'])


def parent(output,config_path):
    config=validate_config(config_path)
    if output.exists(): raise FileExistsError('P005 parent output exists')
    output.mkdir(parents=True)
    summary={'status':'running','config_path':str(config_path.relative_to(TASK)),
             'protocol_path':'configs/P005_PROTOCOL.md','cohorts':[], 'final_test_evaluated':False,
             'config_sha256':sha256(config_path),'protocol_sha256':sha256(PROTOCOL)}
    monitor=ResourceMonitor(config['resource_limits'])
    monitor.start()
    try:
        expected=_manifest_hashes()
        summary['source_data_sha256']={}
        for source,relative in ((EVENT_SOURCE,'yambda/flat/50m/multi_event.parquet'),
                                (ARTIST_SOURCE,'yambda/artist_item_mapping.parquet'),(ALBUM_SOURCE,'yambda/album_item_mapping.parquet')):
            digest=sha256(source)
            if digest!=expected[relative]: raise ValueError('Raw source hash drift')
            summary['source_data_sha256'][relative]=digest
            monitor.check()
        exclusion=TASK/config['exclusion_metadata']
        summary['exclusion_metadata_sha256']=sha256(exclusion)
        excluded=json.loads(exclusion.read_text(encoding='utf-8'))['selected_users']
        if len(excluded) != 64 or len(set(excluded)) != 64: raise ValueError('Original cohort count drift')
        membership=prepare_memberships(output/'membership',seeds=config['cohort_seeds'],users=config['users_per_cohort'],exclude_users=excluded,monitor=monitor)
        validate_memberships(membership, excluded, config)
        summary['memberships_sha256']=sha256(output/'membership/memberships.json')
    except Exception as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        summary['membership_resources']=monitor.finish()
        write_json(output/'summary.json',summary)
    if summary['membership_resources']['limit_exceeded']: raise MemoryError(summary['membership_resources']['limit_exceeded'])
    for cohort in membership['cohorts']:
        seed=cohort['seed']
        name=f'cohort_{seed}'
        if sha256(output/'membership/memberships.json')!=summary['memberships_sha256']: raise ValueError('Membership changed')
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),
                    '--output',str(output/name),'--child-seed',str(seed),'--memberships',str(output/'membership/memberships.json')],stdout=log,stderr=subprocess.STDOUT)
        metric=output/name/'metrics.json'
        state=json.loads(metric.read_text(encoding='utf-8')) if metric.exists() else {}
        ok=result.returncode==0 and state.get('status')=='complete'
        summary['cohorts'].append({'seed':seed,'run_dir':name,'status':'complete' if ok else 'failed','exit_code':result.returncode})
        write_json(output/'summary.json',summary)
        print(f'P005 {name}: {summary["cohorts"][-1]["status"]}',flush=True)
    summary['status']='complete' if all(c['status']=='complete' for c in summary['cohorts']) else 'complete_with_failures'
    write_json(output/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output',type=Path,default=TASK/'artifacts/p005_robustness')
    parser.add_argument('--child-seed',type=int)
    parser.add_argument('--memberships',type=Path)
    args=parser.parse_args()
    if args.child_seed is None: parent(args.output.resolve(),args.config.resolve())
    else: child(args.output.resolve(),args.config.resolve(),args.memberships.resolve(),args.child_seed)
if __name__=='__main__': main()
