"""Freeze verified development inputs and runtime bytes before P006 opens test."""
from __future__ import annotations
import json
from pathlib import Path

if __package__:
    from .run_pilot import sha256, write_json
else:
    from run_pilot import sha256, write_json

TASK=Path(__file__).resolve().parents[1]


def main():
    output=TASK/'configs/p006_freeze_manifest.json'
    if output.exists(): raise FileExistsError('P006 freeze manifest is immutable')
    config=json.loads((TASK/'configs/p006_final_test.json').read_text(encoding='utf-8'))
    files={}
    def pin(path,expected=None):
        path=Path(path);actual=sha256(path)
        if expected is not None and actual!=expected: raise ValueError(f'Frozen development evidence drift: {path}')
        files[path.relative_to(TASK).as_posix()]=actual
    sources=['run_final_test.py','final_test_data.py','final_test_analysis.py','run_robustness.py',
             'run_auto_features.py','auto_features.py','auto_search.py','pilot_features.py','pilot_evaluation.py',
             'run_feature_benchmark.py','run_retrieval.py','run_pilot.py','pilot_storage.py',
             'run_feature_scale.py','scale_data.py','robustness_data.py','pilot_data.py','verify_final_test.py']
    for name in sources: pin(TASK/'src'/name)
    for name in ['configs/p006_final_test.json','configs/P006_PROTOCOL.md','data/download_manifest.json',
                 'data/raw/yambda/SOURCE_README.md','report/pilot_001_snapshot/metrics.json',
                 'report/p005_snapshot/membership/memberships.json','report/p005_snapshot/verification_independent.json']:
        pin(TASK/name)
    records=[]
    for cohort in config['cohorts']:
        run=TASK/cohort['development_run'];snapshot=TASK/cohort['snapshot']
        saved=json.loads((snapshot/'metrics.json').read_text(encoding='utf-8'))
        if saved['status']!='complete' or saved['final_test_evaluated']: raise ValueError('Incomplete development input')
        pin(run/'metrics.json',sha256(snapshot/'metrics.json'))
        pin(snapshot/'metrics.json')
        for relative,expected in saved['source_code_sha256'].items(): pin(TASK/'src'/relative,expected)
        pin(run/'features.npy',saved['build']['matrix_file_sha256'])
        pin(run/'feature_inputs.jsonl',saved['build']['feature_inputs_sha256'])
        pin(run/'feature_schema.json',saved['build']['feature_schema_sha256'])
        pin(run/'selection_frozen.json',saved['selection_frozen_sha256'])
        pin(snapshot/'selection_frozen.json',saved['selection_frozen_sha256'])
        pin(run/'specs.json',saved['specs_sha256'])
        for relative,digest in saved['input_sha256'].items(): pin(TASK/relative,digest)
        for relative in [cohort['development_queries'],cohort['history'],cohort['membership_metadata'],
                         f"{cohort['mapping_dir']}/artist_mapping.parquet",f"{cohort['mapping_dir']}/album_mapping.parquet"]:
            pin(TASK/relative)
        # Query/candidate order is independently reconstructed from saved blocks.
        queries=[json.loads(line) for line in (TASK/cohort['development_queries']).read_text(encoding='utf-8').splitlines()]
        import hashlib
        structural=hashlib.sha256();offset=0
        for q in queries:
            if q['timestamp']>=config['test_start_raw']: raise ValueError('Development query crossed test boundary')
            row={key:q[key] for key in ['query_id','user_id','timestamp']}
            row.update(offset=offset,size=len(q['candidate_items']),candidate_items=q['candidate_items'])
            structural.update((json.dumps(row,separators=(',',':'))+'\n').encode('utf-8'))
            offset+=len(q['candidate_items'])
        if structural.hexdigest()!=saved['build']['feature_inputs_sha256']: raise ValueError('Development row order drift')
        records.append({'id':cohort['id'],'development_rows':offset,'development_queries':len(queries),
                        'selection_sha256':saved['selection_frozen_sha256']})
    write_json(output,{'experiment':'P006','status':'frozen_before_test','files':files,'cohorts':records,
                       'method':'Auto_A2 vs M_UIDC with fixed cohort-specific development masks; no test tuning'})
    print(f'P006 manifest: {len(files)} frozen source/input files; no reserved payload decoded')


if __name__=='__main__': main()
