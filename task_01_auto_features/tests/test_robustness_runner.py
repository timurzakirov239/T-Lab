import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from task_01_auto_features.src import run_robustness as runner
from task_01_auto_features.src.auto_features import generate_specs
from task_01_auto_features.src.pilot_features import FeatureEngine


class NoLimit:
    def check(self): pass


class RobustnessRunnerTests(unittest.TestCase):
    def test_frozen_config_rejects_model_or_query_seed_drift(self):
        config = runner.validate_config()
        self.assertEqual(config['seed'], 20261025)
        self.assertEqual(config['cohort_seeds'], [20261026,20261027])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            config['seed']=20261026
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'drift'): runner.validate_config(path)

    def test_variable_row_a1_gate_checks_every_byte_and_rejects_signed_zero(self):
        specs=generate_specs([201117,670390,2011171],670390)
        engine=FeatureEngine({}, {},670390,670390)
        schema={'feature_names':engine.feature_names,'feature_groups':engine.feature_groups}
        baseline=np.zeros((3,92),dtype=np.float32)
        matrix=np.zeros((3,206),dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json'
            self.assertEqual(runner.assert_manual_equivalence(matrix,baseline,specs,schema,path)['cells_compared'],198)
            matrix[2,65]=-0.0
            with self.assertRaises(AssertionError): runner.assert_manual_equivalence(matrix,baseline,specs,schema,path)
            with self.assertRaises(ValueError): runner.assert_manual_equivalence(matrix,baseline[:2],specs,schema,path)

    def test_candidate_target_perturbation_and_equal_time_are_causal(self):
        events={'uid':np.array([1,1,2]),'item_id':np.array([11,22,33]),'timestamp':np.array([1,5,5]),
                'event_code':np.zeros(3,dtype=int),'event_names':['listen']}
        config={'max_candidates':100,'candidate_quotas':{'popular':25,'repeat':25,'artist':35,'album':15}}
        queries=[{'query_id':0,'user_id':1,'timestamp':5,'split':'train','target_item':22},
                 {'query_id':1,'user_id':1,'timestamp':6,'split':'validation','target_item':99}]
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            a.mkdir(); b.mkdir()
            original=copy.deepcopy(queries)
            runner.seal_candidates(events,queries,{}, {},config,a,NoLimit())
            original[0]['target_item']=999
            runner.seal_candidates(events,original,{}, {},config,b,NoLimit())
            self.assertEqual(queries[0]['candidate_items'],[11])
            self.assertEqual(set(queries[1]['candidate_items']),{11,22,33})
            self.assertEqual([q['candidate_items'] for q in queries],[q['candidate_items'] for q in original])

    def test_membership_overlap_and_count_rejected(self):
        config=runner.validate_config()
        from task_01_auto_features.src.robustness_data import _hash_uid_list
        cohorts=[{'seed':seed,'selected_users':list(range(start,start+64)),
                  'selected_users_sha256':_hash_uid_list(list(range(start,start+64)))}
                 for seed,start in [(20261026,1),(20261027,65)]]
        m={'status':'frozen','seeds':config['cohort_seeds'],'users_per_cohort':64,'excluded_users':[0],'cohorts':cohorts}
        runner.validate_memberships(m,[0],config)
        m['cohorts'][1]['selected_users'][0]=1
        with self.assertRaises(ValueError): runner.validate_memberships(m,[0],config)


if __name__=='__main__': unittest.main()
