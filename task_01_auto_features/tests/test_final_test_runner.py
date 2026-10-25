import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from task_01_auto_features.src import run_final_test as runner
from task_01_auto_features.src.final_test_analysis import user_bootstrap, contribution


class NoLimit:
    def check(self): pass


class FinalTestRunnerTests(unittest.TestCase):
    def test_candidates_ignore_labels_and_same_time_events_for_all_users(self):
        events={'uid':np.array([1,2,1]),'item_id':np.array([11,22,33]),'timestamp':np.array([1,5,5]),
                'event_code':np.zeros(3,dtype=np.uint8),'event_names':['listen']}
        qs=[{'query_id':0,'user_id':1,'timestamp':5,'target_item':33},
            {'query_id':1,'user_id':2,'timestamp':5,'target_item':22},
            {'query_id':2,'user_id':2,'timestamp':6,'target_item':99}]
        c={'max_candidates':100,'candidate_quotas':{'popular':25,'repeat':25,'artist':35,'album':15}}
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a',Path(temp)/'b';a.mkdir();b.mkdir()
            other=copy.deepcopy(qs);other[0]['target_item']=999
            runner.seal_candidates(events,qs,{}, {},c,a,NoLimit())
            runner.seal_candidates(events,other,{}, {},c,b,NoLimit())
            self.assertEqual(qs[0]['candidate_items'],[11]);self.assertEqual(qs[1]['candidate_items'],[11])
            self.assertEqual(set(qs[2]['candidate_items']),{11,22,33})
            self.assertEqual([q['candidate_items'] for q in qs],[q['candidate_items'] for q in other])

    def test_paired_bootstrap_is_deterministic_and_resamples_users(self):
        constant=user_bootstrap([.25,.25,.25])
        self.assertEqual(constant['mean'],.25);self.assertEqual(constant['ci_lower'],.25)
        self.assertEqual(constant['ci_upper'],.25);self.assertEqual(constant['users'],3)
        self.assertEqual(user_bootstrap([-.1,.2,.4]),user_bootstrap([-.1,.2,.4]))
        with self.assertRaises(ValueError): user_bootstrap([])

    def test_score_ties_use_item_id_and_missing_target_is_zero(self):
        q={'candidate_items':[4,2,3],'target_item':2}
        self.assertEqual(contribution(q,[1.,1.,1.])['rank'],1)
        q['target_item']=99
        self.assertEqual(contribution(q,[1.,1.,1.])['ndcg20'],0)

    def test_parent_preserves_failed_child_then_continues_without_resampling(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);manifest=p/'manifest.json';manifest.write_text('{}')
            output=p/'run';lock=p/'lock.json'
            config={'cohorts':[{'id':'a'},{'id':'b'},{'id':'c'}]}
            def fake_run(command,**kwargs):
                folder=Path(command[command.index('--output')+1]);folder.mkdir()
                failed=folder.name=='a'
                (folder/'metrics.json').write_text(json.dumps({'status':'failed' if failed else 'complete','final_test_evaluated':not failed}))
                return type('Completed',(),{'returncode':1 if failed else 0})()
            with patch.object(runner,'validate_inputs',return_value=(config,{})),patch.object(runner,'LOCK',lock),patch.object(runner,'MANIFEST',manifest),patch.object(runner.subprocess,'check_output',return_value='frozen'),patch.object(runner.subprocess,'run',side_effect=fake_run):
                runner.parent(output)
                saved=json.loads((output/'summary.json').read_text())
                self.assertEqual(saved['status'],'complete_with_failures')
                self.assertEqual([c['id'] for c in saved['cohorts']],['a','b','c'])
                with self.assertRaises(FileExistsError): runner.parent(p/'rerun')

    def test_final_training_includes_former_validation_and_rejects_test(self):
        queries=[{'split':'train','timestamp':1,'candidate_items':[1,2],'target_item':2},
                 {'split':'validation','timestamp':9,'candidate_items':[3,4],'target_item':3},
                 {'split':'validation','timestamp':9,'candidate_items':[3,4],'target_item':99}]
        rows,labels,counts=runner.development_training_rows(queries,10)
        self.assertEqual(rows.tolist(),[0,1,2,3]);self.assertEqual(labels.tolist(),[0,1,1,0])
        self.assertEqual(counts['queries_requested'],3);self.assertEqual(counts['queries_with_target'],2)
        self.assertEqual(queries[1]['split'],'validation')
        queries[1]['timestamp']=10
        with self.assertRaises(ValueError): runner.development_training_rows(queries,10)


if __name__=='__main__': unittest.main()
