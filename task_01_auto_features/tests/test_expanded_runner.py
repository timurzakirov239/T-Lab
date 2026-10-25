import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from run_expanded_search import quick_ndcg, ranking_rows
from pilot_evaluation import evaluate_ranking


class ExpandedRunnerTests(unittest.TestCase):
    def test_inner_fast_metric_and_saved_ranks_match_independent_evaluator(self):
        queries=[]
        for qid, (uid,target,items) in enumerate([(1,7,[7,2,9]),(1,4,[3,5]),(2,8,[8,2,4])]):
            queries.append({'query_id':qid,'user_id':uid,'timestamp':10+qid,'target_item':target,
                'candidate_items':items,'split':'validation','target_is_repeat':qid!=1,
                'target_known_before':True})
        scores=np.asarray([.5,.5,.2, .9,.1, .1,.2,.3])
        offsets=np.asarray([0,3,5,8])
        expected=evaluate_ranking(queries,[scores[offsets[i]:offsets[i+1]] for i in range(3)])
        self.assertAlmostEqual(quick_ndcg(queries,scores),expected['overall']['user_macro']['ndcg@20'],places=14)
        rows=ranking_rows('synthetic',queries,offsets,scores)
        self.assertEqual([r['target_rank'] for r in rows],[2,None,3])
        self.assertEqual(rows[0]['top_items'][0]['item_id'],2)
        self.assertIsNone(rows[1]['target_score'])

    def test_inner_score_rejects_wrong_flattened_length(self):
        with self.assertRaises(ValueError):
            quick_ndcg([],np.ones(1))


if __name__=='__main__':unittest.main()
