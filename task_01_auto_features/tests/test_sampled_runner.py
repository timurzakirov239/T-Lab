import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SRC=Path(__file__).resolve().parents[1]/'src'
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
import run_sampled_training as runner


class SampledVectorGrammarTests(unittest.TestCase):
    def test_66_column_adapter_preserves_prefix_ties_and_chunk_reuse(self):
        events={'uid':np.array([1,2,1,2]),'item_id':np.array([100,200,100,300]),
                'timestamp':np.array([1,2,2,4]),
                'event_type':np.array(['listen','listen','like','listen'])}
        specs=runner.generate_specs()[:66]
        engine=runner.AutoFeatureEngine({100:(7,),200:(7,),300:(8,)},
            {100:(11,12,13),200:(12,),300:(13,)},670390,670390,specs=specs)
        for i in range(3):
            engine.observe({'uid':int(events['uid'][i]),'item_id':int(events['item_id'][i]),
                            'timestamp':int(events['timestamp'][i]),'event_type':str(events['event_type'][i])})
        vector=runner.FullCatalogFeatures(engine,events)
        for uid in (1,2):
            items=[100,200,300,999]
            expected=engine.transform(uid,items,3)
            actual=np.concatenate([vector.transform(uid,items[:2],3),vector.transform(uid,items[2:],3)])
            self.assertEqual(expected.shape,(4,66))
            self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertNotIn(300,engine._items)


class AllFitsFreezeTests(unittest.TestCase):
    def fixture(self,root):
        config={'cohorts':[{'id':f'c{i}'} for i in range(3)]}
        for c in config['cohorts']:
            folder=root/c['id']/'fit';folder.mkdir(parents=True)
            (folder/'training_frozen.json').write_text('{}',encoding='utf-8')
            models=[{'model_origin':'p007'} for _ in range(2)]+[{'model_origin':'p008'} for _ in range(6)]
            saved={'status':'complete','models':models,
                   'training_frozen_sha256':hashlib.sha256(b'{}').hexdigest()}
            (folder/'run_status.json').write_text(json.dumps(saved),encoding='utf-8')
        (root/'config').write_text('{}',encoding='utf-8')
        (root/'protocol').write_text('protocol',encoding='utf-8')
        return config

    def context(self,root):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(runner,'CONFIG',root/'config'))
        stack.enter_context(patch.object(runner,'PROTOCOL',root/'protocol'))
        stack.enter_context(patch.object(runner,'sources',return_value={}))
        def require(folder):
            saved=runner.read(folder/'run_status.json')
            if saved['status']!='complete':raise ValueError('Incomplete fits')
            return saved
        stack.enter_context(patch.object(runner,'require_complete',side_effect=require))
        return stack

    def test_last_cohort_must_finish_before_any_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            path=root/'c2/fit/run_status.json';saved=runner.read(path);saved['status']='running'
            path.write_text(json.dumps(saved),encoding='utf-8')
            with self.context(root),self.assertRaises(ValueError):runner.require_all_fits(config,root)
            self.assertFalse((root/'all_fits_frozen.json').exists())

    def test_missing_new_fit_cannot_be_counted_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            path=root/'c2/fit/run_status.json';saved=runner.read(path);saved['models'].pop()
            path.write_text(json.dumps(saved),encoding='utf-8')
            with self.context(root),self.assertRaises(ValueError):runner.require_all_fits(config,root)

    def test_freeze_is_stable_and_rejects_model_metadata_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.fixture(root)
            with self.context(root):
                before=runner.require_all_fits(config,root)
                self.assertEqual(before,runner.require_all_fits(config,root))
                path=root/'c0/fit/run_status.json';saved=runner.read(path)
                saved['models'][0]['model_sha256']='changed'
                path.write_text(json.dumps(saved),encoding='utf-8')
                with self.assertRaises(ValueError):runner.require_all_fits(config,root)


if __name__=='__main__':unittest.main()
