"""Reward correctness and data-integrity checks for math RLVR."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dpo.data import sha256
from rlvr.data import load_splits
from rlvr.rewards import load_math_utils, verify_answer


class MathRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(x) is None for x in ('sympy','antlr4')):
            raise unittest.SkipTest('Requires rlvr dependencies')
        cls.math_utils=load_math_utils()

    def test_equivalent_final_answers(self):
        self.assertEqual(verify_answer(r'reasoning</think>Final: \boxed{\frac{1}{2}}','0.5',self.math_utils),1.0)
        self.assertEqual(verify_answer(r'reasoning</think>Final: \boxed{83}','83',self.math_utils),1.0)

    def test_wrong_missing_and_unfinished_answers(self):
        for text in (r'\boxed{83}',r'<think>\boxed{83}',r'reasoning</think>83',
                     r'reasoning</think>\boxed{82}',r'reasoning</think>\boxed{83'):
            with self.subTest(text=text):
                self.assertEqual(verify_answer(text,'83',self.math_utils),0.0)

    def test_reasoning_box_does_not_override_final_answer(self):
        self.assertEqual(verify_answer(r'\boxed{83}</think>\boxed{82}','83',self.math_utils),0.0)
        self.assertEqual(verify_answer(r'reasoning</think>\boxed{82} then \boxed{83}','83',self.math_utils),1.0)


class SplitIntegrityTests(unittest.TestCase):
    def test_overlap_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row={'id':'same','prompt_hash':'same','prompt':[{'role':'user','content':'Q'}],
                 'answer':'1','prompt_tokens':10}
            manifest={'source_manifest':json.loads(Path('configs/rlvr-data.json').read_text()),
                      'max_prompt_length':512,'splits':{}}
            for split in ['train','eval']:
                path=root/f'{split}.jsonl';path.write_text(json.dumps(row)+'\n')
                manifest['splits'][split]={'rows':1,'sha256':sha256(path)}
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_splits(root)
            (root/'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_splits(root)
