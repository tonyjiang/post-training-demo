"""Scorer checks previously embedded in the SFT evaluation command."""
import importlib.util
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from tempfile import TemporaryDirectory

from evaluate_and_compare import (compare_saved_results, load_checkers, load_saved_results,
                                      score_instructions, score_math, summarize)


class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = {'arguments': {}, 'data_manifest': {
            'cases_sha256': 'cases', 'config': {'max_new_tokens': 1024},
            'selection': {'gsm8k': {'selected_ids': [1]}}},
            'adapter_sha256': 'adapter', 'base_manifest': {'revision': 'base'},
            'chat_template_sha256': 'template', 'versions': {}, 'metrics': {}}
        self.row = {'model': 'sft', 'task': 'gsm8k', 'id': 1, 'prompt': 'Question',
                    'correct': True, 'hit_token_limit': False, 'closed_think_tag': True,
                    'generated_tokens': 10}
        for label in ('sft', 'dpo'):
            folder = self.root / label
            folder.mkdir()
            run = deepcopy(self.run)
            run['arguments']['adapter_name'] = label
            row = dict(self.row, model=label)
            (folder / 'run-config.json').write_text(json.dumps(run))
            (folder / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            (folder / 'results.json').write_text(json.dumps({
                'scores': summarize([row]), 'data_sha256': 'cases', 'adapter_sha256': 'adapter'}))

    def test_reuses_only_selected_models_and_preserves_inputs(self):
        baseline = self.root / 'sft' / 'predictions.jsonl'
        baseline.write_text(baseline.read_text() + json.dumps(dict(self.row, model='base')) + '\n')
        original = baseline.read_bytes()
        with redirect_stdout(StringIO()):
            report = compare_saved_results(self.root / 'sft', 'sft', self.root / 'dpo',
                                           'dpo', self.root / 'comparison')
        self.assertEqual(set(report['scores']), {'sft', 'dpo'})
        self.assertFalse(report['generation_performed'])
        self.assertEqual(baseline.read_bytes(), original)
        with self.assertRaises(FileExistsError), redirect_stdout(StringIO()):
            compare_saved_results(self.root / 'sft', 'sft', self.root / 'dpo',
                                  'dpo', self.root / 'comparison')

    def test_rejects_different_generation_budget(self):
        expected = deepcopy(self.run)
        expected['data_manifest']['config']['max_new_tokens'] = 2048
        with self.assertRaisesRegex(ValueError, 'incompatible data_manifest'):
            load_saved_results(self.root / 'sft', 'sft', expected)

    def test_rejects_cached_1k_results_for_2k_evaluation(self):
        expected = deepcopy(self.run)
        expected['evaluation_config'] = {'max_new_tokens': 2048}
        with self.assertRaisesRegex(ValueError, 'incompatible evaluation_config'):
            load_saved_results(self.root / 'sft', 'sft', expected)

    def test_accepts_matching_effective_config_with_original_data_manifest(self):
        expected = deepcopy(self.run)
        expected['evaluation_config'] = {'max_new_tokens': 2048}
        path = self.root / 'sft' / 'run-config.json'
        run = json.loads(path.read_text())
        run['evaluation_config'] = expected['evaluation_config']
        path.write_text(json.dumps(run))
        _, rows, _ = load_saved_results(self.root / 'sft', 'sft', expected)
        self.assertEqual(len(rows), 1)

    def test_rejects_duplicate_or_missing_predictions(self):
        predictions = self.root / 'sft' / 'predictions.jsonl'
        for rows in ([], [self.row, self.row]):
            with self.subTest(rows=len(rows)):
                predictions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaisesRegex(ValueError, 'incomplete or duplicate'):
                    load_saved_results(self.root / 'sft', 'sft')

    def test_rejects_changed_prompts(self):
        (self.root / 'dpo' / 'predictions.jsonl').write_text(
            json.dumps(dict(self.row, model='dpo', prompt='Different question')) + '\n')
        with self.assertRaisesRegex(ValueError, 'prompts differ'):
            compare_saved_results(self.root / 'sft', 'sft', self.root / 'dpo',
                                  'dpo', self.root / 'comparison')


class MathScorerTests(unittest.TestCase):
    def test_boxed_number(self):
        self.assertTrue(score_math({'answer': 'calculation #### 1200'},
                                   r'Therefore \boxed{1,200}')['correct'])

    def test_rejects_wrong_unboxed_and_nonfinite_answers(self):
        for answer in (r'1200 then \boxed{12}', 'The reasoning mentions 1200', r'\boxed{NaN}'):
            with self.subTest(answer=answer):
                self.assertFalse(score_math({'answer': 'calculation #### 1200'}, answer)['correct'])


class InstructionScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dependencies = ('langdetect', 'nltk', 'spacy', 'emoji', 'syllapy', 'immutabledict')
        if any(importlib.util.find_spec(name) is None for name in dependencies):
            raise unittest.SkipTest('Requires the eval dependency group')
        if not Path('third_party/olmes').is_dir():
            raise unittest.SkipTest('Requires the pinned third_party/olmes checkout (see README)')
        import nltk
        nltk.data.path.insert(0, str(Path('data/eval-sources/nltk').resolve()))
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # OLMES otherwise downloads this at import time. Keep discovery offline.
            raise unittest.SkipTest('Requires cached NLTK punkt data (see README evaluation setup)')
        config = json.loads(Path('configs/eval-small.json').read_text())
        registry, extract = load_checkers(config)
        cls.registry = registry
        cls.extract = staticmethod(extract)

    def test_reasoning_output_extraction(self):
        result = self.extract({'continuation': 'reasoning</think>\n<answer>done</answer>'})
        self.assertEqual(result['continuation'], 'done')

    def test_strict_instruction_checks(self):
        doc = {'prompt': 'Use lowercase without commas.',
               'instruction_id_list': ['change_case:english_lowercase', 'punctuation:no_comma'],
               'kwargs': [{}, {}]}
        self.assertTrue(score_instructions(doc, 'hello world', self.registry)['correct'])
        self.assertEqual(score_instructions(doc, 'Hello, world', self.registry)['instruction_checks'],
                         [False, False])
        self.assertFalse(score_instructions(doc, '', self.registry)['correct'])


if __name__ == '__main__':
    unittest.main()
