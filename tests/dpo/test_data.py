"""CPU-only checks for preference pairing, truncation, and split integrity."""
import json
from pathlib import Path
import tempfile
import unittest

from dpo.data import load_splits, preference, sha256, tokenize_pair


def row(chosen='yes', rejected='no', prompt='question', id='a'):
    prefix = [{'role': 'user', 'content': prompt}]
    return {'id': id, 'dataset_source': 'test',
            'chosen': prefix + [{'role': 'assistant', 'content': chosen}],
            'rejected': prefix + [{'role': 'assistant', 'content': rejected}]}


class Tokenizer:
    """Simple exact-prefix tokenizer; EOS is a distinct token."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True):
        prefix = [1] + [ord(c) for c in messages[0]['content']] + [2]
        ids = prefix if add_generation_prompt else prefix + [ord(c) for c in messages[-1]['content']] + [3]
        return {'input_ids': ids}


class PreferenceTests(unittest.TestCase):
    def test_shared_prompt_and_final_responses(self):
        record = preference(row())
        self.assertEqual(record['prompt'], [{'role': 'user', 'content': 'question'}])
        self.assertEqual(record['chosen'], [{'role': 'assistant', 'content': 'yes'}])
        bad = row()
        bad['rejected'][0] = {'role': 'user', 'content': 'different question'}
        with self.assertRaisesRegex(ValueError, 'same nonempty'):
            preference(bad)

    def test_invalid_or_identical_completions(self):
        for chosen, rejected in [('yes', 'yes'), ('', 'no'), ('yes', ' ')]:
            with self.assertRaises(ValueError):
                preference(row(chosen, rejected))

    def test_truncation_preserves_prompt_and_does_not_append_eos(self):
        record = preference(row('abcdef', 'uvwxyz', prompt='q'))
        tokens, lengths = tokenize_pair(Tokenizer(), record, 6)
        self.assertEqual(tokens['prompt_ids'], [1, ord('q'), 2])
        self.assertEqual(tokens['chosen_ids'], list(map(ord, 'abc')))
        self.assertEqual(lengths['chosen'], {'original': 10, 'retained': 6})

    def test_drop_empty_and_indistinguishable_truncated_pairs(self):
        for record, limit in [(preference(row(prompt='q')), 3),
                              (preference(row('same good', 'same bad', prompt='q')), 7)]:
            with self.assertRaises(ValueError):
                tokenize_pair(Tokenizer(), record, limit)

    def test_template_prefix_mismatch_is_rejected(self):
        class BadTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if kwargs.get('add_generation_prompt'):
                    result['input_ids'].append(99)
                return result
        with self.assertRaisesRegex(ValueError, 'exact generation prompt prefix'):
            tokenize_pair(BadTokenizer(), preference(row()), 64)

    def test_split_overlap_and_file_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = []
            for id in ('a', 'b'):
                record = preference(row(prompt=id, id=id))
                tokens, _ = tokenize_pair(Tokenizer(), record, 64)
                records.append({'id': id, 'prompt_hash': record['prompt_hash'], **tokens})
            manifest = {'max_length': 64, 'splits': {}}
            def save():
                for split, record in zip(('train', 'eval'), records):
                    target = path / f'{split}.jsonl'
                    target.write_text(json.dumps(record) + '\n')
                    manifest['splits'][split] = {'rows': 1, 'sha256': sha256(target)}
                (path / 'manifest.json').write_text(json.dumps(manifest))
            save()
            self.assertEqual(len(load_splits(path)[1]['train']), 1)
            (path / 'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_splits(path)
            records[1]['prompt_hash'] = records[0]['prompt_hash']
            save()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_splits(path)


if __name__ == '__main__':
    unittest.main()
