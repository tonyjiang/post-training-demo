"""Quality rewards fail closed on judge errors; preserve conversations."""
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from rlvr.quality import QualityJudge, final_answer
from rlvr.wildchat_data import parse_conversation, exceeds_word_budget


class ConversationTests(unittest.TestCase):
    def test_requested_length_filter(self):
        for text in ('Write a 2000 word blog', 'Write a 2,000-word blog', 'Write a 1500–word essay'):
            self.assertTrue(exceeds_word_budget([{'role':'user','content':text}],600))
        self.assertFalse(exceeds_word_budget([{'role':'user','content':'Write 600 words'}],600))
        self.assertFalse(exceeds_word_budget([{'role':'assistant','content':'A 2000 word example'}],600))

    def test_preserve_multiturn(self):
        rows=parse_conversation('system: Be concise\nuser: Hi\nassistant: Hello\nuser: Explain it')
        self.assertEqual([r['role'] for r in rows],['system','user','assistant','user'])
        self.assertEqual(rows[-1]['content'],'Explain it')

    def test_reject_incomplete_or_malformed(self):
        for text in ('user: \nassistant: Hi\nuser: Q','user: Q\nuser: Again','assistant: A','prefix\nuser: Q'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_conversation(text)

    def test_strip_thinking_and_stop_tokens(self):
        self.assertEqual(final_answer('secret reasoning</think> Answer<|im_end|>'),'Answer')
        self.assertEqual(final_answer('unfinished thinking'),'')


class JudgeTests(unittest.TestCase):
    def make_judge(self):
        recipe=json.loads(Path('configs/rlvr-wildchat.json').read_text())
        recipe['judge']['max_attempts']=1
        with patch.dict(os.environ,{'RLVR_JUDGE_BASE_URL':'http://localhost:1234/v1','RLVR_JUDGE_MODEL':'test'}):
            return QualityJudge(recipe)

    def reply(self, content, finish='stop'):
        return io.BytesIO(json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}]}).encode())

    def test_normalizes_and_uses_final_answer_only(self):
        judge=self.make_judge()
        with patch('urllib.request.urlopen',return_value=self.reply('{"REASONING":"Good","SCORE":"8"}')) as call:
            result=judge.score('user: Q','A')
        self.assertEqual(result['reward'],0.8)
        body=json.loads(call.call_args.args[0].data)
        self.assertIn('[AI Answer]\nA',body['messages'][0]['content'])

    def test_invalid_scores_and_failed_responses_raise(self):
        judge=self.make_judge()
        for text,finish in [('garbage','stop'),('[]','stop'),('{"SCORE":11}','stop'),('{"SCORE":true}','stop'),
                            ('{"SCORE":8}','length')]:
            with self.subTest(text=text),patch('urllib.request.urlopen',return_value=self.reply(text,finish)):
                with self.assertRaisesRegex(RuntimeError,'refusing to substitute zero'):judge.score('Q','A')
        with patch('urllib.request.urlopen',side_effect=TimeoutError):
            with self.assertRaises(RuntimeError):judge.score('Q','A')

    def test_missing_endpoint_is_explicit(self):
        recipe=json.loads(Path('configs/rlvr-wildchat.json').read_text())
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'Configure'):
            QualityJudge(recipe)
