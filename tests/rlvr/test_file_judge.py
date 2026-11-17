"""Interactive rewards are bound to exact requests; absent scores never become zero."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from rlvr.file_judge import FileJudge


class FileJudgeTests(unittest.TestCase):
    def make(self, folder, timeout=0):
        return FileJudge(json.loads(Path('configs/rlvr-wildchat.json').read_text()),folder,timeout=timeout)

    def reply(self, judge, score=8, **overrides):
        digest=hashlib.sha256(judge.template.format(input='Q',output='A').encode()).hexdigest()
        result={'request_sha256':digest,'judge':judge.identity,'score':score,'reasoning':'Relevant and correct.'}
        result.update(overrides);judge.queue.mkdir(parents=True,exist_ok=True)
        (judge.queue/f'{digest}.response.json').write_text(json.dumps(result))

    def test_bound_score_and_request_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            judge=self.make(folder);self.reply(judge)
            self.assertEqual(judge.score('Q','A')['reward'],0.8)
            request=json.loads(next(judge.queue.glob('*.request.json')).read_text())
            self.assertEqual(request['answer'],'A')
            self.assertEqual(hashlib.sha256(request['prompt'].encode()).hexdigest(),request['request_sha256'])

    def test_batch_publishes_all_requests_before_collecting(self):
        with tempfile.TemporaryDirectory() as folder:
            judge=self.make(folder)
            for query,answer,score in [('Q1','A1',8),('Q2','A2',3)]:
                digest=hashlib.sha256(judge.template.format(input=query,output=answer).encode()).hexdigest()
                judge.queue.mkdir(parents=True,exist_ok=True)
                (judge.queue/f'{digest}.response.json').write_text(json.dumps({
                    'request_sha256':digest,'judge':judge.identity,'score':score,'reasoning':'Audited example'}))
            collect=judge.collect
            def checked_collect(pending):
                self.assertEqual(len(list(judge.queue.glob('*.request.json'))),2)
                return collect(pending)
            judge.collect=checked_collect
            self.assertEqual([r['reward'] for r in judge.score_many([('Q1','A1'),('Q2','A2')])],[0.8,0.3])

    def test_timeout_without_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):self.make(folder).score('Q','A')

    def test_bad_score_or_wrong_request_rejected(self):
        for overrides in ({'score':True},{'score':11},{'request_sha256':'wrong'},{'reasoning':''}):
            with self.subTest(overrides=overrides),tempfile.TemporaryDirectory() as folder:
                judge=self.make(folder);self.reply(judge,**overrides)
                with self.assertRaises(ValueError):judge.score('Q','A')
