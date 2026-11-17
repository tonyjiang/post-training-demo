"""Batch publication preserves truncation handling, order, and math/API fallback."""
import unittest
from rlvr.reward_batch import score_records, training_rewards


class BatchRewardTests(unittest.TestCase):
    def test_cutoffs_do_not_enter_relative_reward_comparison(self):
        records=[{'truncated':False},{'truncated':True},{'truncated':False}]
        scores=[{'reward':0.5},{'reward':0.0},{'reward':0.9}]
        self.assertEqual(training_rewards(records,scores,3),[0.5,None,0.9])
        # Two equal finished answers provide no contrast, even beside a cutoff.
        scores[2]['reward']=0.5
        self.assertEqual(training_rewards(records,scores,3),[0.5,None,0.5])

    def test_fewer_than_two_finished_answers_have_no_learning_signal(self):
        scores=[{'reward':0.9},{'reward':0.0},{'reward':0.0}]
        for flags in ([False,True,True],[True,True,True]):
            records=[{'truncated':flag} for flag in flags]
            self.assertEqual(training_rewards(records,scores,3),[0.0,0.0,0.0])

    def test_one_call_with_final_answers_only(self):
        class Judge:
            def score_many(self,pairs):
                self.pairs=pairs
                return [{'reward':0.8},{'reward':0.0},{'reward':0.0}]
        judge=Judge()
        records=[{'text':'private</think>Answer','gold':'unused','query':'Q','truncated':False},
                 {'text':'private</think>partial','gold':'unused','query':'Q','truncated':True},
                 {'text':'unfinished thinking','gold':'unused','query':'Q','truncated':False}]
        result=score_records(records,judge,lambda *args: self.fail('Unexpected fallback'))
        self.assertEqual(judge.pairs,[('Q','Answer'),('Q',''),('Q','')])
        self.assertEqual([r['reward'] for r in result],[0.8,0.0,0.0])
        self.assertEqual(result[1]['reasoning'],'Token-truncated completion')

    def test_existing_fallback_is_preserved(self):
        records=[{'text':'T','gold':'G','query':'Q','truncated':False}]
        seen=[]
        result=score_records(records,None,lambda *args: seen.append(args) or {'reward':1.0})
        self.assertEqual(seen,[('T','G',False,'Q')])
        self.assertEqual(result,[{'reward':1.0}])
