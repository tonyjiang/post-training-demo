"""Refuse checkpoint resumes against different data or training state."""
import json
from pathlib import Path
import tempfile
import unittest
from rlvr.resume import validate_checkpoint


class ResumeTests(unittest.TestCase):
    def prepare(self, folder):
        root=Path(folder);checkpoint=root/'checkpoint-2';checkpoint.mkdir()
        config=json.loads(Path('configs/rlvr-wildchat.json').read_text())['experiment']
        for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth'):
            (checkpoint/name).write_bytes(b'test-state')
        (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':2}))
        (root/'run-config.json').write_text(json.dumps({'data_manifest':{'id':'data'},'base_manifest':{'id':'base'},
            'parent_adapter_sha256':'parent','recipe':{'experiment':config},'arguments':{'max_completion_length':2048}}))
        (root/'rollouts.jsonl').write_text(''.join(json.dumps({'id':f'p{i}'})+'\n' for i in range(2) for _ in range(4)))
        return checkpoint,config

    def test_complete_checkpoint_accepts_larger_completion_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint,config=self.prepare(folder);config['max_completion_length']=3072
            result=validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,8)
            self.assertEqual(result['step'],2)
            self.assertEqual(result['prior_completion_length'],2048)
            self.assertEqual(len(result['files']),6)

    def test_mismatched_or_incomplete_state_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint,config=self.prepare(folder)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                validate_checkpoint(checkpoint,{'id':'other'},{'id':'base'},'parent',config,8)
            with self.assertRaisesRegex(ValueError,'later target'):
                validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,2)
            (checkpoint/'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError,'lacks'):
                validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,8)

    def test_generation_change_requires_opt_in_and_preserves_prompt_position(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint,config=self.prepare(folder)
            config.update(num_generations=3,gradient_accumulation_steps=3)
            with self.assertRaisesRegex(ValueError,'opt-in'):
                validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,12)
            result=validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,12,True)
            self.assertEqual(result['consumed_prompt_ids'],['p0','p1'])
            self.assertEqual(result['generation_count_change']['to'],3)
            config['gradient_accumulation_steps']=4
            with self.assertRaisesRegex(ValueError,'one prompt'):
                validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,12,True)

    def test_generation_change_rejects_incomplete_history(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint,config=self.prepare(folder)
            config.update(num_generations=3,gradient_accumulation_steps=3)
            (checkpoint.parent/'rollouts.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'committed rollout'):
                validate_checkpoint(checkpoint,{'id':'data'},{'id':'base'},'parent',config,12,True)
