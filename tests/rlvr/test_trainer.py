"""Tiny CPU GRPO integration with known group rewards and a frozen base."""
import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('trl'), 'Requires rlvr dependency group')
class GRPOTrainerTests(unittest.TestCase):
    def test_sequential_rollouts_update_only_adapter(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Olmo3Config, Olmo3ForCausalLM, PreTrainedTokenizerFast
        from trl import GRPOConfig
        from dpo.train import adapter_digest
        from rlvr.train import trainer_class
        torch.manual_seed(42);torch.set_num_threads(2)
        model=get_peft_model(Olmo3ForCausalLM(Olmo3Config(vocab_size=32,hidden_size=16,
            intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
            head_dim=8,max_position_embeddings=64,pad_token_id=0,bos_token_id=1,eos_token_id=2)),
            LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {f't{i}':i for i in range(32)},unk_token='t3')),pad_token='t0',bos_token='t1',
            eos_token='t2',unk_token='t3',padding_side='left')
        model.generation_config.eos_token_id=[2,3]
        initial=adapter_digest(model,'default')
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        counts=[];seen=[]
        def rewards(completions,**kwargs):
            counts.append(len(completions))
            seen.append(kwargs['id'])
            return [float(i%2) for i in range(len(completions))]
        def completed_rewards(completions,**kwargs):
            # Exercise TRL's None reward support: the omitted response must
            # not shift the two equal completed responses' baseline.
            self.assertEqual(len(completions),3)
            return [0.5,None,0.5]
        with tempfile.TemporaryDirectory() as directory:
            args=GRPOConfig(output_dir=directory,use_cpu=True,bf16=False,per_device_train_batch_size=1,
                gradient_accumulation_steps=4,num_generations=4,max_completion_length=8,
                max_steps=1,learning_rate=0.01,lr_scheduler_type='constant',beta=0.0,loss_type='grpo',
                gradient_checkpointing=False,mask_truncated_completions=False,
                generation_kwargs={'eos_token_id':[2,3]},
                report_to='none',save_steps=1,save_total_limit=2,disable_tqdm=True,logging_steps=1)
            trainer=trainer_class()(model=model,args=args,processing_class=tokenizer,
                train_dataset=Dataset.from_list([{'prompt':f't{i+4}','id':f'p{i}'} for i in range(4)]),reward_funcs=rewards)
            expected_order=[f'p{i}' for i in dict.fromkeys(trainer._get_train_sampler())]
            self.assertEqual(set(trainer.eos_token_ids),{2,3})
            trainer.train()
            self.assertEqual(counts,[4])
            saved=adapter_digest(model,'default')
            from pathlib import Path
            checkpoint=Path(directory)/'checkpoint-1'
            self.assertTrue((checkpoint/'optimizer.pt').is_file())
            from dataclasses import replace
            from transformers import TrainerCallback
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:parameter.add_(0.25)
            outer=self
            class CheckRestore(TrainerCallback):
                def on_train_begin(self,args,state,control,**kwargs):
                    outer.assertEqual(adapter_digest(kwargs['model'],'default'),saved)
            resumed=trainer_class()(model=model,args=replace(args,output_dir=str(Path(directory)/'resumed'),max_steps=2,generation_batch_size=None),
                processing_class=tokenizer,train_dataset=trainer.train_dataset,reward_funcs=rewards,
                callbacks=[CheckRestore()])
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step,2)
            self.assertNotEqual(adapter_digest(model,'default'),saved)
            saved=adapter_digest(model,'default')
            checkpoint=Path(directory)/'resumed/checkpoint-2'
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:parameter.add_(0.25)
            from rlvr.resume import restore_policy
            restore_policy(model,checkpoint)
            self.assertEqual(adapter_digest(model,'default'),saved)
            class CheckOptimizerRestore(CheckRestore):
                def on_train_begin(self,args,state,control,**kwargs):
                    super().on_train_begin(args,state,control,**kwargs)
                    outer.assertTrue(kwargs['optimizer'].state)
                    outer.assertTrue(all(float(s['step'])==2 for s in kwargs['optimizer'].state.values()))
            three=trainer_class()(model=model,args=replace(args,output_dir=str(Path(directory)/'three'),
                max_steps=3,gradient_accumulation_steps=3,num_generations=3,
                generation_batch_size=None,steps_per_generation=None),
                processing_class=tokenizer,train_dataset=trainer.train_dataset,reward_funcs=rewards,
                callbacks=[CheckOptimizerRestore()])
            self.assertEqual([f'p{i}' for i in dict.fromkeys(three._get_train_sampler())],expected_order)
            three.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(three.state.global_step,3)
            self.assertNotEqual(adapter_digest(model,'default'),saved)
            saved=adapter_digest(model,'default')
            check=trainer_class()(model=model,args=replace(three.args,
                output_dir=str(Path(directory)/'cutoff-check'),max_steps=1,
                generation_batch_size=None,steps_per_generation=None),
                processing_class=tokenizer,train_dataset=trainer.train_dataset,
                reward_funcs=completed_rewards)
            check.train()
            self.assertEqual(adapter_digest(model,'default'),saved)
            self.assertTrue(all(entry.get('grad_norm',0)==0 for entry in check.state.log_history))
        self.assertEqual(counts,[4,4,3])
        self.assertEqual(seen,[[expected_order[0]]*4,[expected_order[1]]*4,[expected_order[2]]*3])
        self.assertNotEqual(adapter_digest(model,'default'),initial)
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]),n)
