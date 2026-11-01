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
        counts=[]
        def rewards(completions,**kwargs):
            counts.append(len(completions))
            return [float(i%2) for i in range(len(completions))]
        with tempfile.TemporaryDirectory() as directory:
            args=GRPOConfig(output_dir=directory,use_cpu=True,bf16=False,per_device_train_batch_size=1,
                gradient_accumulation_steps=4,num_generations=4,max_completion_length=8,
                max_steps=1,learning_rate=0.01,beta=0.0,loss_type='grpo',
                gradient_checkpointing=False,mask_truncated_completions=False,
                generation_kwargs={'eos_token_id':[2,3]},
                report_to='none',save_strategy='no',disable_tqdm=True,logging_steps=1)
            trainer=trainer_class()(model=model,args=args,processing_class=tokenizer,
                train_dataset=Dataset.from_list([{'prompt':'t4'},{'prompt':'t5'}]),reward_funcs=rewards)
            self.assertEqual(set(trainer.eos_token_ids),{2,3})
            trainer.train()
        self.assertEqual(counts,[4])
        self.assertNotEqual(adapter_digest(model,'default'),initial)
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]),n)
