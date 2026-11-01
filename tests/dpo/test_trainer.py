"""Tiny CPU model integration: SFT reference, normalized loss, and frozen weights."""
import importlib.util
import copy
import math
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('trl'), 'Requires the dpo dependency group')
class TrainerTests(unittest.TestCase):
    def test_reference_and_preference_update(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Olmo3Config, Olmo3ForCausalLM, PreTrainedTokenizerFast
        from trl import DPOConfig

        from dpo.train import adapter_digest, trainer_class

        torch.manual_seed(42)
        torch.set_num_threads(2)
        config = Olmo3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=64, pad_token_id=0,
                            bos_token_id=1, eos_token_id=2)
        model = get_peft_model(Olmo3ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        # Simulate a learned SFT adapter, rather than a zero-initialized LoRA.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=0.2)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {f't{i}': i for i in range(32)}, unk_token='t3')),
            pad_token='t0', bos_token='t1', eos_token='t2', unk_token='t3')
        rows = [{'prompt_ids': [1, 4, 5], 'chosen_ids': [6, 7, 2], 'rejected_ids': [8, 2]},
                {'prompt_ids': [1, 9], 'chosen_ids': [10, 2], 'rejected_ids': [11, 12, 13, 2]}]
        base_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        initial = adapter_digest(model, 'default')
        resume_model = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as directory:
            args = DPOConfig(output_dir=directory, use_cpu=True, bf16=False,
                loss_type=['sigmoid_norm'], beta=5.0, precompute_ref_log_probs=True,
                max_length=16, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                max_steps=2, learning_rate=0.01, gradient_checkpointing=False,
                report_to='none', save_strategy='steps', save_steps=1, logging_steps=1, disable_tqdm=True)
            trainer = trainer_class()(model=model, args=args, processing_class=tokenizer,
                train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows))
            self.assertEqual(adapter_digest(model, 'ref'), initial)
            self.assertFalse(any(p.requires_grad for name, p in model.named_parameters() if '.ref.' in name))
            batch = trainer.data_collator([trainer.train_dataset[0]])
            self.assertEqual(batch['completion_mask'][0].tolist(), [0, 0, 0, 1, 1, 1])
            # At policy == SFT reference, the DPO loss is log(2).
            self.assertAlmostEqual(trainer.evaluate()['eval_loss'], math.log(2), places=5)
            # The cached scores must come from SFT, not the adapter-disabled base.
            with model.disable_adapter(), torch.no_grad():
                labels = batch['input_ids'].masked_fill(batch['completion_mask'] == 0, -100)
                base_scores = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                    labels=labels, fused_lm_head=True).log_probs.sum(dim=1)
            cached = torch.stack([batch['ref_chosen_logps'][0], batch['ref_rejected_logps'][0]])
            self.assertFalse(torch.allclose(base_scores, cached, atol=1e-6))
            trainer.train()
            self.assertNotEqual(adapter_digest(model, 'default'), initial)
            self.assertEqual(adapter_digest(model, 'ref'), initial)
            for name, parameter in base_before.items():
                self.assertTrue(torch.equal(model.get_parameter(name), parameter), name)
            # Compare TRL's loss against the explicit length-normalized DPO equation.
            model.eval()
            with torch.no_grad():
                labels = batch['input_ids'].masked_fill(batch['completion_mask'] == 0, -100)
                scores = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               labels=labels, fused_lm_head=True).log_probs.sum(dim=1)
                normalized = (scores - cached) / batch['completion_mask'].sum(dim=1)
                expected = -torch.nn.functional.logsigmoid(5 * (normalized[0] - normalized[1]))
                actual = trainer.compute_loss(model, batch)
            self.assertAlmostEqual(actual.item(), expected.item(), places=5)
            final = Path(directory) / 'adapter'
            model.save_pretrained(final, selected_adapters=['default'])
            model.load_adapter(final, adapter_name='reloaded')
            self.assertEqual(adapter_digest(model, 'default'), adapter_digest(model, 'reloaded'))
            resume_args = DPOConfig(output_dir=str(Path(directory) / 'resumed'), use_cpu=True, bf16=False,
                loss_type=['sigmoid_norm'], beta=5.0, precompute_ref_log_probs=True,
                max_length=16, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                max_steps=2, learning_rate=0.01, gradient_checkpointing=False,
                report_to='none', save_strategy='no', save_steps=1, logging_steps=1, disable_tqdm=True)
            resumed = trainer_class()(model=resume_model, args=resume_args, processing_class=tokenizer,
                train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows))
            resumed.train(resume_from_checkpoint=str(Path(directory) / 'checkpoint-1'))
            self.assertEqual(resumed.state.global_step, 2)
            self.assertEqual(adapter_digest(resume_model, 'ref'), initial)
            for name, parameter in resume_model.named_parameters():
                if '.default.' in name:
                    torch.testing.assert_close(parameter, model.get_parameter(name), rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
