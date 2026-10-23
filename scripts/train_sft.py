"""Small Dolci SFT experiment: frozen NF4 Olmo base + rank-8 LoRA adapters."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed)


class CheckFiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ('loss', 'eval_loss', 'grad_norm'):
            if logs and name in logs and not math.isfinite(logs[name]):
                raise RuntimeError(f'Nonfinite {name}: {logs[name]}')


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--data', default='data/dolci-small')
    parser.add_argument('--output-dir')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs before submitting a detached run')
    parser.add_argument('--liger-fused-loss', action='store_true', help='Use the tested OLMo-3 fused loss to fit 4K training')
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--save-steps', type=int, default=25)
    parser.add_argument('--eval-limit', type=int)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument('--resume-from-checkpoint', help='Resume the same data and training state')
    continuation.add_argument('--init-adapter', help='Continue learned adapter weights on new data with a fresh optimizer/schedule')
    args = parser.parse_args()
    if not args.check_only and not args.output_dir:
        parser.error('--output-dir is required for training')
    if args.gradient_accumulation < 1 or args.save_steps < 1 or args.epochs <= 0:
        parser.error('Gradient accumulation, save steps and epochs must be positive')
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error('--eval-limit must be positive')
    set_seed(42)
    data = Path(args.data)
    if not (data / 'manifest.json').is_file():
        parser.error(f'{data} is not prepared training data: missing manifest.json. '
                     'For the expanded sample, run scripts/prepare_combined_sft.py first.')
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest.get('required_training_options', {}).get('liger_fused_loss') and not args.liger_fused_loss:
        parser.error('This data requires --liger-fused-loss to fit our GPU')
    parent = None
    if args.init_adapter:
        adapter = Path(args.init_adapter).resolve()
        parent = {'path': str(adapter), 'sha256': hashlib.sha256(
            (adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()}
    if args.resume_from_checkpoint:
        previous = json.loads((Path(args.resume_from_checkpoint).parent / 'run-config.json').read_text())
        if previous['data_manifest'] != manifest:
            parser.error('Checkpoint resume requires the same data; use --init-adapter for a new dataset.')
    splits = {}
    seen_ids = set()
    for split in ['train', 'eval']:
        content = (data / f'{split}.jsonl').read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest['splits'][split]['sha256']
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == manifest['splits'][split]['rows'] and rows
        for row in rows:
            assert row['id'] not in seen_ids, 'Duplicate ID or train/eval overlap'
            seen_ids.add(row['id'])
            assert 1 < len(row['input_ids']) <= manifest['max_length']
            assert len(row['input_ids']) == len(row['attention_mask']) == len(row['labels'])
            assert all(value == 1 for value in row['attention_mask'])
            assert any(x != -100 for x in row['labels'][1:]), 'No assistant prediction target'
            assert all(label == -100 or label == token for token, label in zip(row['input_ids'], row['labels']))
        splits[split] = [{k: r[k] for k in ['input_ids', 'attention_mask', 'labels']} for r in rows]
    if args.eval_limit:
        splits['eval'] = splits['eval'][:args.eval_limit]
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 2**30:
        raise RuntimeError('Need at least 9 GiB free VRAM; stop other model servers first.')
    tokenizer = AutoTokenizer.from_pretrained(data / 'tokenizer', local_files_only=True)
    if args.liger_fused_loss:
        from liger_kernel.transformers import apply_liger_kernel_to_olmo3
        apply_liger_kernel_to_olmo3(rope=False, rms_norm=False, swiglu=False,
                                   cross_entropy=False, fused_linear_cross_entropy=True)
    if args.check_only:
        print(f"Preflight passed: {len(splits['train'])} train, {len(splits['eval'])} eval, "
              f"{manifest['max_length']}-token cap; random conversation sampler", flush=True)
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run-config.json').write_text(json.dumps({**vars(args), 'parent_adapter': parent, 'data_manifest': manifest,
        'base_manifest': json.loads((Path(args.model) / 'download-manifest.json').read_text())}, indent=2))
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True,
                                          local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                               target_modules='all-linear', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(splits['train']) / args.gradient_accumulation) * args.epochs)
    training_args = TrainingArguments(output_dir=str(output),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4, lr_scheduler_type='cosine', warmup_steps=math.ceil(total_steps * 0.05),
        bf16=True, tf32=True, optim='adamw_torch', max_grad_norm=1.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, logging_nan_inf_filter=False, disable_tqdm=True, report_to='none',
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=2,
        eval_strategy='steps', eval_steps=args.save_steps, prediction_loss_only=True,
        train_sampling_strategy='random', seed=42, data_seed=42)
    (output / 'training-args.json').write_text(training_args.to_json_string())
    trainer = Trainer(model=model, args=training_args, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        train_dataset=splits['train'], eval_dataset=splits['eval'], callbacks=[CheckFiniteLoss()])
    print(f'Training sampler: {type(trainer._get_train_sampler()).__name__}; seed 42', flush=True)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output / 'final-adapter'))
    trainer.save_state()
    metrics = {**result.metrics, **trainer.evaluate(),
               'process_runtime_seconds': time.monotonic() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    trainer.save_metrics('all', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
