"""Continue the SFT adapter with length-normalized DPO on a frozen NF4 base."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

from .data import load_splits, sha256


def adapter_digest(model, adapter):
    """Hash adapter tensors independently of the adapter's name."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if f'.{adapter}.' in name:
            digest.update(name.replace(f'.{adapter}.', '.adapter.').encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def trainer_class():
    from trl import DPOTrainer

    class PreparedDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            # Preparation already pins token IDs, masks and truncation. The standard
            # TRL collator/loss consume these fields; avoid re-rendering conversations.
            required = {'prompt_ids', 'chosen_ids', 'rejected_ids'}
            if not required.issubset(dataset.column_names):
                raise ValueError(f'{dataset_name} is not prepared DPO data')
            return dataset

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # Transformers 5.20.2 sees the ref/ subdirectory and skips the default
            # adapter at the checkpoint root. Restore the policy explicitly into
            # existing parameters; keep the original SFT reference and parameter order.
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.torch import load_file

            model = model if model is not None else self.model
            weights = load_file(str(Path(resume_from_checkpoint) / 'adapter_model.safetensors'))
            expected = get_peft_model_state_dict(model, adapter_name='default')
            if weights.keys() != expected.keys():
                raise ValueError('Checkpoint policy adapter keys differ from the SFT adapter')
            reference_before = adapter_digest(model, 'ref')
            set_peft_model_state_dict(model, weights, adapter_name='default')
            if adapter_digest(model, 'ref') != reference_before:
                raise RuntimeError('Checkpoint loading changed the fixed SFT reference')

    return PreparedDPOTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dpo.json')
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--data', default='data/dolci-dpo-small')
    parser.add_argument('--init-adapter')
    parser.add_argument('--output-dir')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--warmup-ratio', type=float)
    parser.add_argument('--lr-scheduler', choices=['linear', 'cosine', 'constant_with_warmup'])
    parser.add_argument('--beta', type=float)
    parser.add_argument('--epochs', type=float)
    parser.add_argument('--gradient-accumulation', type=int)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--save-steps', type=int, default=25)
    parser.add_argument('--resume-from-checkpoint')
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text())
    defaults = recipe['experiment']
    for name in ('learning_rate', 'warmup_ratio', 'beta', 'epochs', 'gradient_accumulation', 'init_adapter'):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    args.lr_scheduler = args.lr_scheduler or defaults['lr_scheduler_type']
    if (not all(math.isfinite(value) and value > 0 for value in (args.learning_rate, args.beta, args.epochs))
            or not 0 <= args.warmup_ratio < 1 or args.gradient_accumulation < 1 or args.save_steps < 1
            or (args.max_steps != -1 and args.max_steps < 1)):
        parser.error('Invalid learning rate, beta, epochs, warmup, accumulation, or step count')
    if not args.check_only and not args.output_dir:
        parser.error('--output-dir is required')
    manifest, splits = load_splits(args.data)
    if manifest['max_length'] > defaults['max_length']:
        parser.error('Prepared context exceeds the configured memory budget')
    pinned_data = json.loads(Path('configs/dpo-data.json').read_text())
    if manifest['source_manifest'] != pinned_data:
        parser.error('Prepared data differs from the pinned DPO source')
    parent = Path(args.init_adapter)
    parent_info = {'path': str(parent.resolve()), 'sha256': sha256(parent / 'adapter_model.safetensors'),
                   'config_sha256': sha256(parent / 'adapter_config.json')}
    base_manifest = json.loads((Path(args.model) / 'download-manifest.json').read_text())
    if base_manifest != json.loads(Path('configs/base-model.json').read_text()):
        parser.error('Base model manifest differs from the pinned SFT base')
    training = {key: getattr(args, key) for key in ('learning_rate', 'warmup_ratio', 'lr_scheduler',
                'beta', 'epochs', 'gradient_accumulation', 'max_steps')}
    run_config = {'training': training, 'data_manifest': manifest, 'parent_adapter': parent_info,
                  'base_manifest': base_manifest, 'recipe': recipe}
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint)
        previous = json.loads((checkpoint.parent / 'run-config.json').read_text())
        for key in run_config:
            if previous[key] != run_config[key]:
                parser.error(f'Resume would change {key}; use the same data, SFT reference and schedule')
        required = ('trainer_state.json', 'adapter_model.safetensors', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth')
        if any(not (checkpoint / name).is_file() for name in required):
            parser.error('Resume requires policy, optimizer, scheduler, RNG, and Trainer state')

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(args.data) / 'tokenizer', local_files_only=True)
    template = json.loads(Path('configs/sft-tokenizer-config.json').read_text())['chat_template']
    if tokenizer.chat_template != template or hashlib.sha256(template.encode()).hexdigest() != manifest['chat_template_sha256']:
        parser.error('Prepared tokenizer does not use the pinned SFT chat template')
    vocab_size = len(tokenizer)
    if any(token >= vocab_size for rows in splits.values() for row in rows
           for key in ('prompt_ids', 'chosen_ids', 'rejected_ids') for token in row[key]):
        parser.error('Prepared token ID exceeds tokenizer vocabulary')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 9 * 2**30:
        parser.error('Need at least 9 GiB free CUDA memory; stop other GPU jobs first')
    if shutil.disk_usage('.').free < 2 * 2**30:
        parser.error('Need at least 2 GiB free disk for adapters, checkpoints and reference cache')
    if args.check_only:
        print(f"DPO preflight passed: {len(splits['train'])} train / {len(splits['eval'])} eval pairs; "
              f"{manifest['max_length']}-token cap; SFT reference {parent_info['sha256']}", flush=True)
        return

    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import DPOConfig

    class CheckFinite(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, (int, float)) and not math.isfinite(value)
                   for value in (logs or {}).values()):
                raise RuntimeError(f'Nonfinite DPO metric: {logs}')

    started = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {'train.log', 'pid'} for path in output.iterdir()):
        parser.error(f'Refusing to overwrite nonempty run directory {output}')
    set_seed(defaults['seed'])
    torch.cuda.reset_peak_memory_stats()
    run_config['versions'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'peft', 'bitsandbytes', 'trl', 'datasets')}
    run_config['arguments'] = vars(args)
    run_config['gpu'] = torch.cuda.get_device_name()
    run_config['cuda'] = torch.version.cuda
    (output / 'run-config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = PeftModel.from_pretrained(model, parent, is_trainable=True, local_files_only=True)
    # Keep FP32 adapter/optimizer precision, as in SFT. TRL casts QLoRA policy
    # weights to BF16 during construction; restore them before optimizer creation.
    initial = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    if not initial or any('.default.' not in name or 'lora_' not in name for name in initial):
        raise ValueError('Expected only a trainable default LoRA adapter')
    initial_digest = adapter_digest(model, 'default')
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(splits['train']) / args.gradient_accumulation) * args.epochs)
    training_args = DPOConfig(output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler,
        warmup_steps=math.ceil(total_steps * args.warmup_ratio), weight_decay=0.0,
        loss_type=[defaults['loss_type']], beta=args.beta, max_length=manifest['max_length'],
        truncation_mode='keep_start', precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        bf16=True, tf32=True, optim='adamw_torch', max_grad_norm=1.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, logging_nan_inf_filter=False, disable_tqdm=True, report_to='none',
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=2,
        eval_strategy='steps', eval_steps=args.save_steps, seed=defaults['seed'], data_seed=defaults['seed'])
    trainer = trainer_class()(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=Dataset.from_list(splits['train']), eval_dataset=Dataset.from_list(splits['eval']),
        callbacks=[CheckFinite()])
    for name, value in initial.items():
        parameter = model.get_parameter(name)
        parameter.data = value.to(parameter.device)
    del initial
    if 'ref' not in model.peft_config or adapter_digest(model, 'ref') != initial_digest:
        raise RuntimeError('DPO reference must be an exact frozen copy of our SFT adapter')
    if any(p.requires_grad for name, p in model.named_parameters() if '.ref.' in name):
        raise RuntimeError('Reference adapter is trainable')
    if adapter_digest(model, 'default') != initial_digest:
        raise RuntimeError('DPO policy initialization differs from SFT')
    if any(p.requires_grad and ('.default.' not in name or 'lora_' not in name)
           for name, p in model.named_parameters()):
        raise RuntimeError('Unexpected trainable base/reference weights')
    (output / 'training-args.json').write_text(training_args.to_json_string())
    # Save the exact cached SFT scores alongside their token IDs for inspection.
    for split, dataset in [('train', trainer.train_dataset), ('eval', trainer.eval_dataset)]:
        dataset.to_json(str(output / f'reference-{split}.jsonl'))
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    if adapter_digest(model, 'ref') != initial_digest:
        raise RuntimeError('Frozen SFT reference changed during DPO')
    final = output / 'final-adapter'
    model.save_pretrained(final, selected_adapters=['default'])
    tokenizer.save_pretrained(final)
    trainer.save_state()
    metrics = {**result.metrics, **evaluation, 'optimizer_steps': trainer.state.global_step,
               'reference_unchanged': True, 'initial_adapter_digest': initial_digest,
               'final_adapter_digest': adapter_digest(model, 'default'),
               'final_adapter_sha256': sha256(final / 'adapter_model.safetensors'),
               'process_runtime_seconds': time.monotonic() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    (output / 'all_results.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
