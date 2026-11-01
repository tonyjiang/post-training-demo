"""Continue our DPO adapter with single-GPU GRPO and pinned math verification."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from dpo.data import sha256
from dpo.train import adapter_digest
from .data import load_splits
from .rewards import load_math_utils, verify_answer


def trainer_class():
    from trl import GRPOTrainer

    class SequentialGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations,
                                  has_tool_images=False):
            if images is not None or multimodal_fields or has_tool_images:
                raise ValueError('This pilot supports text-only math')
            completions = []
            for index, ids in enumerate(prompt_ids, 1):
                generated, logprobs, sampling_mask = super()._generate_single_turn(
                    [ids], None, {}, num_generations=1)
                if logprobs is not None or sampling_mask is not None or len(generated) != 1:
                    raise RuntimeError('Unexpected rollout backend result')
                completions.extend(generated)
                print(f"rollout: {index}/{len(prompt_ids)}, {len(generated[0])} completion tokens", flush=True)
            return completions, None, None

    return SequentialGRPOTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/dolci-rlvr-math')
    p.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, default=16, help='Bounded pilot by default; -1 runs one epoch')
    p.add_argument('--max-completion-length', type=int, default=2048)
    p.add_argument('--eval-size', type=int, default=4, help='Fixed held-out prompts evaluated before and after the pilot')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    recipe = json.loads(Path('configs/rlvr.json').read_text()); config = recipe['experiment']
    if args.max_steps != -1 and args.max_steps < 1: p.error('Invalid max steps')
    if not 1 <= args.eval_size <= config['eval_size']: p.error('Invalid evaluation size')
    if not 128 <= args.max_completion_length <= config['max_completion_length']: p.error('Invalid completion length')
    manifest, splits = load_splits(args.data)
    parent = Path(config['init_adapter'])
    parent_hash = sha256(parent / 'adapter_model.safetensors')
    dpo_results = json.loads((parent.parent / 'all_results.json').read_text())
    if parent_hash != dpo_results['final_adapter_sha256']: raise ValueError('DPO parent hash mismatch')
    base = json.loads((Path(args.model) / 'download-manifest.json').read_text())
    if base != json.loads(Path('configs/base-model.json').read_text()): raise ValueError('Base manifest mismatch')
    utils = load_math_utils()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Path(args.data) / 'tokenizer', local_files_only=True, padding_side='left')
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != manifest['chat_template_sha256']:
        raise ValueError('Chat template mismatch')
    if args.check_only:
        print(f"Verified {len(splits['train'])}/{len(splits['eval'])} math prompts and DPO parent {parent_hash}")
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(x.name not in {'train.log', 'pid'} for x in output.iterdir()):
        p.error('Refusing to overwrite run outputs')
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import GRPOConfig
    started = time.monotonic(); set_seed(config['seed']); torch.set_num_threads(8)
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 2**30: raise RuntimeError('Need at least 9 GiB free GPU memory')
    torch.cuda.reset_peak_memory_stats()
    run = {'arguments': vars(args), 'recipe': recipe, 'data_manifest': manifest, 'base_manifest': base,
           'parent_adapter_sha256': parent_hash, 'gpu': torch.cuda.get_device_name(),
           'source_hashes': {name: sha256(name) for name in ('rlvr/train.py','rlvr/data.py','rlvr/rewards.py')},
           'versions': {n: importlib.metadata.version(n) for n in ['torch','transformers','peft','trl','bitsandbytes']}}
    (output / 'run-config.json').write_text(json.dumps(run, indent=2)+'\n')
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, device_map={'':0},
        dtype=torch.bfloat16, attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':False})
    model = PeftModel.from_pretrained(model,parent,is_trainable=True,local_files_only=True)
    initial_digest = adapter_digest(model,'default')
    initial = {n:v.detach().float().cpu().clone() for n,v in model.named_parameters() if v.requires_grad}
    if any(v.requires_grad and 'lora_' not in n for n,v in model.named_parameters()):
        raise RuntimeError('Unexpected trainable base weights')
    eos_ids = [tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]
    model.generation_config.eos_token_id = eos_ids
    group_stats = {'groups':0,'mixed_reward_groups':0,'correct_rollouts':0,'rollouts':0,'truncated_rollouts':0}

    def math_reward(completions, answer, completion_ids, id, **kwargs):
        rewards=[]
        with (output/'rollouts.jsonl').open('a') as handle:
            for completion,gold,ids,row_id in zip(completions,answer,completion_ids,id,strict=True):
                text = tok.decode(ids, skip_special_tokens=False)
                truncated = not any(t in eos_ids for t in ids)
                reward = 0.0 if truncated else verify_answer(text,gold,utils)
                rewards.append(reward)
                group_stats['truncated_rollouts'] += int(truncated)
                handle.write(json.dumps({'id':row_id,'answer':gold,'completion':text,'tokens':len(ids),
                                         'truncated':truncated,'reward':reward})+'\n')
        for start in range(0,len(rewards),config['num_generations']):
            group=rewards[start:start+config['num_generations']]
            group_stats['groups']+=1
            group_stats['mixed_reward_groups']+=int(len(set(group))>1)
        group_stats['rollouts']+=len(rewards);group_stats['correct_rollouts']+=sum(rewards)
        return rewards

    class CheckFinite(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if state.global_step >= 8 and group_stats['mixed_reward_groups'] == 0:
                print('Stopping pilot: no within-group reward variation after eight updates.', flush=True)
                control.should_training_stop = True
            if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise RuntimeError(f'Nonfinite RLVR metric: {logs}')

    train_args = GRPOConfig(output_dir=str(output),num_train_epochs=1,max_steps=args.max_steps,
        per_device_train_batch_size=1,gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_generations=config['num_generations'],max_completion_length=args.max_completion_length,
        learning_rate=config['learning_rate'],lr_scheduler_type='constant',warmup_steps=0,weight_decay=0.0,
        beta=0.0,loss_type='grpo',temperature=1.0,top_p=1.0,top_k=0,mask_truncated_completions=True,
        generation_kwargs={'eos_token_id':eos_ids},bf16=True,tf32=True,optim='adamw_torch',
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_steps=8,save_total_limit=2,logging_steps=1,logging_nan_inf_filter=False,
        disable_tqdm=True,report_to='none',seed=config['seed'],data_seed=config['seed'])
    trainer = trainer_class()(model=model,args=train_args,processing_class=tok,
                              train_dataset=Dataset.from_list(splits['train']),reward_funcs=math_reward,
                              callbacks=[CheckFinite()])
    if set(trainer.eos_token_ids) != set(eos_ids):
        raise RuntimeError('Generation and truncation masking must recognize both stop tokens')
    for name,value in initial.items(): model.get_parameter(name).data=value.to(model.get_parameter(name).device)
    del initial
    if adapter_digest(model,'default')!=initial_digest: raise RuntimeError('Policy initialization changed')
    (output/'training-args.json').write_text(train_args.to_json_string())

    def evaluate(label):
        rows=[];model.eval();model.config.use_cache=True
        with torch.no_grad():
            for case in splits['eval'][:args.eval_size]:
                batch=tok.apply_chat_template(case['prompt'],tokenize=True,add_generation_prompt=True,
                                               return_dict=True,return_tensors='pt').to('cuda')
                generated=model.generate(**batch,do_sample=False,max_new_tokens=args.max_completion_length,
                                         eos_token_id=eos_ids,pad_token_id=tok.pad_token_id,use_cache=True)
                ids=generated[0,batch['input_ids'].shape[1]:].tolist()
                text=tok.decode(ids,skip_special_tokens=False)
                truncated=not any(t in eos_ids for t in ids)
                reward=0.0 if truncated else verify_answer(text,case['answer'],utils)
                rows.append({'id':case['id'],'completion':text,'answer':case['answer'],'correct':bool(reward),
                             'tokens':len(ids),'truncated':truncated})
                print(f'{label}: {len(rows)}/{args.eval_size}',flush=True)
        (output/f'{label}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        model.train();model.config.use_cache=False
        return {'examples':len(rows),'correct':sum(r['correct'] for r in rows),
                'truncated':sum(r['truncated'] for r in rows)}

    before=evaluate('before')
    result=trainer.train()
    after=evaluate('after')
    final=output/'final-adapter';model.save_pretrained(final);tok.save_pretrained(final);trainer.save_state()
    if sha256(parent/'adapter_model.safetensors')!=parent_hash: raise RuntimeError('DPO parent changed')
    metrics={**result.metrics,'before':before,'after':after,**group_stats,'optimizer_steps':trainer.state.global_step,
             'policy_changed':adapter_digest(model,'default')!=initial_digest,
             'stopped_for_no_learning_signal':trainer.state.global_step>=8 and group_stats['mixed_reward_groups']==0,
             'parent_unchanged':True,'final_adapter_sha256':sha256(final/'adapter_model.safetensors'),
             'process_runtime_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    (output/'all_results.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
