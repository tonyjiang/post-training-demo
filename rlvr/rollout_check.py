"""Bounded generation-only feasibility check; no quality scores or policy updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dpo.data import sha256
from .data import load_splits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/rlvr-wildchat.json')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--adapter',help='Override the parent adapter for a saved-policy check')
    p.add_argument('--case-ids',nargs='+',help='Exact prepared train/eval IDs to inspect')
    p.add_argument('--compare-brief-reasoning',action='store_true',help='Paired original versus brief-reasoning system instruction; generation only')
    p.add_argument('--compare-length',type=int,help='Paired smaller budget; reset the seed per case and check token prefixes')
    p.add_argument('--max-completion-length',type=int,default=1024)
    args=p.parse_args()
    recipe=json.loads(Path(args.config).read_text());exp=recipe['experiment']
    manifest,splits=load_splits(exp['data'],recipe['data_source_config'])
    cases=splits['eval']
    if args.case_ids:
        catalog={row['id']:row for rows in splits.values() for row in rows}
        if len(set(args.case_ids)) != len(args.case_ids) or any(i not in catalog for i in args.case_ids):
            p.error('Unknown or duplicate case IDs')
        cases=[catalog[i] for i in args.case_ids]
    if not 1 <= args.count <= len(cases):p.error('Invalid count')
    if args.compare_length and args.compare_brief_reasoning:p.error('Choose one paired comparison')
    if args.compare_length is not None and not 128 <= args.compare_length < args.max_completion_length:
        p.error('Comparison budget must be at least 128 and smaller than the maximum')
    if not 128 <= args.max_completion_length <= 4096:p.error('Invalid completion budget')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if any(x.name not in {'train.log','pid'} for x in output.iterdir()):p.error('Refusing to overwrite probe')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    set_seed(exp['seed']);torch.set_num_threads(8)
    if torch.cuda.mem_get_info()[0] < 9*2**30:raise RuntimeError('Need 9 GiB free GPU memory')
    base=Path('/root/models/Olmo-3-1025-7B');parent=Path(args.adapter or exp['init_adapter'])
    if json.loads((base/'download-manifest.json').read_text()) != json.loads(Path('configs/base-model.json').read_text()):
        raise ValueError('Base manifest mismatch')
    parent_hash=sha256(parent/'adapter_model.safetensors')
    checkpoint=None
    if parent.name.startswith('checkpoint-'):
        from .resume import validate_checkpoint
        dpo=Path(exp['init_adapter'])
        checkpoint=validate_checkpoint(parent,manifest,json.loads((base/'download-manifest.json').read_text()),
            sha256(dpo/'adapter_model.safetensors'),exp,-1)
    elif parent_hash != json.loads((parent.parent/'all_results.json').read_text())['final_adapter_sha256']:
        raise ValueError('Parent adapter mismatch')
    (output/'run-config.json').write_text(json.dumps({'arguments':vars(args),'recipe':recipe,
        'data_manifest':manifest,'parent_adapter_sha256':parent_hash,'checkpoint':checkpoint,'kind':'generation-only; no reward or training'},indent=2)+'\n')
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(Path(exp['data'])/'tokenizer',local_files_only=True)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != manifest['chat_template_sha256']:
        raise ValueError('Chat template mismatch')
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,device_map={'':0},dtype=torch.bfloat16,
        attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=PeftModel.from_pretrained(model,parent,is_trainable=False,local_files_only=True);model.eval()
    eos=[tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')];rows=[]
    budgets=[args.compare_length,args.max_completion_length] if args.compare_length else [args.max_completion_length]
    pairs=[]
    with torch.no_grad():
        for index,case in enumerate(cases[:args.count]):
            case_seed=(exp['seed']+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)) % 2**32
            generated_ids=[]
            conditions=['original','brief'] if args.compare_brief_reasoning else ['original']
            for condition in conditions:
                prompt=[dict(message) for message in case['prompt']]
                if condition=='brief':
                    instruction='Reason briefly. Avoid repeating your analysis or exploring unnecessary alternatives. Then provide the final answer promptly, following the requested format.'
                    if prompt[0]['role']=='system':prompt[0]['content']+='\n\n'+instruction
                    else:prompt.insert(0,{'role':'system','content':instruction})
                batch=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,
                                              return_dict=True,return_tensors='pt').to('cuda')
                for budget in budgets:
                    if args.compare_length or args.compare_brief_reasoning:set_seed(case_seed)
                    generated=model.generate(**batch,max_new_tokens=budget,do_sample=True,
                        temperature=1.0,top_p=1.0,top_k=0,eos_token_id=eos,pad_token_id=tok.pad_token_id,use_cache=True)
                    ids=generated[0,batch['input_ids'].shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=False)
                    generated_ids.append(ids)
                    row={'id':case['id'],'prompt':prompt,'condition':condition,'prompt_tokens':batch['input_ids'].shape[1],
                         'tokens':len(ids),'completion_limit':budget,
                         'truncated':not any(t in eos for t in ids),'closed_thinking':'</think>' in text,'completion':text}
                    if args.compare_length or args.compare_brief_reasoning:row.update(seed=case_seed,completion_ids=ids)
                    rows.append(row)
                    with (output/'predictions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(f"probe {index+1}/{args.count}, {condition}, budget {budget}: {len(ids)} tokens; truncated={row['truncated']}",flush=True)
            if args.compare_length:
                pairs.append({'id':case['id'],'seed':case_seed,
                              'prefix_matches':generated_ids[1][:len(generated_ids[0])]==generated_ids[0]})
    if sha256(parent/'adapter_model.safetensors') != parent_hash:raise RuntimeError('Parent changed')
    results={'kind':'generation-only; no reward or training','examples':len(rows),
        'truncated':sum(r['truncated'] for r in rows),'closed_thinking':sum(r['closed_thinking'] for r in rows),
        'tokens':[r['tokens'] for r in rows],
        'by_budget':{str(budget):{'examples':sum(r['completion_limit']==budget for r in rows),
            'truncated':sum(r['truncated'] for r in rows if r['completion_limit']==budget),
            'closed_thinking':sum(r['closed_thinking'] for r in rows if r['completion_limit']==budget)} for budget in budgets},
        'by_condition':{condition:{'examples':sum(r['condition']==condition for r in rows),
            'truncated':sum(r['truncated'] for r in rows if r['condition']==condition),
            'closed_thinking':sum(r['closed_thinking'] for r in rows if r['condition']==condition)} for condition in conditions},
        'paired_prefix_checks':pairs,'runtime_seconds':time.monotonic()-started,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'parent_unchanged':True}
    (output/'all_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':main()
