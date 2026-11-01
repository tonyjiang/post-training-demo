"""Probe full Dolci conversations with disposable adapter updates; never save weights."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from sft.prepare import training_tokenizer, encode_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    p.add_argument('--sample', default='data/dolci-all-sources')
    p.add_argument('--adapter', required=True)
    p.add_argument('--liger', action='store_true', help='Probe only Liger fused linear cross entropy')
    p.add_argument('--output', required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[2048,4096,8192,16384,32768,43008])
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * 2**30:
        raise RuntimeError('GPU is occupied; need at least 9 GiB free')
    folder = Path(args.sample)
    content = (folder/'conversations.jsonl').read_bytes()
    manifest = json.loads((folder/'sample-manifest.json').read_text())
    assert hashlib.sha256(content).hexdigest() == manifest['sha256']
    rows = [json.loads(line) for line in content.splitlines()]
    tokenizer, _ = training_tokenizer(args.model)
    # Compute exact lengths without truncation or assistant-mask overhead.
    lengths = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False, add_generation_prompt=False)
        lengths.append(len(ids))
        if len(lengths) % 500 == 0:
            print(f'Profiled {len(lengths)}/{len(rows)} complete conversations', flush=True)
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_olmo3
        apply_liger_kernel_to_olmo3(rope=False, rms_norm=False, swiglu=False,
                                   cross_entropy=False, fused_linear_cross_entropy=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        device_map={'':0}, dtype=torch.bfloat16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache=False
    model = prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    model = PeftModel.from_pretrained(model,args.adapter,is_trainable=True,local_files_only=True)
    model.train()
    parameters=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4)
    report={'liger_fused_loss': args.liger,
        'liger_version': importlib.metadata.version('liger-kernel') if args.liger else None,
        'adapter': args.adapter, 'adapter_sha256':hashlib.sha256(
        (Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'sample_sha256':manifest['sha256'], 'gpu':torch.cuda.get_device_name(),
        'scope':'Two optimizer steps per target on complete conversations; disposable in-memory adapter, no saved weights; not a full-run guarantee',
        'gradient_accumulation':1, 'results':[]}
    for target in args.lengths:
        candidates=sorted((n,i) for i,n in enumerate(lengths) if n<=target)
        if len(candidates)<2:
            raise ValueError(f'Not enough conversations below {target}')
        selected=candidates[-2:]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result={'target_tokens':target,'examples':[], 'status':'running'}
        report['results'].append(result)
        started=time.monotonic()
        try:
            for n,index in selected:
                row=rows[index]
                record=encode_record(tokenizer,row,target)
                assert record and len(record['input_ids'])==n
                batch={k:torch.tensor([record[k]],device='cuda') for k in ('input_ids','attention_mask','labels')}
                optimizer.zero_grad(set_to_none=True)
                result['phase']='forward'
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(**batch).loss
                assert torch.isfinite(loss)
                result['phase']='backward'
                loss.backward()
                result['phase']='optimizer'
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
                assert norm>0
                optimizer.step()
                result['examples'].append({'id':row['id'],'source':row['dataset_source'],
                    'tokens':n,'loss':loss.item(),'gradient_norm':norm.item()})
                del loss,batch
            torch.cuda.synchronize()
            result['status']='passed'
        except torch.cuda.OutOfMemoryError as exc:
            result['status']='out_of_memory'
            result['error']=str(exc)
        result.update(seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        if result['status']!='passed':
            break


if __name__=='__main__':
    main()
