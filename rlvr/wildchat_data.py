"""Prepare the pinned WildChat English general-quality subset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from dpo.data import sha256


def parse_conversation(text):
    matches = list(re.finditer(r'(?m)^(system|user|assistant):(?:[ \t]|\n|$)', text))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError('Unrecognized conversation prefix')
    messages = []
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[match.end():end].strip()
        if not content:
            raise ValueError('Empty conversation turn')
        messages.append({'role': match.group(1), 'content': content})
    if messages[-1]['role'] != 'user':
        raise ValueError('Conversation does not end with a user query')
    conversational = [m for m in messages if m['role'] != 'system']
    if not conversational or any(m['role'] != ('user' if i%2==0 else 'assistant')
                                 for i,m in enumerate(conversational)):
        raise ValueError('Nonalternating conversation')
    if any(m['role']=='system' for m in messages[1:]):
        raise ValueError('System message after conversation start')
    return messages


def exceeds_word_budget(messages, maximum):
    """Conservative screening of numeric word-count mentions in user turns."""
    if maximum is None:return False
    pattern=r'(?i)\b(\d[\d,]*)\s*[-–—]?\s*words?\b'
    return any(int(n.replace(',',''))>maximum for m in messages if m['role']=='user'
               for n in re.findall(pattern,m['content']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/rlvr-wildchat.json')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    exp=config['experiment']; source=json.loads(Path(config['data_source_config']).read_text())
    root=Path(exp['shards']); output=Path(exp['data'])
    if output.exists():parser.error(f'Refusing to overwrite {output}')
    if json.loads((root/'download-manifest.json').read_text()) != source:
        raise ValueError('Source download manifest mismatch')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(exp['init_adapter'],local_files_only=True)
    tok.chat_template=json.loads(Path('configs/sft-tokenizer-config.json').read_text())['chat_template']
    raw=[]
    for entry in source['files']:
        path=root/entry['rfilename']
        if sha256(path)!=entry['lfs']['sha256']:raise ValueError('Shard hash mismatch')
        raw.extend(pq.read_table(path,columns=['original_dataset','prompt','ground_truth','custom_id','dataset'],
                   filters=[('original_dataset','=',config['dataset_filter'])]).to_pylist())
    if len(raw)!=6421:raise ValueError('Expected all 6421 WildChat English records')
    random.Random(exp['seed']).shuffle(raw)
    excluded=Counter(); selected=[]; ids=set(); prompts=set()
    benchmark=Path('data/eval-small/cases.jsonl')
    benchmark_prompts={re.sub(r'\s+',' ',json.loads(x)['prompt']).strip().casefold()
                       for x in benchmark.read_text().splitlines()}
    for row in raw:
        if row['dataset'] != ['general-quality']:
            excluded['wrong_reward_type']+=1;continue
        try:messages=parse_conversation(row['prompt'])
        except ValueError as error:
            excluded[str(error)]+=1;continue
        if exceeds_word_budget(messages,exp.get('max_requested_words')):
            excluded['overlong_requested_word_count']+=1;continue
        normalized=[{'role':m['role'],'content':re.sub(r'\s+',' ',m['content']).strip().casefold()}
                    for m in messages]
        digest=hashlib.sha256(json.dumps(normalized,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        row_id=str(row['custom_id'])
        if row_id in ids or digest in prompts:
            excluded['duplicate_id_or_conversation']+=1;continue
        if any(m['content'] in benchmark_prompts for m in normalized if m['role']=='user'):
            excluded['benchmark_overlap']+=1;continue
        prompt_tokens=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                   return_dict=True)['input_ids'])
        if prompt_tokens>exp['max_prompt_length']:
            excluded['overlong_prompt']+=1;continue
        refs=row['ground_truth']
        if len(refs)!=1 or not isinstance(refs[0],str) or not refs[0].strip():
            excluded['missing_reference']+=1;continue
        reference_tokens=len(tok.encode(refs[0],add_special_tokens=False))
        if exp.get('max_reference_tokens') is not None and reference_tokens>exp['max_reference_tokens']:
            excluded['overlong_reference']+=1;continue
        ids.add(row_id);prompts.add(digest)
        selected.append({'id':row_id,'prompt_hash':digest,'prompt':messages,'answer':refs[0],
                         'prompt_tokens':prompt_tokens,'source':row['original_dataset'],
                         'reward_type':'general-quality','judge_query':row['prompt'],
                         'reference_tokens':reference_tokens})
        if len(selected)==exp['train_size']+exp['eval_size']:break
    if len(selected)!=exp['train_size']+exp['eval_size']:raise ValueError('Insufficient eligible examples')
    output.mkdir(parents=True)
    manifest={'source_manifest':source,'seed':exp['seed'],'reward_type':'general-quality',
              'max_prompt_length':exp['max_prompt_length'],
              'length_filters':{k:exp.get(k) for k in ('max_requested_words','max_reference_tokens')},
              'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
              'benchmark_cases_sha256':sha256(benchmark),'excluded':dict(excluded),
              'sampling':'Seed-shuffled WildChat English; preserve valid conversation turns; drop malformed/overlong cases; optional numeric requested-word and reference-token filters recorded in length_filters',
              'splits':{}}
    for split,rows in [('train',selected[:exp['train_size']]),('eval',selected[exp['train_size']:])]:
        path=output/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifest['splits'][split]={'rows':len(rows),'sha256':sha256(path),
                                  'sources':dict(Counter(r['source'] for r in rows))}
    tok.save_pretrained(output/'tokenizer')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
