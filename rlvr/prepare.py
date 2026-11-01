"""Sample disjoint, untruncated math prompts from the pinned Dolci shard."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from dpo.data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', default='data/dolci-rlvr-shards')
    p.add_argument('--output', default='data/dolci-rlvr-math')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists(): p.error(f'Refusing to overwrite {output}')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    config = json.loads(Path('configs/rlvr.json').read_text())['experiment']
    source = json.loads(Path('configs/rlvr-data.json').read_text())
    shard_dir = Path(args.shards)
    if json.loads((shard_dir / 'download-manifest.json').read_text()) != source:
        raise ValueError('Download manifest mismatch')
    tok = AutoTokenizer.from_pretrained(config['init_adapter'], local_files_only=True)
    tok.chat_template = json.loads(Path('configs/sft-tokenizer-config.json').read_text())['chat_template']
    rows = []
    for entry in source['files']:
        path = shard_dir / entry['rfilename']
        if sha256(path) != entry['lfs']['sha256']: raise ValueError('Shard hash mismatch')
        rows.extend(pq.read_table(path, columns=['custom_id', 'prompt', 'ground_truth', 'dataset',
                                               'original_dataset', 'passrate']).to_pylist())
    random.Random(config['seed']).shuffle(rows)
    seen, selected, excluded = set(), [], Counter()
    # Keep our existing benchmark prompts outside the new RL training data.
    benchmark = Path('data/eval-small/cases.jsonl')
    if not benchmark.exists(): raise ValueError('Prepare the fixed evaluation cases first')
    benchmark_rows = [json.loads(x) for x in benchmark.read_text().splitlines()]
    eval_questions = [re.sub(r'\s+', ' ', (r['doc'].get('question') or r['prompt'])).strip().casefold()
                      for r in benchmark_rows]
    for row in rows:
        if row['dataset'] != ['math'] or len(row['ground_truth']) != 1:
            excluded['not_math_single_answer'] += 1; continue
        text = row['prompt']
        if not text.startswith('user: ') or re.search(r'\n(?:user|assistant|system):', text):
            excluded['unsupported_conversation'] += 1; continue
        question = text[len('user: '):].strip()
        normalized = re.sub(r'\s+', ' ', question).casefold()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen or any(q in normalized or normalized in q for q in eval_questions):
            excluded['duplicate_or_benchmark_overlap'] += 1; continue
        prompt = [{'role': 'user', 'content': question + '\n\nShow your work and put your final answer in \\boxed{...}.'}]
        ids = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=True)['input_ids']
        if len(ids) > config['max_prompt_length']:
            excluded['overlong_prompt'] += 1; continue
        answer = str(row['ground_truth'][0]).strip()
        if not answer:
            excluded['empty_answer'] += 1; continue
        seen.add(digest)
        selected.append({'id': row['custom_id'], 'prompt_hash': digest, 'prompt': prompt,
                         'answer': answer, 'prompt_tokens': len(ids), 'source': row['original_dataset'],
                         'upstream_passrate': row['passrate']})
        if len(selected) == config['train_size'] + config['eval_size']: break
    if len(selected) != config['train_size'] + config['eval_size']:
        raise ValueError('Insufficient eligible math prompts')
    output.mkdir(parents=True)
    manifest = {'source_manifest': source, 'seed': config['seed'], 'max_prompt_length': config['max_prompt_length'],
                'chat_template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                'benchmark_cases_sha256': sha256(benchmark), 'excluded': dict(excluded),
                'sampling': 'Seed-shuffled first-shard math prompts; no passrate filtering; drop overlong prompts without truncation',
                'prompt_instruction': 'Append a work-and-boxed-final-answer instruction.', 'splits': {}}
    for split, subset in [('train', selected[:config['train_size']]), ('eval', selected[config['train_size']:])]:
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset))
        manifest['splits'][split] = {'rows': len(subset), 'sha256': sha256(path),
                                   'sources': dict(Counter(r['source'] for r in subset))}
    tok.save_pretrained(output / 'tokenizer')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
