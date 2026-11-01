"""Prepare a seeded, disjoint Dolci DPO subset using the pinned SFT chat template."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .data import preference, sha256, tokenize_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dpo.json')
    parser.add_argument('--manifest', default='configs/dpo-data.json')
    parser.add_argument('--shards', default='data/dolci-dpo-shards')
    parser.add_argument('--tokenizer', default='outputs/sft/final-adapter')
    parser.add_argument('--output', default='data/dolci-dpo-small')
    parser.add_argument('--train-size', type=int)
    parser.add_argument('--eval-size', type=int)
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())['experiment']
    for name in ('train_size', 'eval_size', 'max_length', 'seed'):
        if getattr(args, name) is None:
            setattr(args, name, config[name])
    if args.train_size < 1 or args.eval_size < 1 or not 16 <= args.max_length <= 2048:
        parser.error('Positive split sizes and a context length between 16 and 2048 are required')
    output = Path(args.output)
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source = json.loads(Path(args.manifest).read_text())
    shards = Path(args.shards)
    if json.loads((shards / 'download-manifest.json').read_text()) != source:
        raise ValueError('Download manifest differs from pinned DPO data')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template_config = json.loads(Path('configs/sft-tokenizer-config.json').read_text())
    tokenizer.chat_template = template_config['chat_template']
    for token_id, token in template_config['added_tokens_decoder'].items():
        if tokenizer.convert_tokens_to_ids(token['content']) != int(token_id):
            raise ValueError('Tokenizer differs from pinned SFT token IDs')
    rng = random.Random(args.seed)
    tables, candidates = [], []
    for entry in source['files']:
        path = shards / entry['rfilename']
        if path.stat().st_size != entry['size'] or sha256(path) != entry['lfs']['sha256']:
            raise ValueError(f'Shard hash mismatch: {path}')
        table = pq.read_table(path, columns=['id', 'dataset_source', 'chosen', 'rejected'])
        candidates.extend((len(tables), index) for index in range(table.num_rows))
        tables.append(table)
    rng.shuffle(candidates)
    selected, rejected, seen_ids, seen_prompts = [], Counter(), set(), set()
    truncation = Counter()
    for table_index, index in candidates:
        # Slice before converting to Python; avoid materializing every conversation.
        row = tables[table_index].slice(index, 1).to_pylist()[0]
        try:
            record = preference(row)
            if record['id'] in seen_ids or record['prompt_hash'] in seen_prompts:
                raise ValueError('Duplicate ID or prompt')
            tokens, lengths = tokenize_pair(tokenizer, record, args.max_length)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        seen_ids.add(record['id'])
        seen_prompts.add(record['prompt_hash'])
        selected.append({key: record[key] for key in ('id', 'dataset_source', 'prompt_hash')} | tokens)
        for side in ('chosen', 'rejected'):
            truncation[side] += int(lengths[side]['original'] > args.max_length)
        if len(selected) == args.train_size + args.eval_size:
            break
    if len(selected) < args.train_size + args.eval_size:
        raise ValueError(f'Only {len(selected)} eligible pairs; exclusions: {dict(rejected)}')
    rng.shuffle(selected)
    manifest = {'dataset': source['repo_id'], 'revision': source['revision'], 'source_manifest': source,
                'seed': args.seed, 'max_length': args.max_length,
                'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                'template_source': json.loads(Path('configs/sft-template-source.json').read_text()),
                'sampling': 'Seed-shuffled rows; unique IDs and prompts across both splits',
                'truncation': 'Right-truncate each prompt+response; drop empty or identical retained responses',
                'truncated_branches': dict(truncation), 'excluded': dict(rejected), 'splits': {}}
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in [('train', selected[:args.train_size]), ('eval', selected[args.train_size:])]:
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        manifest['splits'][split] = {'rows': len(rows), 'sha256': sha256(path),
                                     'sources': dict(Counter(row['dataset_source'] for row in rows))}
    tokenizer.save_pretrained(output / 'tokenizer')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
