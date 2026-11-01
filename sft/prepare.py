"""Build a small, seeded, short-conversation Dolci subset from pinned shards."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def training_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    reference = json.loads(Path('configs/sft-tokenizer-config.json').read_text())
    for token_id, token in reference['added_tokens_decoder'].items():
        assert tokenizer.convert_tokens_to_ids(token['content']) == int(token_id)
    original = reference['chat_template']
    # Only annotate the existing assistant output branch; rendered text stays identical.
    start = "{% if message.get('content', none) is not none %}"
    end = "{% elif message['role'] == 'environment' %}"
    assert original.count(start) == original.count(end) == 1
    tokenizer.chat_template = original.replace(start, '{% generation %}' + start).replace(
        end, '{% endgeneration %}' + end)
    return tokenizer, original


def encode_record(tokenizer, row, max_length):
    encoded = tokenizer.apply_chat_template(
        row['messages'], tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True,
        truncation=True, max_length=max_length + 1)
    ids = encoded['input_ids']
    mask = encoded['assistant_masks']
    if len(ids) > max_length or not any(mask):
        return None
    labels = [token if learn else -100 for token, learn in zip(ids, mask)]
    return {'id': row['id'], 'dataset_source': row['dataset_source'],
            'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--shards', default='data/dolci-shards')
    parser.add_argument('--output', default='data/dolci-small')
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--max-chars', type=int, default=8192)
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--eval-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    tokenizer, original = training_tokenizer(args.model)
    manifest = json.loads(Path('configs/sft-data.json').read_text())
    downloaded = json.loads((Path(args.shards) / 'download-manifest.json').read_text())
    assert downloaded == manifest, 'Download manifest does not match pinned dataset'
    pools = defaultdict(list)
    seen_ids = set()
    checked_rendering = False
    inspected = eligible = 0
    for entry in manifest['files']:
        parquet = pq.ParquetFile(Path(args.shards) / entry['rfilename'])
        # Random candidate rows within each shard, rather than its first rows.
        indices = set(rng.sample(range(parquet.metadata.num_rows),
                                 min(5000, parquet.metadata.num_rows)))
        offset = 0
        for batch in parquet.iter_batches(batch_size=256):
            wanted = [i for i in range(batch.num_rows) if offset + i in indices]
            offset += batch.num_rows
            if not wanted:
                continue
            for row in batch.take(wanted).to_pylist():
                inspected += 1
                if row['id'] in seen_ids:
                    continue
                seen_ids.add(row['id'])
                if not row['messages'] or row['messages'][-1]['role'] != 'assistant':
                    continue
                # An explicit additional sampling constraint avoids encoding enormous traces.
                if sum(len(m.get('content') or '') for m in row['messages']) > args.max_chars:
                    continue
                if not checked_rendering:
                    assert tokenizer.apply_chat_template(row['messages'], tokenize=False) == \
                        tokenizer.apply_chat_template(row['messages'], tokenize=False, chat_template=original)
                    checked_rendering = True
                record = encode_record(tokenizer, row, args.max_length)
                if record:
                    eligible += 1
                    pools[row['dataset_source']].append(record)
        print(f"Scanned {entry['rfilename']}: {eligible} eligible examples so far", flush=True)
    for pool in pools.values():
        rng.shuffle(pool)
    # Round-robin source sampling intentionally favors coverage over original mix proportions.
    selected = []
    required = args.train_size + args.eval_size
    sources = sorted(pools)
    while len(selected) < required:
        progress = False
        for source in sources:
            if pools[source] and len(selected) < required:
                selected.append(pools[source].pop())
                progress = True
        if not progress:
            raise ValueError(f'Only {len(selected)} eligible records; need {required}')
    rng.shuffle(selected)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stats = {'dataset': manifest['repo_id'], 'revision': manifest['revision'],
             'template_source': json.loads(Path('configs/sft-template-source.json').read_text()),
             'seed': args.seed, 'max_length': args.max_length, 'max_chars': args.max_chars,
             'candidate_rows': inspected,
             'eligible_rows': eligible, 'sampling': '8 spaced shards, up to 5000 random rows per shard; '
             'raw content <= max_chars and complete conversations <= max_length; '
             'round-robin sources; not official proportions',
             'splits': {}}
    for name, rows in [('train', selected[:args.train_size]), ('eval', selected[args.train_size:])]:
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        (output / f'{name}.jsonl').write_text(content)
        stats['splits'][name] = {'rows': len(rows), 'sha256': hashlib.sha256(content.encode()).hexdigest(),
                                'sources': dict(Counter(row['dataset_source'] for row in rows))}
    tokenizer.save_pretrained(output / 'tokenizer')
    (output / 'manifest.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
