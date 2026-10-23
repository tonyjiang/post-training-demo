"""Sample complete conversations per Dolci family; keep old train/eval IDs excluded."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', default='data/dolci-source-shards')
    parser.add_argument('--exclude-data', default='data/previous-sft')
    parser.add_argument('--output', default='data/dolci-all-sources')
    parser.add_argument('--per-source', type=int, default=250)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error('--per-source must be positive')
    manifest = json.loads(Path('configs/sft-source-shards.json').read_text())
    mapping = json.loads(Path('configs/sft-source-families.json').read_text())
    shards = Path(args.shards)
    assert json.loads((shards / 'download-manifest.json').read_text()) == manifest
    excluded, exclusion_hashes = set(), {}
    for split in ('train', 'eval'):
        path = Path(args.exclude_data) / f'{split}.jsonl'
        excluded.update(json.loads(line)['id'] for line in path.read_text().splitlines())
        exclusion_hashes[split] = digest(path)
    rng = random.Random(args.seed)
    pools, seen_ids = defaultdict(list), set()
    available, observed_labels, invalid = Counter(), Counter(), Counter()
    for entry in manifest['files']:
        path = shards / entry['rfilename']
        assert digest(path) == entry['lfs']['sha256'], f'Hash mismatch: {path}'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                label = row['dataset_source']
                family = mapping[label]  # Fail on unknown labels; never silently lose a source.
                observed_labels[label] += 1
                if row['id'] in excluded or row['id'] in seen_ids:
                    continue
                seen_ids.add(row['id'])
                if not row['messages'] or row['messages'][-1]['role'] != 'assistant':
                    invalid[family] += 1
                    continue
                available[family] += 1
                # Reservoir sampling is uniform over eligible rows in the selected shards.
                pool = pools[family]
                if len(pool) < args.per_source:
                    pool.append(row)
                else:
                    index = rng.randrange(available[family])
                    if index < args.per_source:
                        pool[index] = row
        print(f'Scanned {entry["rfilename"]}: {dict(available)}', flush=True)
    missing = set(mapping.values()) - set(pools)
    if missing:
        raise ValueError(f'Missing source families: {sorted(missing)}')
    rows = [row for family in sorted(pools) for row in pools[family]]
    rng.shuffle(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    (output / 'conversations.jsonl').write_text(content)
    report = {'dataset': manifest['repo_id'], 'revision': manifest['revision'],
              'seed': args.seed, 'target_per_family': args.per_source,
              'sampling': 'reservoir per family over selected pinned shards; complete conversations; no length filter',
              'training_ready': False, 'rows': len(rows),
              'sha256': hashlib.sha256(content.encode()).hexdigest(),
              'excluded_ids': len(excluded), 'excluded_split_sha256': exclusion_hashes,
              'families': dict(sorted(Counter(mapping[r['dataset_source']] for r in rows).items())),
              'sources': dict(sorted(Counter(r['dataset_source'] for r in rows).items())),
              'eligible_in_selected_shards': dict(available), 'invalid_last_role': dict(invalid),
              'observed_labels': dict(observed_labels), 'input_manifest': manifest,
              'shortfalls': {k: args.per_source-len(v) for k,v in pools.items() if len(v)<args.per_source}}
    (output / 'sample-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('rows','families','shortfalls')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
