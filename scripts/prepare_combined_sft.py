"""Combine the original training split and expanded Dolci sample with right truncation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from prepare_sft import training_tokenizer


def read_verified(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f'Hash mismatch: {path}')
    return content, [json.loads(line) for line in content.splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--previous-data', default='data/dolci-small')
    parser.add_argument('--sample', default='data/dolci-all-sources')
    parser.add_argument('--output', default='data/dolci-combined-4k')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_length < 2 or args.max_length > 4096:
        parser.error('Choose a context length between 2 and our tested 4096-token ceiling.')
    output = Path(args.output)
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')
    previous = Path(args.previous_data)
    sample = Path(args.sample)
    prior_manifest = json.loads((previous / 'manifest.json').read_text())
    sample_manifest = json.loads((sample / 'sample-manifest.json').read_text())
    if (prior_manifest['dataset'], prior_manifest['revision']) != (
            sample_manifest['dataset'], sample_manifest['revision']):
        raise ValueError('Original and expanded data must use the same pinned dataset.')
    _, train = read_verified(previous / 'train.jsonl', prior_manifest['splits']['train']['sha256'])
    eval_content, evaluation = read_verified(previous / 'eval.jsonl', prior_manifest['splits']['eval']['sha256'])
    _, raw = read_verified(sample / 'conversations.jsonl', sample_manifest['sha256'])
    all_ids = [row['id'] for row in train + evaluation + raw]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Duplicate IDs or train/evaluation overlap')
    if len(raw) != sample_manifest['rows']:
        raise ValueError('Sample row count does not match its manifest')
    if any(len(row['input_ids']) > args.max_length for row in train + evaluation):
        raise ValueError('Context cap must accommodate the unchanged original splits')

    tokenizer, original = training_tokenizer(args.model)
    families = json.loads(Path('configs/sft-source-families.json').read_text())
    family_stats = {family: Counter() for family in sorted(set(families.values()))}
    truncated, dropped = [], []
    original_train_rows = len(train)
    for index, row in enumerate(raw, 1):
        stats = family_stats[families[row['dataset_source']]]
        stats['sampled'] += 1
        if not row['messages'] or row['messages'][-1]['role'] != 'assistant':
            raise ValueError(f"Sample does not end in an assistant message: {row['id']}")
        if index == 1:
            assert tokenizer.apply_chat_template(row['messages'], tokenize=False) == \
                tokenizer.apply_chat_template(row['messages'], tokenize=False, chat_template=original)
        # Mask the full conversation first, then slice IDs and masks together. This keeps
        # assistant masks correct when a generation span crosses the truncation boundary.
        encoded = tokenizer.apply_chat_template(row['messages'], tokenize=True,
            add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True,
            truncation=False)
        full_length = len(encoded['input_ids'])
        ids = encoded['input_ids'][:args.max_length]
        mask = encoded['assistant_masks'][:args.max_length]
        if len(mask) != len(ids):
            raise ValueError(f"Invalid assistant mask: {row['id']}")
        detail = {'id': row['id'], 'family': families[row['dataset_source']],
                  'original_tokens': full_length, 'retained_tokens': len(ids)}
        if full_length > args.max_length:
            stats['over_limit'] += 1
            truncated.append(detail)
        # A causal loss predicts tokens 1..N-1. A prompt-only prefix has no learning target.
        if not any(mask[1:]):
            stats['dropped_no_assistant_target'] += 1
            dropped.append({**detail, 'reason': 'No assistant prediction target remains after truncation'})
            continue
        stats['kept'] += 1
        stats['truncated_kept'] += int(full_length > args.max_length)
        train.append({'id': row['id'], 'dataset_source': row['dataset_source'],
            'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [token if learn else -100 for token, learn in zip(ids, mask)]})
        if index % 500 == 0:
            print(f'Encoded {index}/{len(raw)} expanded conversations', flush=True)

    # Mix complete records across sources; never reorder messages or tokens within a record.
    random.Random(args.seed).shuffle(train)
    output.mkdir(parents=True, exist_ok=False)
    train_content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in train).encode()
    manifest = {'dataset': sample_manifest['dataset'], 'revision': sample_manifest['revision'],
        'template_source': prior_manifest['template_source'], 'seed': args.seed,
        'max_length': args.max_length, 'training_ready': True,
        'required_training_options': {'liger_fused_loss': args.max_length > 2048},
        'sampling': 'Original train plus expanded sample; right truncation; seed-shuffled records; original eval unchanged',
        'truncation': 'Keep the first max_length tokens; preserve assistant masks; do not append EOS or invent a completion',
        'inputs': {'previous_manifest': prior_manifest, 'sample_sha256': sample_manifest['sha256'],
                   'sample_manifest_sha256': hashlib.sha256((sample / 'sample-manifest.json').read_bytes()).hexdigest()},
        'original_train_rows': original_train_rows, 'expanded_sample_rows': len(raw),
        'expanded_kept_rows': len(train) - original_train_rows, 'over_limit_rows': len(truncated),
        'dropped_no_assistant_target_rows': len(dropped), 'expanded_families': family_stats, 'splits': {}}
    for name, rows, content in [('train', train, train_content), ('eval', evaluation, eval_content)]:
        (output / f'{name}.jsonl').write_bytes(content)
        manifest['splits'][name] = {'rows': len(rows), 'sha256': hashlib.sha256(content).hexdigest(),
            'sources': dict(Counter(row['dataset_source'] for row in rows)),
            'families': dict(Counter(families[row['dataset_source']] for row in rows)),
            'tokens': sum(len(row['input_ids']) for row in rows),
            'supervised_tokens': sum(sum(label != -100 for label in row['labels'][1:]) for row in rows),
            'max_tokens': max(len(row['input_ids']) for row in rows)}
    tokenizer.save_pretrained(output / 'tokenizer')
    (output / 'truncation-report.json').write_text(json.dumps({'truncated': truncated, 'dropped': dropped}, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ['original_train_rows', 'expanded_kept_rows',
        'over_limit_rows', 'dropped_no_assistant_target_rows', 'expanded_families', 'splits']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
