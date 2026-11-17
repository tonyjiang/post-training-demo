"""Integrity checks for the pinned Dolci experiments."""
import json
from pathlib import Path
from dpo.data import sha256


def load_splits(directory, source_config="configs/rlvr-data.json"):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['source_manifest'] != json.loads(Path(source_config).read_text()):
        raise ValueError('RLVR dataset differs from pinned source')
    seen_ids, seen_prompts, splits = set(), set(), {}
    for split in ('train', 'eval'):
        path = directory / f'{split}.jsonl'
        entry = manifest['splits'][split]
        if sha256(path) != entry['sha256']:
            raise ValueError(f'{split} hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != entry['rows'] or not rows:
            raise ValueError('Invalid split size')
        for row in rows:
            if row['id'] in seen_ids or row['prompt_hash'] in seen_prompts:
                raise ValueError('Duplicate ID or prompt across splits')
            if not row['answer'].strip() or row['prompt_tokens'] > manifest['max_prompt_length']:
                raise ValueError('Invalid answer or overlong prompt')
            seen_ids.add(row['id']); seen_prompts.add(row['prompt_hash'])
        splits[split] = rows
    return manifest, splits
