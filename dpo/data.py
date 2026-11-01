"""Preference validation and tokenization, shared by preparation and training."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preference(row):
    """Keep the shared prompt and only the final assistant turn of each branch."""
    chosen, rejected = row['chosen'], row['rejected']
    if len(chosen) < 2 or len(rejected) < 2 or chosen[:-1] != rejected[:-1]:
        raise ValueError('Chosen and rejected must have the same nonempty conversation prefix')
    if any(branch[-1]['role'] != 'assistant' for branch in (chosen, rejected)):
        raise ValueError('Both branches must end in an assistant response')
    if any(not isinstance(branch[-1]['content'], str) or not branch[-1]['content'].strip()
           for branch in (chosen, rejected)):
        raise ValueError('Empty assistant response')
    if chosen[-1] == rejected[-1]:
        raise ValueError('Identical responses')
    prompt = chosen[:-1]
    prompt_hash = hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'id': str(row['id']), 'dataset_source': row['dataset_source'],
            'prompt_hash': prompt_hash, 'prompt': prompt,
            'chosen': chosen[-1:], 'rejected': rejected[-1:]}


def tokenize_pair(tokenizer, record, max_length):
    """Require an exact prompt prefix; truncate right without inventing EOS tokens."""
    prompt = tokenizer.apply_chat_template(record['prompt'], tokenize=True, add_generation_prompt=True,
                                           return_dict=True)['input_ids']
    if not prompt or len(prompt) >= max_length:
        raise ValueError('Prompt leaves no completion tokens')
    output = {'prompt_ids': prompt}
    lengths = {}
    for side in ('chosen', 'rejected'):
        full = tokenizer.apply_chat_template(record['prompt'] + record[side], tokenize=True,
                                             add_generation_prompt=False, return_dict=True)['input_ids']
        if full[:len(prompt)] != prompt:
            raise ValueError('Chat template does not preserve the exact generation prompt prefix')
        completion = full[len(prompt):max_length]
        if not completion:
            raise ValueError('Empty retained completion')
        output[f'{side}_ids'] = completion
        lengths[side] = {'original': len(full), 'retained': min(len(full), max_length)}
    if output['chosen_ids'] == output['rejected_ids']:
        raise ValueError('Responses become identical after truncation')
    return output, lengths


def validate_tokens(record, max_length):
    for key in ('prompt_ids', 'chosen_ids', 'rejected_ids'):
        ids = record[key]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f'Invalid {key}')
    if any(len(record['prompt_ids']) + len(record[f'{side}_ids']) > max_length
           for side in ('chosen', 'rejected')):
        raise ValueError('Pair exceeds configured context length')
    if record['chosen_ids'] == record['rejected_ids']:
        raise ValueError('Identical retained completions')


def load_splits(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    splits, seen_ids, seen_prompts = {}, set(), set()
    for split in ('train', 'eval'):
        path = directory / f'{split}.jsonl'
        if sha256(path) != manifest['splits'][split]['sha256']:
            raise ValueError(f'{split} hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows or len(rows) != manifest['splits'][split]['rows']:
            raise ValueError(f'{split} count mismatch')
        for row in rows:
            validate_tokens(row, manifest['max_length'])
            if row['id'] in seen_ids or row['prompt_hash'] in seen_prompts:
                raise ValueError('Duplicate ID or prompt across prepared splits')
            seen_ids.add(row['id'])
            seen_prompts.add(row['prompt_hash'])
        splits[split] = rows
    return manifest, splits
