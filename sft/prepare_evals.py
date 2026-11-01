"""Prepare pinned, seeded benchmark subsets, excluding exact SFT prompt matches."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eval-small.json')
    parser.add_argument('--sources', default='data/eval-sources')
    parser.add_argument('--training-data', default='data/dolci-combined-4k')
    parser.add_argument('--output', default='data/eval-small')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    if output.exists():
        parser.error(f'{output} already exists; choose a new output directory')
    sources = Path(args.sources)
    sources.mkdir(parents=True, exist_ok=True)
    train = Path(args.training_data)
    training_manifest = json.loads((train / 'manifest.json').read_text())
    assert sha256(train / 'train.jsonl') == training_manifest['splits']['train']['sha256']
    tokenizer = AutoTokenizer.from_pretrained(train / 'tokenizer', local_files_only=True)
    training_text = '\n\n'.join(
        normalize(tokenizer.decode(json.loads(line)['input_ids']))
        for line in (train / 'train.jsonl').read_text().splitlines())
    cases, selections = [], {}
    for source in config['datasets']:
        path = sources / source['local_file']
        if not path.exists():
            url = (f"https://huggingface.co/datasets/{source['repo_id']}/resolve/"
                   f"{source['revision']}/{source['file']}")
            subprocess.run(['curl', '-fL', '--retry', '3', '--connect-timeout', '20',
                            '-o', str(path), url], check=True)
        assert sha256(path) == source['sha256'], f'Hash mismatch: {path}'
        rows = (pq.read_table(path).to_pylist() if path.suffix == '.parquet'
                else [json.loads(line) for line in path.read_text().splitlines()])
        order = list(range(len(rows)))
        random.Random(config['seed']).shuffle(order)
        selected, excluded = [], []
        for index in order:
            row = rows[index]
            question = row['question'] if source['name'] == 'gsm8k' else row['prompt']
            if normalize(question) in training_text:
                excluded.append(index)
                continue
            prompt = (config['gsm8k_prompt'].format(question=question)
                      if source['name'] == 'gsm8k' else question)
            cases.append({'task': source['name'], 'id': index, 'prompt': prompt, 'doc': row})
            selected.append(index)
            if len(selected) == config['examples_per_task']:
                break
        assert len(selected) == config['examples_per_task']
        selections[source['name']] = {'selected_ids': selected, 'excluded_exact_matches': excluded}
    output.mkdir(parents=True)
    content = ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases)
    (output / 'cases.jsonl').write_text(content)
    manifest = {'config': config, 'cases_sha256': sha256(output / 'cases.jsonl'),
                'training_sha256': sha256(train / 'train.jsonl'), 'selection': selections,
                'overlap_check': 'Case-insensitive, whitespace-normalized prompt substring in retained SFT tokens; no fuzzy or pretraining decontamination'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'cases': len(cases), 'selection': selections}, indent=2))


if __name__ == '__main__':
    main()
