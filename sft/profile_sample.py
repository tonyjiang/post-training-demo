"""Measure complete sampled conversations before choosing a training context limit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default='data/dolci-all-sources')
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    args = parser.parse_args()
    folder = Path(args.sample)
    content = (folder / 'conversations.jsonl').read_bytes()
    manifest = json.loads((folder / 'sample-manifest.json').read_text())
    assert hashlib.sha256(content).hexdigest() == manifest['sha256']
    rows = [json.loads(line) for line in content.splitlines()]
    mapping = json.loads(Path('configs/sft-source-families.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.chat_template = json.loads(Path('configs/sft-tokenizer-config.json').read_text())['chat_template']
    lengths = defaultdict(list)
    for i in range(0, len(rows), 32):
        batch = rows[i:i+32]
        texts = [tokenizer.apply_chat_template(r['messages'], tokenize=False,
                    add_generation_prompt=False) for r in batch]
        encoded = tokenizer(texts, add_special_tokens=False, return_length=True)
        for row, length in zip(batch, encoded['length']):
            lengths[mapping[row['dataset_source']]].append(length)
        if i % 320 == 0:
            print(f'Profiled {min(i+32,len(rows))}/{len(rows)}', flush=True)
    result = {'sample_sha256': manifest['sha256'], 'families': {}}
    for family, values in sorted(lengths.items()):
        values.sort()
        result['families'][family] = {'rows': len(values), 'min': values[0],
            'median': values[len(values)//2], 'max': values[-1],
            'fits': {str(n): sum(v<=n for v in values) for n in (1024,2048,4096,8192,16384,32768)}}
    (folder / 'length-report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
