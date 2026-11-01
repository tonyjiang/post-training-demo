"""Compare an adapter against saved evaluation results or an explicitly requested base run."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(text):
    try:
        result = Decimal(text.strip().replace(',', '').replace('$', ''))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def score_math(doc, answer):
    # Deliberately explicit: grade the last boxed number, not a number in the reasoning.
    boxes = re.findall(r'\\boxed\{([^{}]+)\}', answer)
    predicted = number(boxes[-1]) if boxes else None
    gold = number(doc['answer'].rsplit('####', 1)[1])
    assert gold is not None, 'Invalid GSM8K gold answer'
    return {'correct': predicted is not None and predicted == gold,
            'predicted_number': str(predicted) if predicted is not None else None,
            'gold_number': str(gold)}


def score_instructions(doc, answer, registry):
    checks = []
    for instruction_id, kwargs in zip(doc['instruction_id_list'], doc['kwargs'], strict=True):
        instruction = registry[instruction_id](instruction_id)
        instruction.build_description(**{k: v for k, v in kwargs.items() if v})
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=doc['prompt'])
        # Checker exceptions fail the run rather than silently count as model failures.
        checks.append(bool(answer.strip()) and bool(instruction.check_following(answer)))
    assert checks
    return {'correct': all(checks), 'instruction_checks': checks}


def load_checkers(config):
    upstream = Path('third_party/olmes')
    for name, digest in config['olmes']['files'].items():
        assert sha256(upstream / name) == digest, f'OLMES source mismatch: {name}'
    sys.path.insert(0, str(upstream.resolve()))
    from langdetect import DetectorFactory
    from oe_eval.dependencies.ifeval.instructions_registry import INSTRUCTION_DICT
    from oe_eval.models.output_processors import extract_think_answer
    DetectorFactory.seed = 0
    return INSTRUCTION_DICT, extract_think_answer


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['model'], row['task'])].append(row)
    summary = {}
    for (model, task), rows in groups.items():
        metrics = {'examples': len(rows), 'correct': sum(row['correct'] for row in rows),
                   'accuracy': sum(row['correct'] for row in rows) / len(rows),
                   'hit_token_limit': sum(row['hit_token_limit'] for row in rows),
                   'closed_think_tag': sum(row['closed_think_tag'] for row in rows),
                   'generated_tokens': sum(row['generated_tokens'] for row in rows)}
        if task == 'ifeval':
            checks = [check for row in rows for check in row['instruction_checks']]
            metrics['instruction_accuracy'] = sum(checks) / len(checks)
            metrics['instructions'] = len(checks)
        summary.setdefault(model, {})[task] = metrics
    return summary


def load_saved_results(directory, label, expected_run=None):
    directory = Path(directory)
    run = json.loads((directory / 'run-config.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['model'] == label]
    if label != 'base' and label != run['arguments'].get('adapter_name', 'sft'):
        raise ValueError(f'{directory}: {label!r} is not the recorded adapter label')
    expected_ids = {(task, i) for task, selection in run['data_manifest']['selection'].items()
                    for i in selection['selected_ids']}
    actual_ids = [(row['task'], row['id']) for row in rows]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError(f'{directory}: incomplete or duplicate {label} predictions')
    if summarize(rows).get(label) != results['scores'].get(label):
        raise ValueError(f'{directory}: saved scores do not match predictions')
    if (results['data_sha256'] != run['data_manifest']['cases_sha256'] or
            results['adapter_sha256'] != run['adapter_sha256']):
        raise ValueError(f'{directory}: result provenance mismatch')
    if expected_run is not None:
        for key in ('data_manifest', 'base_manifest', 'chat_template_sha256', 'versions', 'metrics'):
            if run[key] != expected_run[key]:
                raise ValueError(f'{directory}: incompatible {key}; rerun with matching evaluation settings')
        actual_config = run.get('evaluation_config', run['data_manifest']['config'])
        expected_config = expected_run.get('evaluation_config', expected_run['data_manifest']['config'])
        if actual_config != expected_config:
            raise ValueError(f'{directory}: incompatible evaluation_config; rerun with matching generation settings')
    source = {'directory': str(directory), 'label': label, 'adapter_sha256': run['adapter_sha256'],
              'files': {name: sha256(directory / name) for name in
                        ('run-config.json', 'results.json', 'predictions.jsonl')}}
    return run, rows, source


def compare_saved_results(baseline_dir, baseline_name, candidate_dir, candidate_name, output):
    if baseline_name == candidate_name:
        raise ValueError('Baseline and candidate labels must differ')
    candidate_run, candidate, candidate_source = load_saved_results(candidate_dir, candidate_name)
    _, baseline, baseline_source = load_saved_results(baseline_dir, baseline_name, candidate_run)
    prompts = lambda rows: {(r['task'], r['id']): r['prompt'] for r in rows}
    if prompts(baseline) != prompts(candidate):
        raise ValueError('Saved baseline and candidate prompts differ')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'scores': summarize(baseline + candidate), 'generation_performed': False,
              'data_sha256': candidate_run['data_manifest']['cases_sha256'],
              'baseline': baseline_source, 'candidate': candidate_source}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/eval-small')
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--adapter', default='outputs/sft/final-adapter')
    parser.add_argument('--adapter-name', default='adapter', help='Result label, e.g. sft or dpo')
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline-results', help='Reuse a completed evaluation directory; no baseline generation')
    baseline.add_argument('--compare-base', action='store_true', help='Generate base answers for a new comparison')
    baseline.add_argument('--adapter-only', action='store_true', help='Evaluate only the supplied adapter, e.g. to establish a new SFT baseline')
    parser.add_argument('--baseline-name', default='sft', help='Label to reuse from --baseline-results')
    parser.add_argument('--candidate-results', help='Reuse candidate results too; compare saved runs without loading a model')
    parser.add_argument('--output', default='outputs/evals')
    parser.add_argument('--max-new-tokens', type=int, help='Override the configured generation budget')
    args = parser.parse_args()
    if not args.adapter_name.strip() or args.adapter_name == 'base':
        parser.error("--adapter-name must be nonempty and different from base")
    if args.baseline_results and args.baseline_name == args.adapter_name:
        parser.error('Baseline and candidate labels must differ')
    if args.candidate_results:
        if not args.baseline_results:
            parser.error('--candidate-results requires --baseline-results')
        compare_saved_results(args.baseline_results, args.baseline_name,
                              args.candidate_results, args.adapter_name, args.output)
        return
    config = json.loads(Path('configs/eval-small.json').read_text())
    if args.max_new_tokens is not None:
        config['max_new_tokens'] = args.max_new_tokens
    if not 0 < config['max_new_tokens'] < config['max_context']:
        parser.error('Generation budget must be positive and smaller than total context')
    registry, extract = load_checkers(config)

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    started = time.monotonic()
    set_seed(config['seed'])
    torch.set_num_threads(8)
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    # Generation length does not change sampled cases. Preserve the prepared
    # manifest, and record the effective evaluation config separately.
    sampling_config = lambda c: {k: v for k, v in c.items() if k not in ('max_new_tokens', 'batch_size')}
    assert sampling_config(manifest['config']) == sampling_config(config)
    assert sha256(data / 'cases.jsonl') == manifest['cases_sha256']
    cases = [json.loads(line) for line in (data / 'cases.jsonl').read_text().splitlines()]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new --output')
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True, padding_side='left')
    tokenizer.chat_template = json.loads(Path('configs/sft-tokenizer-config.json').read_text())['chat_template']
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
                                            tokenize=False, add_generation_prompt=True) for case in cases]
    for prompt in prompts:
        assert len(tokenizer.encode(prompt, add_special_tokens=False)) + config['max_new_tokens'] <= config['max_context']
    free, _ = torch.cuda.mem_get_info()
    assert free >= 9 * 2**30, 'Need 9 GiB free GPU memory for evaluation'
    run = {'arguments': vars(args), 'data_manifest': manifest, 'evaluation_config': config,
           'runner_sha256': sha256(__file__),
           'base_manifest': json.loads((Path(args.model) / 'download-manifest.json').read_text()),
           'adapter_sha256': sha256(Path(args.adapter) / 'adapter_model.safetensors'),
           'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
           'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'bitsandbytes', 'nltk', 'spacy']},
           'comparison': ('Supplied adapter only.' if args.adapter_only else
                          'Saved baseline versus supplied adapter.' if args.baseline_results else
                          'Same chat prompts and frozen NF4 base; adapter disabled versus enabled. This is not an optimized few-shot base-model evaluation.'),
           'metrics': {'gsm8k': 'Exact numeric match of last boxed answer; stricter than OLMES flexible extraction',
                       'ifeval': 'Strict prompt and instruction accuracy using pinned OLMES checkers'},
           'limitations': f"Small subsets, greedy decoding, {config['max_new_tokens']} generated-token cap, quantization; not official benchmark scores. Exact SFT overlap checked; pretraining and fuzzy overlap unknown."}
    records = []
    if args.baseline_results:
        _, records, run['baseline'] = load_saved_results(args.baseline_results, args.baseline_name, run)
        if {(r['task'], r['id']): r['prompt'] for r in records} != {(c['task'], c['id']): c['prompt'] for c in cases}:
            raise ValueError('Saved baseline prompts differ from current cases')
        print(f'{args.baseline_name}: reused {len(records)} saved predictions; no generation', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run-config.json').write_text(json.dumps(run, indent=2) + '\n')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    assert all(p.device.type == 'cuda' for p in model.parameters())
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')]
    with (output / 'predictions.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        variants = ['base', args.adapter_name] if args.compare_base else [args.adapter_name]
        for variant in variants:
            set_seed(config['seed'])
            with model.disable_adapter() if variant == 'base' else nullcontext():
                for start in range(0, len(cases), config['batch_size']):
                    batch_cases = cases[start:start + config['batch_size']]
                    batch = tokenizer(prompts[start:start + len(batch_cases)], add_special_tokens=False,
                                      padding=True, return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        generated = model.generate(**batch, do_sample=False, num_beams=1,
                                                   max_new_tokens=config['max_new_tokens'],
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=stop_ids, use_cache=True)
                    for case, sequence in zip(batch_cases, generated, strict=True):
                        ids = sequence[batch['input_ids'].shape[1]:].tolist()
                        stop = next((i for i, token in enumerate(ids) if token in stop_ids), None)
                        hit_limit = stop is None and len(ids) >= config['max_new_tokens']
                        if stop is not None:
                            ids = ids[:stop]
                        raw = tokenizer.decode(ids, skip_special_tokens=False)
                        answer = extract({'continuation': raw})['continuation'].strip()
                        random.seed(config['seed'] + case['id'])
                        score = (score_math(case['doc'], answer) if case['task'] == 'gsm8k'
                                 else score_instructions(case['doc'], answer, registry))
                        record = {'model': variant, 'task': case['task'], 'id': case['id'],
                                  'prompt': case['prompt'], 'raw_output': raw, 'answer': answer,
                                  'generated_tokens': len(ids), 'hit_token_limit': hit_limit,
                                  'closed_think_tag': '</think>' in raw, **score}
                        records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
                    print(f'{variant}: {start + len(batch_cases)}/{len(cases)} examples; '
                          f'{time.monotonic() - started:.1f}s elapsed', flush=True)
    results = {'scores': summarize(records), 'runtime_seconds': time.monotonic() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'data_sha256': manifest['cases_sha256'], 'adapter_sha256': run['adapter_sha256']}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
