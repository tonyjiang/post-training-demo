"""Hardware smoke test only: synthetic tokens, not the Dolci SFT recipe."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def adapter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if 'lora_' in name:
            digest.update(name.encode())
            digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/models/Olmo-3-1025-7B')
    parser.add_argument('--output', default='outputs/nf4-smoke')
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    report = {'versions': {p: importlib.metadata.version(p) for p in
                          ['torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate']},
              'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
              'sequence_length': args.seq_len, 'batch_size': 1, 'synthetic_data': True,
              'revision': json.loads((Path(args.model) / 'download-manifest.json').read_text())['revision']}
    print(json.dumps(report), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert quantized and all(m.weight.quant_state.quant_type == 'nf4' for m in quantized)
    assert all(p.device.type == 'cuda' for p in model.parameters()), 'Unexpected CPU offloading'
    report['nf4_linear_layers'] = len(quantized)
    report['loaded_allocated_gib'] = torch.cuda.memory_allocated() / 2**30
    print(f'Loaded {len(quantized)} NF4 layers, {report["loaded_allocated_gib"]:.2f} GiB allocated', flush=True)
    if args.reload:
        # Match training preparation: casts unquantized parameters to FP32.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, output / 'adapter', is_trainable=False)
        previous = json.loads((output / 'train-report.json').read_text())
        assert adapter_digest(model) == previous['adapter_sha256'], 'Adapter reload mismatch'
        model.eval()
        batch = tokenizer('The capital of France is', return_tensors='pt').to('cuda')
        with torch.no_grad():
            result = model.generate(**batch, max_new_tokens=16, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        report['generated_text'] = tokenizer.decode(result[0], skip_special_tokens=True)
        report['adapter_reload_verified'] = True
    else:
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                               target_modules='all-linear', task_type='CAUSAL_LM'))
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
        report['trainable_parameters'] = sum(p.numel() for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4)
        text = 'Question: What is two plus two? Answer: Two plus two equals four. '
        ids = tokenizer(text * args.seq_len, add_special_tokens=False, return_tensors='pt')['input_ids']
        ids = ids[:, :args.seq_len].to('cuda')
        labels = ids.clone()
        labels[:, :args.seq_len // 4] = -100
        before = adapter_digest(model)
        report['losses'] = []
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            assert norm > 0, 'No adapter gradient'
            optimizer.step()
            report['losses'].append(loss.item())
            print(f'Step {step + 1}: loss={loss.item():.5f}, gradient_norm={norm.item():.5f}', flush=True)
        report['adapter_sha256'] = adapter_digest(model)
        assert before != report['adapter_sha256'], 'Adapter weights did not change'
        model.save_pretrained(output / 'adapter')
        tokenizer.save_pretrained(output / 'adapter')
    torch.cuda.synchronize()
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    report['status'] = 'passed'
    (output / ('reload-report.json' if args.reload else 'train-report.json')).write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
