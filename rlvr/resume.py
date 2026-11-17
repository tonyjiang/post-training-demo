"""Validate the provenance and complete state needed to resume GRPO."""
import json
from pathlib import Path
from dpo.data import sha256


def consumed_prompt_ids(checkpoint, step, visited=None):
    """Recover only committed prompt groups across the preserved resume chain."""
    checkpoint=Path(checkpoint).resolve();visited=set() if visited is None else visited
    if checkpoint in visited:raise ValueError('Cyclic resume history')
    visited.add(checkpoint)
    run=checkpoint.parent;prior=json.loads((run/'run-config.json').read_text())
    resume=prior.get('resume_checkpoint');start=resume['step'] if resume else 0
    ids=consumed_prompt_ids(resume['path'],start,visited) if resume else []
    size=prior['recipe']['experiment']['num_generations'];count=(step-start)*size
    rows=[json.loads(line) for line in (run/'rollouts.jsonl').read_text().splitlines()]
    if count<0 or len(rows)<count:raise ValueError('Resume history lacks committed rollout groups')
    for offset in range(0,count,size):
        group={row['id'] for row in rows[offset:offset+size]}
        if len(group)!=1:raise ValueError('Resume history has mixed prompt groups')
        row_id=group.pop()
        if row_id in ids:raise ValueError('Resume history repeats a training prompt')
        ids.append(row_id)
    if len(ids)!=step:raise ValueError('Resume history does not match the prompt position')
    return ids


def restore_policy(model, checkpoint):
    """Load the saved policy before baseline evaluation and Trainer construction."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    weights=load_file(str(Path(checkpoint)/'adapter_model.safetensors'))
    expected=get_peft_model_state_dict(model,adapter_name='default')
    if weights.keys()!=expected.keys() or any(weights[k].shape!=expected[k].shape for k in weights):
        raise ValueError('Checkpoint policy adapter differs from the loaded adapter')
    set_peft_model_state_dict(model,weights,adapter_name='default')


def validate_checkpoint(path, manifest, base, parent_hash, config, max_steps, allow_generation_count_change=False):
    checkpoint=Path(path).resolve()
    required=['adapter_model.safetensors','adapter_config.json','trainer_state.json',
              'optimizer.pt','scheduler.pt','rng_state.pth']
    if any(not (checkpoint/name).is_file() for name in required):
        raise ValueError('Checkpoint lacks adapter, optimizer, scheduler, or RNG state')
    prior=json.loads((checkpoint.parent/'run-config.json').read_text())
    if prior['data_manifest']!=manifest or prior['base_manifest']!=base or prior['parent_adapter_sha256']!=parent_hash:
        raise ValueError('Resume data, base model, or DPO parent mismatch')
    previous=prior['recipe']['experiment']
    for key in ('seed','learning_rate','loss_type','beta','max_prompt_length'):
        if previous[key]!=config[key]:raise ValueError(f'Incompatible resume setting: {key}')
    step=json.loads((checkpoint/'trainer_state.json').read_text())['global_step']
    if not isinstance(step,int) or step<1 or (max_steps!=-1 and max_steps<=step):
        raise ValueError('Resume requires a later target optimizer step')
    change=None;consumed=None
    if any(previous[k]!=config[k] for k in ('num_generations','gradient_accumulation_steps')):
        if not allow_generation_count_change:raise ValueError('Incompatible resume generation count; explicit opt-in required')
        if any(c['num_generations']<2 or c['gradient_accumulation_steps']!=c['num_generations'] for c in (previous,config)):
            raise ValueError('Generation-count changes require one prompt per optimizer update')
        consumed=consumed_prompt_ids(checkpoint,step)
        change={'from':previous['num_generations'],'to':config['num_generations'],
                'prompt_position':step,'note':'Explicit recipe change; optimizer state retained, sampling trajectory changes'}
    return {'path':str(checkpoint),'step':step,'files':{name:sha256(checkpoint/name) for name in required},
            'prior_completion_length':prior['arguments']['max_completion_length'],
            'generation_count_change':change,'consumed_prompt_ids':consumed}
