"""Batch file-queue judging without changing reward meanings or rollout order."""
from .quality import final_answer


def training_rewards(records, judgments, num_generations):
    """Exclude cutoffs from GRPO baselines; groups need two finished answers."""
    if len(records) != len(judgments) or len(records) % num_generations:
        raise ValueError('Incomplete reward groups')
    rewards=[]
    for start in range(0,len(records),num_generations):
        group=records[start:start+num_generations]
        scores=judgments[start:start+num_generations]
        if sum(not r['truncated'] for r in group) < 2:
            # No relative comparison is possible. Equal rewards give zero
            # advantages without all-NaN/singleton reward statistics in TRL.
            rewards.extend([0.0]*num_generations)
        else:
            rewards.extend(None if r['truncated'] else s['reward']
                           for r,s in zip(group,scores,strict=True))
    return rewards


def score_records(records, judge, fallback):
    if judge is None or not hasattr(judge,'score_many'):
        return [fallback(r['text'],r['gold'],r['truncated'],r['query']) for r in records]
    results=judge.score_many([(r['query'],'' if r['truncated'] else final_answer(r['text'])) for r in records])
    if len(results)!=len(records):raise ValueError('Judge batch returned the wrong number of scores')
    return [({'reward':0.0,'reasoning':'Token-truncated completion'} if r['truncated'] else result)
            for r,result in zip(records,results,strict=True)]
