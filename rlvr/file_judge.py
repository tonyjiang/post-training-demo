"""Interactive judge queue: training waits for explicit, auditable external scores."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from dpo.data import sha256


class FileJudge:
    def __init__(self, recipe, output, timeout=3600):
        self.queue=Path(output)/'judge-queue'
        self.timeout=timeout
        self.identity='Codex interactive session'
        root=Path('third_party/open-instruct')
        for name,digest in recipe['upstream']['files'].items():
            if sha256(root/name)!=digest:raise ValueError(f'Upstream source mismatch: {name}')
        sys.path.insert(0,str(root.resolve()))
        from open_instruct.judge_utils import JUDGE_PROMPT_MAP
        self.template=JUDGE_PROMPT_MAP['quality']

    def metadata(self):
        return {'backend':'file','judge':self.identity,'rubric':'Ai2 general-quality',
                'queue':str(self.queue),'timeout_seconds':self.timeout,
                'batch_requests':True,
                'limitation':'Requires an active interactive agent; exact model version is not pinned'}

    def submit(self, query, answer):
        if not answer.strip():return None
        prompt=self.template.format(input=query,output=answer)
        digest=hashlib.sha256(prompt.encode()).hexdigest()
        self.queue.mkdir(parents=True,exist_ok=True)
        request=self.queue/f'{digest}.request.json'
        response=self.queue/f'{digest}.response.json'
        payload={'request_sha256':digest,'judge':self.identity,'prompt':prompt,
                 'query':query,'answer':answer}
        if request.exists():
            if json.loads(request.read_text())!=payload:raise ValueError('Judge request mismatch')
        else:
            temporary=request.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');temporary.replace(request)
        return digest,response

    def collect(self, pending):
        if pending is None:
            return {'reward':0.0,'reasoning':'No completed final answer','judgment':None}
        digest,response=pending
        print(f'Awaiting interactive quality judgment: {digest}',flush=True)
        deadline=time.monotonic()+self.timeout
        while not response.exists():
            if time.monotonic()>=deadline:
                raise TimeoutError('Interactive judge timed out; refusing to substitute zero reward')
            time.sleep(1)
        result=json.loads(response.read_text())
        if result.get('request_sha256')!=digest or result.get('judge')!=self.identity:
            raise ValueError('Judge response identity or request hash mismatch')
        value=result.get('score')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 1<=value<=10:
            raise ValueError('Judge score must be a finite number in 1..10')
        if not isinstance(result.get('reasoning'),str) or not result['reasoning'].strip():
            raise ValueError('Judge response needs a written rationale')
        return {'reward':value/10,'reasoning':result['reasoning'],'judgment':result,
                'judge_request_sha256':digest}

    def score(self, query, answer):
        return self.collect(self.submit(query, answer))

    def score_many(self, pairs):
        # Publish the complete group before blocking, so one agent response can grade it.
        pending=[self.submit(query, answer) for query,answer in pairs]
        return [self.collect(item) for item in pending]
