"""Inspect pending interactive judge requests or atomically submit scores from stdin."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',required=True)
    p.add_argument('--pending',action='store_true')
    p.add_argument('--brief',action='store_true',help='Omit the duplicated rubric prompt from pending requests')
    args=p.parse_args()
    queue=Path(args.queue)
    if args.pending:
        requests=[json.loads(path.read_text()) for path in sorted(queue.glob('*.request.json'))
                  if not path.with_name(path.name.replace('.request.json','.response.json')).exists()]
        if args.brief:
            requests=[{k:v for k,v in request.items() if k != 'prompt'} for request in requests]
        print(json.dumps(requests,ensure_ascii=False,indent=2));return
    responses=json.load(sys.stdin)
    # Validate the whole batch before delivering any scores.
    pending=[]
    for result in responses:
        digest=result['request_sha256']
        if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):raise ValueError('Invalid request hash')
        request=json.loads((queue/f'{digest}.request.json').read_text())
        if hashlib.sha256(request['prompt'].encode()).hexdigest()!=digest:raise ValueError('Request hash mismatch')
        if result.get('judge')!=request['judge']:raise ValueError('Judge identity mismatch')
        score=result['score']
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not 1<=score<=10:raise ValueError('Invalid score')
        if not isinstance(result.get('reasoning'),str) or not result['reasoning'].strip():raise ValueError('Missing rationale')
        path=queue/f'{digest}.response.json'
        if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Refusing to overwrite a different score')
        pending.append((path,result))
    for path,result in pending:
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    print(f'Delivered {len(pending)} judgments')


if __name__=='__main__':main()
