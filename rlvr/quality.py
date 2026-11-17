"""Explicit OpenAI-compatible judge using Ai2's pinned general-quality rubric."""
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dpo.data import sha256


def final_answer(text):
    if '</think>' not in text:
        return ''
    text = text.rsplit('</think>', 1)[-1]
    return re.sub(r'<\|(?:im_end|endoftext)\|>', '', text).strip()


class QualityJudge:
    def __init__(self, recipe):
        settings = recipe['judge']
        self.settings = settings
        self.base_url = os.environ.get(settings['base_url_env'], '').rstrip('/')
        self.model = os.environ.get(settings['model_env'], '')
        self.api_key = os.environ.get(settings['api_key_env'], '')
        if not self.base_url or not self.model:
            raise ValueError(f"Configure {settings['base_url_env']} and {settings['model_env']} before quality training")
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Judge base URL must be HTTP(S) without embedded credentials, query or fragment')
        root = Path('third_party/open-instruct')
        for name, digest in recipe['upstream']['files'].items():
            if sha256(root / name) != digest:
                raise ValueError(f'Upstream source mismatch: {name}')
        sys.path.insert(0, str(root.resolve()))
        from open_instruct.judge_utils import JUDGE_PROMPT_MAP, EXTRACTOR_MAP
        self.template = JUDGE_PROMPT_MAP['quality']
        self.extractor = EXTRACTOR_MAP['quality']

    def metadata(self):
        return {'base_url': self.base_url, 'model': self.model,
                'rubric': 'Ai2 general-quality', 'temperature': self.settings['temperature']}

    def score(self, query, answer):
        if not answer.strip():
            return {'reward': 0.0, 'reasoning': 'No completed final answer', 'judgment': None}
        body = {'model': self.model, 'messages': [{'role': 'user', 'content':
                self.template.format(input=query, output=answer)}],
                'temperature': self.settings['temperature'], 'max_tokens': self.settings['max_tokens']}
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        failure = None
        for attempt in range(self.settings['max_attempts']):
            try:
                request = urllib.request.Request(self.base_url + '/chat/completions',
                          data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=self.settings['timeout_seconds']) as response:
                    payload = json.load(response)
                choice = payload['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Judge response did not finish normally')
                judgment = choice['message']['content']
                cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', judgment.strip())
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise ValueError('Judge JSON must be an object')
                if isinstance(parsed.get('SCORE'), bool):
                    raise ValueError('Boolean score')
                score = float(parsed['SCORE'])
                if not math.isfinite(score) or not 1 <= score <= 10:
                    raise ValueError('Judge score outside 1..10')
                # Upstream parser handles normalization; strict validation prevents its fallback-to-zero.
                reasoning, reward = self.extractor(judgment)
                if not math.isclose(reward, score / 10):
                    raise ValueError('Judge score normalization mismatch')
                return {'reward': reward, 'reasoning': reasoning, 'judgment': judgment,
                        'usage': payload.get('usage', {})}
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                # Do not log API keys, arbitrary response bodies, or credential-bearing error messages.
                failure = type(error).__name__
                if isinstance(error, urllib.error.HTTPError) and error.code in (400,401,403,404):
                    break
                if attempt + 1 < self.settings['max_attempts']:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f'Quality judge failed ({failure}); refusing to substitute zero reward')
