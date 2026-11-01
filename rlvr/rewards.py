"""Boxed final-answer verification using pinned Ai2 extraction and equivalence."""
import json
from pathlib import Path
import sys
from dpo.data import sha256


def load_math_utils():
    recipe = json.loads(Path('configs/rlvr.json').read_text())
    root = Path('third_party/open-instruct')
    for name, digest in recipe['upstream']['files'].items():
        if sha256(root / name) != digest: raise ValueError(f'Upstream source mismatch: {name}')
    sys.path.insert(0, str(root.resolve()))
    from open_instruct import math_utils
    # Fail early if symbolic checking lacks its parser, rather than during reward calculation.
    if not math_utils.is_equiv('1', '1'): raise RuntimeError('Math parser self-check failed')
    return math_utils


def verify_answer(text, answer, math_utils):
    # Never reward a box from an unfinished thinking trace.
    if '</think>' not in text: return 0.0
    text = text.rsplit('</think>', 1)[-1]
    box = math_utils.last_boxed_only_string(text)
    if box is None: return 0.0
    try:
        value = math_utils.remove_boxed(box)
    except AssertionError:
        return 0.0
    return float(math_utils.hendrycks_is_equiv(value, answer) or math_utils.is_equiv(value, answer))
