"""GPQA-Diamond generation evaluation; answers are scored locally, never by a judge."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

REVISION = '56686c06f5e19865c153de0fdb11be3890014df7'
SOURCE = f'https://raw.githubusercontent.com/idavidrein/gpqa/{REVISION}/dataset.zip'
METHOD = 'cloudbench-gpqa-diamond-v1'
INSTRUCTION = ('Solve the following multiple-choice question. You may reason before answering. '
               'End your response with exactly one final line: ANSWER: X, where X is A, B, C, or D.\n\n')


def prepare(destination: Path) -> dict:
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith('gpqa_diamond.csv'))
        raw = bundle.read(name, pwd=b'deserted-untie-orchid')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != 198:
        raise ValueError('GPQA-Diamond 应包含 198 道题')
    for row in rows:
        for key in ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
            if not row.get(key):
                raise ValueError(f'GPQA 缺少字段 {key}')
    data = {'revision': REVISION, 'source': SOURCE, 'archive_sha256': hashlib.sha256(archive).hexdigest(),
            'csv_sha256': hashlib.sha256(raw).hexdigest(), 'rows': rows}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return data


def questions(data: dict, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    indices = list(range(len(data['rows'])))
    rng.shuffle(indices)
    selected = []
    for index in indices[:count]:
        row = data['rows'][index]
        choices = [(row['Correct Answer'], True)] + [(row[f'Incorrect Answer {n}'], False) for n in range(1, 4)]
        rng.shuffle(choices)
        options = {letter: text for letter, (text, _) in zip('ABCD', choices)}
        expected = next(letter for letter, (_, correct) in zip('ABCD', choices) if correct)
        prompt = INSTRUCTION + row['Question'] + '\n\n' + '\n'.join(f'{k}. {v}' for k, v in options.items())
        selected.append({'source_index': index, 'question': row['Question'], 'options': options,
                         'expected': expected, 'prompt': prompt})
    return selected


def extract_answer(text: str) -> str | None:
    # Score only a committed final answer, never an arbitrary letter in reasoning.
    # Models often wrap the final line in Markdown emphasis, e.g. **ANSWER: C**.
    cleaned = re.sub(r'[*_`]+', '', text)
    matches = re.findall(r'^\s*ANSWER\s*:\s*\(?([A-D])\)?[.。]?\s*$', cleaned, re.I | re.M)
    return matches[-1].upper() if matches else None


def score(text: str, expected: str, finish_reason: str | None) -> dict:
    if finish_reason in {'length', 'max_tokens'}:
        return {'status': 'truncated', 'answer': extract_answer(text), 'correct': False}
    answer = extract_answer(text)
    return {'status': 'scored' if answer else 'unparsed', 'answer': answer, 'correct': answer == expected}


def summary(run: dict) -> dict:
    samples = run['samples']
    correct = sum(s.get('correct', False) for s in samples)
    n = len(samples)
    z = 1.96
    if n:
        p = correct / n
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        margin = z*((p*(1-p)/n + z*z/(4*n*n))**.5)/(1 + z*z/n)
        ci = [round(100*(center-margin), 2), round(100*(center+margin), 2)]
    else:
        ci = None
    costs = [s.get('estimated_cost') for s in samples]
    return {'completed': n, 'total': run['count'], 'correct': correct,
            'accuracy': round(100*correct/n, 2) if n else None, 'confidence_interval': ci,
            'errors': sum(s['status'] == 'error' for s in samples),
            'unparsed': sum(s['status'] == 'unparsed' for s in samples),
            'truncated': sum(s['status'] == 'truncated' for s in samples),
            'prompt_tokens': sum(s.get('prompt_tokens') or 0 for s in samples),
            'completion_tokens': sum(s.get('completion_tokens') or 0 for s in samples),
            'usage_coverage': sum(s.get('completion_tokens') is not None for s in samples),
            'estimated_cost': sum(costs) if costs and all(c is not None for c in costs) else None,
            'currency': (run.get('price') or {}).get('currency')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('datasets/gpqa-diamond.json'))
    args = parser.parse_args()
    data = prepare(args.output)
    print(f"Prepared {len(data['rows'])} GPQA-Diamond questions; CSV SHA256: {data['csv_sha256']}")
