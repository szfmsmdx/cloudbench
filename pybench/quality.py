"""Persisted quality evaluation API, separate from performance benchmark runs."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid

import aiohttp
from aiohttp import web

from pybench import gpqa
from pybench.protocols import request_body, target_endpoint, target_protocol
from pybench.results import utc_now
from pybench.runner import usage_from, estimate_cost


class Quality:
    def __init__(self, owner):
        self.owner = owner
        self.api = sys.modules[type(owner).__module__]
        self.db = owner.db
        self.tasks = {}
        self.db.execute('CREATE TABLE IF NOT EXISTS quality_runs (id TEXT PRIMARY KEY, data_json TEXT NOT NULL)')
        for row in self.db.execute('SELECT id,data_json FROM quality_runs').fetchall():
            run = json.loads(row['data_json'])
            if run['status'] == 'running':
                run.update(status='error', finished_at=utc_now(), error='服务重启，评测已中断；已完成题目保留')
                self.save(run)

    def save(self, run):
        self.db.execute('INSERT OR REPLACE INTO quality_runs VALUES (?,?)', (run['id'], json.dumps(run, ensure_ascii=False)))

    def read(self, ident):
        ApiError = self.api.ApiError
        row = self.db.execute('SELECT data_json FROM quality_runs WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise ApiError('评测不存在', 404)
        return json.loads(row['data_json'])

    def public(self, run, samples=True):
        result = {**run, 'summary': gpqa.summary(run)}
        if not samples:
            result.pop('samples')
        return result

    async def list_runs(self, request):
        json_response = self.api.json_response
        rows = self.db.execute('SELECT data_json FROM quality_runs ORDER BY rowid DESC LIMIT 30').fetchall()
        return json_response({'ready': (self.owner.root / 'datasets/gpqa-diamond.json').is_file(),
                              'runs': [self.public(json.loads(r['data_json']), False) for r in rows]})

    async def get(self, request):
        json_response = self.api.json_response
        return json_response(self.public(self.read(request.match_info['id'])))

    async def export(self, request):
        json_response = self.api.json_response
        response = json_response(self.public(self.read(request.match_info['id'])))
        response.headers['Content-Disposition'] = f'attachment; filename="gpqa-{request.match_info["id"]}.json"'
        return response

    async def create(self, request):
        ApiError, body_json, number, json_response = (self.api.ApiError, self.api.body_json,
                                                    self.api.number, self.api.json_response)
        body = await body_json(request)
        if self.tasks or self.owner.active or getattr(self.owner, 'preparing', False):
            raise ApiError('已有测试正在运行，请等待或停止', 409)
        target = self.owner.one('SELECT * FROM targets WHERE id=?', (body.get('target_id'),))
        if target is None:
            raise ApiError('请选择模型端点')
        if not self.owner.credentials.get(target['id']) and body.get('require_key'):
            raise ApiError('请先在模型端点页面设置 API Key')
        path = self.owner.root / 'datasets/gpqa-diamond.json'
        if not path.is_file():
            raise ApiError('GPQA 数据未准备，请运行 python -m pybench.gpqa')
        data = json.loads(path.read_text(encoding='utf-8'))
        count = number(body.get('count', 198), 'GPQA 全量题数', 198, 198, True)
        seed = number(body.get('seed', 42), '随机种子', 0, 4294967295, True)
        max_tokens = number(body.get('max_tokens', 8192), '输出上限', 128, 65536, True)
        timeout = number(body.get('timeout', 900), '单题超时', 10, 3600, True)
        concurrency = number(body.get('concurrency', 1), '并发数', 1, 64, True)
        chat_template_kwargs = body.get('chat_template_kwargs')
        if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
            raise ApiError('chat_template_kwargs 必须是 JSON 对象')
        price = self.owner.one('SELECT * FROM prices WHERE id=?', (body.get('price_id'),)) if body.get('price_id') else None
        if body.get('price_id') and price is None:
            raise ApiError('价格版本不存在')
        run = {'id': str(uuid.uuid4()), 'target_name': target['name'], 'model': target['model'],
               'protocol': target_protocol(target), 'dataset': 'GPQA-Diamond', 'method': gpqa.METHOD,
               'dataset_revision': data['revision'], 'dataset_sha256': data['csv_sha256'],
               'prompt_instruction': gpqa.INSTRUCTION, 'count': count, 'seed': seed, 'max_tokens': max_tokens,
               'timeout': timeout, 'price': price, 'status': 'running', 'started_at': utc_now(),
               'finished_at': None, 'samples': [], 'error': None, 'temperature': 'provider-default',
               'reasoning': 'provider-default', 'repeats': 1, 'chat_template_kwargs': chat_template_kwargs,
               'concurrency': concurrency}
        self.save(run)
        task = asyncio.create_task(self.execute(run, target, gpqa.questions(data, count, seed)))
        self.tasks[run['id']] = task
        return json_response(self.public(run), 201)

    async def execute(self, run, target, questions):
        key = self.owner.credentials.get(target['id'], '')
        anthropic = target_protocol(target) == 'anthropic'
        headers = {'Content-Type': 'application/json'}
        if anthropic:
            headers['anthropic-version'] = '2023-06-01'
            if key:
                headers['x-api-key'] = key
        elif key:
            headers['Authorization'] = f'Bearer {key}'
        semaphore = asyncio.Semaphore(max(1, int(run.get('concurrency') or 1)))

        async def solve(seq, question, session):
            async with semaphore:
                sample = {**question, 'seq': seq, 'response': '', 'status': 'error', 'correct': False,
                          'answer': None, 'estimated_cost': None, 'error': None}
                try:
                    payload = request_body(target, run,
                                           {'messages': [{'role': 'user', 'content': question['prompt']}]})
                    # Non-streaming generation keeps quality scoring independent of SSE chunks.
                    payload['stream'] = False
                    payload.pop('stream_options', None)
                    async with session.post(target_endpoint(target), json=payload, headers=headers,
                                            allow_redirects=False) as response:
                        if response.status >= 300:
                            raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:300]}')
                        body = await response.json()
                    if body.get('error'):
                        raise RuntimeError(str(body['error']))
                    if anthropic:
                        text = ''.join(b.get('text', '') for b in body.get('content', []) if b.get('type') == 'text')
                        reasoning = ''.join(b.get('thinking', '') for b in body.get('content', []) if b.get('type') == 'thinking')
                        finish = body.get('stop_reason')
                    else:
                        choice = (body.get('choices') or [{}])[0]
                        message = choice.get('message') or {}
                        text = message.get('content') or ''
                        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
                        finish = choice.get('finish_reason')
                    parsed = usage_from(body.get('usage'))
                    sample.update(response=text, reasoning=reasoning, finish_reason=finish,
                                  returned_model=body.get('model'), **parsed, **gpqa.score(text, question['expected'], finish))
                    sample['estimated_cost'] = estimate_cost(parsed, run['price'])
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RuntimeError, TypeError) as exc:
                    sample['error'] = (str(exc).replace(key, '[REDACTED]') if key else str(exc))[:400] or '请求超时'
                run['samples'].append(sample)
                self.save(run)

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run['timeout'])) as session:
                await asyncio.gather(*(solve(seq, question, session)
                                       for seq, question in enumerate(questions, 1)))
                run['samples'].sort(key=lambda item: item['seq'])
            run['status'] = 'completed'
        except asyncio.CancelledError:
            run['status'] = 'cancelled'
        except Exception:
            run.update(status='error', error='评测执行异常，已完成题目保留')
        finally:
            run['finished_at'] = utc_now()
            self.save(run)
            self.tasks.pop(run['id'], None)

    async def cancel(self, request):
        json_response = self.api.json_response
        self.read(request.match_info['id'])
        task = self.tasks.get(request.match_info['id'])
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return json_response({'ok': True})

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
