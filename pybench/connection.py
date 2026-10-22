"""Discover models and verify a small generation before saving a connection.

Candidates are only guesses until a valid generation response is received.
Connection probes never enter benchmark samples or timing windows.
"""
import asyncio
import json
from urllib.parse import urlsplit

import aiohttp

from pybench.protocols import protocol_for
from pybench.results import utc_now
from pybench.runner import measure_request


def candidates(url, protocol=None):
    base = url.rstrip('/')
    path = urlsplit(base).path
    if protocol:
        return [(base, protocol)]  # Advanced override means an exact endpoint.
    if path.endswith('/chat/completions'):
        return [(base, 'openai')]
    if path.endswith('/messages'):
        return [(base, 'anthropic')]
    order = ['anthropic', 'openai'] if protocol_for(base) == 'anthropic' else ['openai', 'anthropic']
    result = []
    if path and not path.endswith('/v1'):
        result.extend((base, kind) for kind in order)
    prefix = base if path.endswith('/v1') else base + '/v1'
    for kind in order:
        suffix = '/messages' if kind == 'anthropic' else '/chat/completions'
        result.append((prefix + suffix, kind))
        if path and not path.endswith('/v1'):
            result.append((base + suffix, kind))
    return list(dict.fromkeys(result))


def model_urls(url):
    base = url.rstrip('/')
    for suffix in ('/chat/completions', '/messages', '/models'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return list(dict.fromkeys([base + '/models' if base.endswith('/v1') else base + '/v1/models', base + '/models']))


async def discover_models(session, url, key):
    # Both conventions are used by model-list endpoints; stay on the supplied origin.
    headers = {'anthropic-version': '2023-06-01'}
    if key:
        headers.update({'Authorization': f'Bearer {key}', 'x-api-key': key})
    for candidate in model_urls(url):
        try:
            async with session.get(candidate, headers=headers, allow_redirects=False,
                                   timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    continue
                raw = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) > 1_000_000:
                        break
                if len(raw) > 1_000_000:
                    continue
                body = json.loads(raw)
                rows = body.get('data', body.get('models', [])) if isinstance(body, dict) else []
                ids = [row['id'] for row in rows if isinstance(row, dict)
                       and isinstance(row.get('id'), str) and 0 < len(row['id']) <= 200]
                if ids:
                    return list(dict.fromkeys(ids))[:500]
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
            continue
    return []


async def connect(url, key=None, model=None, protocol=None):
    """Return connected details, model choices, or a safe actionable error."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as session:
        if not model:
            models = await discover_models(session, url, key)
            if len(models) != 1:
                return {'status': 'needs_model', 'models': models,
                        'message': '请选择要测试的模型。' if models else '服务未提供可用模型列表，请填写部署时的模型名称后重试。'}
            model = models[0]
        attempts = []
        deadline = asyncio.get_running_loop().time() + 60
        for endpoint, kind in candidates(url, protocol):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            target = {'base_url': url, 'endpoint_url': endpoint, 'protocol': kind,
                      'model': model, 'api_key': key}
            try:
                sample = await asyncio.wait_for(measure_request(
                    session, {'target': target, 'run': {'max_tokens': 8}, 'price': None},
                    {'index': 0, 'messages': [{'role': 'user', 'content': 'Reply OK.'}]},
                    1, asyncio.get_running_loop().time(), None), min(12, remaining))
            except (asyncio.TimeoutError, AttributeError, TypeError, KeyError):
                sample = {'status': 'error', 'http_status': None}
            status = sample.get('http_status')
            attempts.append({'url': endpoint, 'protocol': kind, 'http_status': status})
            if sample['status'] == 'ok':
                return {'status': 'connected', 'model': model, 'protocol': kind,
                        'endpoint_url': endpoint, 'checked_at': utc_now(), 'attempts': attempts}
            # Avoid repeated billable/rate-limited calls. Auth may differ by protocol,
            # so a 401/403 is retained while other candidates are still checked.
            if status in (402, 429) or (status and status >= 500):
                raise ValueError(f'连接失败：服务返回 HTTP {status}，请检查额度、限流或服务状态后重试。')
        codes = sorted({a['http_status'] for a in attempts if a['http_status'] is not None})
        if 401 in codes or 403 in codes:
            raise ValueError('连接失败：服务要求有效 API Key，或当前密钥没有访问权限。')
        detail = '、'.join(map(str, codes)) or '超时或网络不可达'
        raise ValueError(f'未找到可用的生成接口（{detail}）。请检查地址和模型名称；自定义接口可在高级设置中指定。')
