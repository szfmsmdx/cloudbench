"""Resolve endpoint paths and adapt the request to its API protocol."""
import json
from urllib.parse import urlsplit


def protocol_for(base_url):
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip('/')
    if path.endswith('/chat/completions'):
        return 'openai'
    return 'anthropic' if (path.endswith('/messages') or 'anthropic' in path.split('/')
                           or parsed.hostname == 'api.anthropic.com') else 'openai'


def endpoint_url(base_url):
    base = base_url.rstrip('/')
    path = urlsplit(base).path
    if path.endswith(('/chat/completions', '/messages')):
        return base
    suffix = '/messages' if protocol_for(base) == 'anthropic' else '/chat/completions'
    return base + (suffix if path.endswith('/v1') else '/v1' + suffix)


def target_protocol(target):
    return target.get('protocol') or protocol_for(target['base_url'])


def target_endpoint(target):
    return target.get('endpoint_url') or endpoint_url(target['base_url'])


def request_body(target, run, item):
    body = {'model': target['model'], 'messages': item['messages'], 'max_tokens': run['max_tokens'], 'stream': True}
    if target_protocol(target) == 'openai':
        if item.get('tools'):
            body['tools'] = item['tools']
        if run.get('include_usage', True):
            body['stream_options'] = {'include_usage': True}
        return body
    messages, system = [], []
    for original in item['messages']:
        role = original['role']
        content = original.get('content') or ''
        if role in {'system', 'developer'}:
            if messages:
                raise ValueError('Anthropic 接口无法无损回放插在会话中间的 system 消息')
            if not isinstance(content, str):
                raise ValueError('Anthropic system 消息目前仅支持文本')
            system.append(content)
            continue
        if original.get('reasoning_content'):
            raise ValueError('轨迹含 reasoning_content，无法无损转成 Anthropic thinking 块；请用 OpenAI 接口回放该轨迹')
        blocks = content if isinstance(content, list) else ([{'type': 'text', 'text': content}] if content else [])
        if role == 'tool':
            role = 'user'
            blocks = [{'type': 'tool_result', 'tool_use_id': original['tool_call_id'], 'content': content}]
        elif role == 'assistant':
            blocks = list(blocks)
            for call in original.get('tool_calls') or []:
                function = call['function']
                arguments = function.get('arguments') or '{}'
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                blocks.append({'type': 'tool_use', 'id': call['id'], 'name': function['name'], 'input': arguments})
        if role not in {'user', 'assistant'} or not blocks:
            raise ValueError('Anthropic 接口不支持此消息角色或空消息')
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'].extend(blocks)
        else:
            messages.append({'role': role, 'content': blocks})
    body['messages'] = messages
    if system:
        body['system'] = '\n\n'.join(system)
    if item.get('tools'):
        body['tools'] = []
        for tool in item['tools']:
            function = tool['function']
            converted = {'name': function['name'], 'input_schema': function.get('parameters', {'type': 'object'})}
            if function.get('description'):
                converted['description'] = function['description']
            body['tools'].append(converted)
    return body


def anthropic_output(chunk):
    if chunk.get('type') == 'content_block_delta':
        delta = chunk.get('delta') or {}
        return bool(delta.get('text') or delta.get('thinking') or delta.get('partial_json'))
    if chunk.get('type') == 'content_block_start':
        block = chunk.get('content_block') or {}
        return bool(block.get('text') or block.get('thinking') or (block.get('type') == 'tool_use' and block.get('name')))
    return False
