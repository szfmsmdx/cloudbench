import asyncio
import json
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from pybench.protocols import endpoint_url, protocol_for, request_body
from pybench.runner import measure_request, usage_from


class ProtocolTests(unittest.TestCase):
    def test_complete_urls_are_preserved_and_base_paths_match_protocol(self):
        cases = {
            'https://api.deepseek.com/anthropic': 'https://api.deepseek.com/anthropic/v1/messages',
            'https://api.deepseek.com/anthropic/v1': 'https://api.deepseek.com/anthropic/v1/messages',
            'https://api.deepseek.com/anthropic/v1/messages': 'https://api.deepseek.com/anthropic/v1/messages',
            'https://api.anthropic.com': 'https://api.anthropic.com/v1/messages',
            'https://api.deepseek.com': 'https://api.deepseek.com/v1/chat/completions',
            'https://host/custom/chat/completions': 'https://host/custom/chat/completions',
            'https://host/custom/messages': 'https://host/custom/messages',
        }
        for base, expected in cases.items():
            self.assertEqual(endpoint_url(base), expected)
        self.assertEqual(protocol_for('https://host/custom/chat/completions'), 'openai')

    def test_anthropic_tool_history_and_system_are_adapted_without_changing_original(self):
        item = {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'inspect'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'function': {'name': 'bash', 'arguments': '{"command":"ls"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'observation'}],
            'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'type': 'object'}}}]}
        before = json.dumps(item)
        body = request_body({'model': 'mock', 'base_url': 'https://host/anthropic'}, {'max_tokens': 128}, item)
        self.assertEqual(body['system'], 'system')
        self.assertEqual(body['messages'][1]['content'][0], {'type': 'tool_use', 'id': 'call-1', 'name': 'bash', 'input': {'command': 'ls'}})
        self.assertEqual(body['messages'][2]['content'][0], {'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'observation'})
        self.assertEqual(body['tools'][0]['input_schema'], {'type': 'object'})
        self.assertNotIn('stream_options', body)
        self.assertEqual(json.dumps(item), before)

    def test_anthropic_cache_accounting_uses_total_logical_input(self):
        usage = usage_from({'input_tokens': 40, 'cache_read_input_tokens': 60, 'cache_creation_input_tokens': 10, 'output_tokens': 8})
        self.assertEqual((usage['prompt_tokens'], usage['cached_tokens'], usage['uncached_tokens'], usage['cache_write_tokens']), (110, 60, 40, 10))


class AnthropicStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_auth_usage_first_output_and_truncation(self):
        mode = 'text'
        requests = []
        async def respond(request):
            self.assertEqual(request.headers['x-api-key'], 'test-secret')
            self.assertEqual(request.headers['anthropic-version'], '2023-06-01')
            body = await request.json()
            self.assertNotIn('stream_options', body)
            requests.append(body)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            async def send(value):
                await response.write(('data: ' + json.dumps(value) + '\n\n').encode())
            await send({'type': 'message_start', 'message': {'usage': {'input_tokens': 40, 'cache_read_input_tokens': 60,
                'cache_creation_input_tokens': 0, 'output_tokens': 1}}})
            await send({'type': 'ping'})
            await asyncio.sleep(.01)
            if mode == 'tool':
                await send({'type': 'content_block_start', 'content_block': {'type': 'tool_use', 'id': 't', 'name': 'bash', 'input': {}}})
            else:
                await send({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '你好'}})
            await send({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if mode == 'tool' else 'end_turn'}, 'usage': {'output_tokens': 10}})
            if mode != 'truncated':
                await send({'type': 'message_stop'})
            return response
        app = web.Application()
        app.router.add_post('/anthropic/v1/messages', respond)
        server = TestServer(app)
        await server.start_server()
        config = {'target': {'model': 'mock', 'base_url': str(server.make_url('/anthropic')), 'api_key': 'test-secret'},
                  'run': {'max_tokens': 32}, 'price': {'input_price': 1, 'output_price': 2, 'cache_read_price': .1}}
        try:
            async with aiohttp.ClientSession() as session:
                for next_mode in ['text', 'tool', 'truncated']:
                    mode = next_mode
                    progress = []
                    result = await measure_request(session, config, {'index': 0, 'messages': [{'role': 'user', 'content': '你好'}]},
                        1, asyncio.get_running_loop().time(), 0, lambda tokens, output: progress.append((tokens, output)))
                    self.assertEqual(result['http_status'], 200)
                    self.assertEqual(result['prompt_tokens'], 100)
                    self.assertEqual(result['completion_tokens'], 10)
                    self.assertEqual(result['cached_tokens'], 60)
                    self.assertIsNotNone(result['ttft_ms'])
                    self.assertIsNotNone(result['tpot_ms'])
                    self.assertEqual([t for t, output in progress if t is not None], [1, 10])
                    if mode == 'truncated':
                        self.assertEqual(result['status'], 'error')
                        self.assertIn('message_stop', result['error'])
                    else:
                        self.assertEqual(result['status'], 'ok', result['error'])
                        self.assertEqual(result['estimated_cost'], .000066)
        finally:
            await server.close()
