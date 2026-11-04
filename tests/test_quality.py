import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from pybench import gpqa
from server import create_app, utc_now


class ScoringTests(unittest.TestCase):
    def test_only_final_commitment_and_truncation(self):
        self.assertIsNone(gpqa.extract_answer('I considered (A), but maybe B.'))
        self.assertEqual(gpqa.extract_answer('ANSWER: A\nActually reconsidering.\nANSWER: C'), 'C')
        self.assertFalse(gpqa.score('ANSWER: C', 'C', 'length')['correct'])
        self.assertTrue(gpqa.score('Some reasoning\nANSWER: C', 'C', 'stop')['correct'])

    def test_seed_and_correct_answer_mapping(self):
        data = {'rows': [{'Question': str(i), 'Correct Answer': 'yes',
                         **{f'Incorrect Answer {j}': str(j) for j in range(1, 4)}} for i in range(20)]}
        first = gpqa.questions(data, 10, 42)
        self.assertEqual(first, gpqa.questions(data, 10, 42))
        self.assertEqual(first, gpqa.questions(data, 20, 42)[:10])
        self.assertEqual(len({q['source_index'] for q in first}), 10)
        for question in first:
            self.assertEqual(question['options'][question['expected']], 'yes')


class QualityApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'public').mkdir()
        (self.root / 'datasets').mkdir()
        self.rows = [{'Question': 'Find the correct choice.', 'Correct Answer': 'gold-value',
                      **{f'Incorrect Answer {i}': f'wrong-{i}' for i in range(1, 4)}} for _ in range(198)]
        (self.root / 'datasets/gpqa-diamond.json').write_text(json.dumps(
            {'revision': gpqa.REVISION, 'csv_sha256': 'fixture', 'rows': self.rows}), encoding='utf-8')
        self.calls = []
        self.mode = 'normal'
        mock = web.Application()
        mock.router.add_post('/v1/chat/completions', self.reply)
        mock.router.add_post('/v1/messages', self.anthropic_reply)
        self.mock = TestServer(mock)
        await self.mock.start_server()
        app = create_app(self.root / 'data', self.root)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        # Obtain the bound owner from a route without adding test-only application state.
        self.owner = next(r.handler.__self__ for r in app.router.routes()
                          if r.resource.canonical == '/api/state')
        self.owner.insert('targets', {'id': 'mock', 'name': 'Mock', 'base_url': str(self.mock.make_url('/v1')),
                                     'model': 'mock', 'created_at': utc_now(), 'protocol': 'openai'})
        self.owner.credentials['mock'] = 'private-test-key'

    async def asyncTearDown(self):
        await self.client.close()
        await self.mock.close()
        self.temp.cleanup()

    async def reply(self, request):
        payload = await request.json()
        self.calls.append(payload)
        self.assertFalse(payload['stream'])
        self.assertNotIn('stream_options', payload)
        self.assertEqual(request.headers['Authorization'], 'Bearer private-test-key')
        if self.mode == 'wait':
            await asyncio.sleep(.3)
        seq = len(self.calls)
        if seq == 3:
            return web.json_response({'error': {'message': 'private-test-key rejected'}}, status=429)
        prompt = payload['messages'][0]['content']
        answer = next(line[0] for line in prompt.splitlines() if line.endswith('. gold-value'))
        text = f'ANSWER: {answer}' if seq == 1 else 'No final answer'
        return web.json_response({'model': 'mock', 'choices': [{'message': {'content': text},
                                  'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 100, 'completion_tokens': 12}})

    async def wait_run(self, ident):
        for _ in range(1000):
            result = await (await self.client.get(f'/api/quality/{ident}')).json()
            if result['status'] != 'running':
                return result
            await asyncio.sleep(.01)
        self.fail('quality run did not finish')

    async def anthropic_reply(self, request):
        payload = await request.json()
        self.assertEqual(request.headers['x-api-key'], 'private-test-key')
        self.assertFalse(payload['stream'])
        prompt = payload['messages'][0]['content'][0]['text']
        answer = next(line[0] for line in prompt.splitlines() if line.endswith('. gold-value'))
        return web.json_response({'model': 'mock-anthropic', 'stop_reason': 'end_turn',
                                 'content': [{'type': 'thinking', 'thinking': 'Reasoning'},
                                             {'type': 'text', 'text': f'ANSWER: {answer}'}],
                                 'usage': {'input_tokens': 100, 'output_tokens': 20,
                                           'cache_read_input_tokens': 50}})

    async def test_anthropic_generation_and_price_snapshot(self):
        self.owner.db.execute("UPDATE targets SET protocol='anthropic',endpoint_url=? WHERE id='mock'",
                              (str(self.mock.make_url('/v1/messages')),))
        price = {'id': 'price', 'provider': 'mock', 'model': 'mock', 'currency': 'USD',
                 'input_price': 1, 'cache_read_price': .1, 'output_price': 2,
                 'checked_at': utc_now(), 'created_at': utc_now()}
        self.owner.insert('prices', price)
        response = await self.client.post('/api/quality', json={'target_id': 'mock', 'count': 198, 'price_id': 'price'})
        ident = (await response.json())['id']
        result = await self.wait_run(ident)
        self.assertEqual(result['summary']['correct'], 198)
        self.assertAlmostEqual(result['summary']['estimated_cost'], .000145 * 198)
        self.assertEqual(result['samples'][0]['reasoning'], 'Reasoning')

    async def test_generation_scoring_export_and_history(self):
        response = await self.client.post('/api/quality', json={'target_id': 'mock', 'count': 198})
        self.assertEqual(response.status, 201)
        ident = (await response.json())['id']
        result = await self.wait_run(ident)
        self.assertEqual(result['summary']['correct'], 1)
        self.assertEqual(result['summary']['unparsed'], 196)
        self.assertEqual(result['summary']['errors'], 1)
        self.assertEqual(result['summary']['accuracy'], .51)
        self.assertNotIn('private-test-key', json.dumps(result))
        exported = await (await self.client.get(f'/api/quality/{ident}/export')).json()
        self.assertEqual(exported, result)
        listing = await (await self.client.get('/api/quality')).json()
        self.assertEqual(listing['runs'][0]['id'], ident)
        self.assertNotIn('samples', listing['runs'][0])

    async def test_cancel_exclusion_and_validation(self):
        response = await self.client.post('/api/quality', json={'target_id': 'mock', 'count': 199})
        self.assertEqual(response.status, 400)
        self.mode = 'wait'
        response = await self.client.post('/api/quality', json={'target_id': 'mock', 'count': 198})
        ident = (await response.json())['id']
        response = await self.client.post('/api/quality', json={'target_id': 'mock'})
        self.assertEqual(response.status, 409)
        await self.client.post(f'/api/quality/{ident}/cancel')
        result = await self.wait_run(ident)
        self.assertEqual(result['status'], 'cancelled')
        self.assertLess(result['summary']['completed'], 198)
        saved = self.owner.quality.read(ident)
        self.assertEqual(saved['status'], 'cancelled')
