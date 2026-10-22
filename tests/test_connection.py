import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from pybench.connection import connect
from server import create_app


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.models = ['local-model']
        self.kind = 'openai'
        self.endpoint = '/v1/chat/completions'
        self.key = None
        self.failure = None
        self.redirect = None
        self.mock = TestServer(web.Application())
        self.mock.app.router.add_route('*', '/{path:.*}', self.respond)
        await self.mock.start_server()
        self.temp = tempfile.TemporaryDirectory()
        self.client = TestClient(TestServer(create_app(Path(self.temp.name))))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.mock.close()
        self.temp.cleanup()

    async def respond(self, request):
        body = await request.json() if request.method == 'POST' else None
        self.calls.append((request.path, body))
        if self.redirect and request.method == 'POST':
            raise web.HTTPTemporaryRedirect(self.redirect)
        if self.key:
            auth = request.headers.get('x-api-key') if self.kind == 'anthropic' else request.headers.get('Authorization', '').removeprefix('Bearer ')
            if auth != self.key:
                return web.json_response({'error': 'unauthorized'}, status=401)
        else:
            self.assertNotIn('Authorization', request.headers)
            self.assertNotIn('x-api-key', request.headers)
        if request.method == 'GET' and request.path == '/v1/models' and self.models is not None:
            return web.json_response({'data': [{'id': m} for m in self.models]})
        if request.method != 'POST' or request.path != self.endpoint:
            return web.Response(status=404)
        if self.failure:
            if self.failure == 'html':
                return web.Response(text='<html>login</html>', content_type='text/html')
            return web.json_response({'error': 'secret should never be reflected'}, status=self.failure)
        if self.kind == 'anthropic':
            if 'stream_options' in body:
                return web.Response(status=400)
            self.assertEqual(request.headers.get('anthropic-version'), '2023-06-01')
            events = [{'type': 'content_block_delta', 'delta': {'text': 'OK'}}, {'type': 'message_stop'}]
        else:
            events = [{'choices': [{'delta': {'content': 'OK'}, 'finish_reason': 'stop'}]}]
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        for event in events:
            await response.write(('data: ' + json.dumps(event) + '\n\n').encode())
        await response.write(b'data: [DONE]\n\n')
        return response

    async def test_url_only_no_key_discovers_verifies_and_persists(self):
        response = await self.client.post('/api/targets', json={'base_url': str(self.mock.make_url(''))})
        target = await response.json()
        self.assertEqual(response.status, 201, target)
        self.assertEqual(target['model'], 'local-model')
        self.assertEqual(target['protocol'], 'openai')
        self.assertTrue(target['endpoint_url'].endswith('/v1/chat/completions'))
        self.assertFalse(target['has_key'])
        with closing(sqlite3.connect(Path(self.temp.name) / 'cloudbench.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 0)
        response = await self.client.post(f"/api/targets/{target['id']}/connect", json={})
        self.assertEqual(response.status, 200)

    async def test_multiple_models_and_missing_list_do_not_generate_or_save(self):
        for models in (['a', 'b'], None):
            self.models = models
            response = await self.client.post('/api/targets', json={'base_url': str(self.mock.make_url(''))})
            result = await response.json()
            self.assertEqual(result['status'], 'needs_model')
            self.assertEqual(result['models'], models or [])
        self.assertFalse(any(body for _, body in self.calls))
        state = await (await self.client.get('/api/state')).json()
        self.assertEqual(state['targets'], [])

    async def test_selected_model_skips_discovery_and_verifies(self):
        self.models = ['a', 'b']
        response = await self.client.post('/api/targets', json={
            'base_url': str(self.mock.make_url('')), 'model': 'b'})
        target = await response.json()
        self.assertEqual(response.status, 201, target)
        self.assertEqual(target['model'], 'b')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]['model'], 'b')

    async def test_redirect_is_not_followed_or_treated_as_success(self):
        self.redirect = str(self.mock.make_url('/other'))
        with self.assertRaises(ValueError):
            await connect(str(self.mock.make_url('/v1/chat/completions')), model='local-model')
        self.assertEqual([path for path, _ in self.calls], ['/v1/chat/completions'])

    async def test_bare_url_falls_back_to_anthropic_with_key(self):
        self.kind, self.endpoint, self.key = 'anthropic', '/v1/messages', 'test-key'
        result = await connect(str(self.mock.make_url('')), self.key)
        self.assertEqual(result['protocol'], 'anthropic')
        self.assertEqual(result['attempts'][0]['http_status'], 401)
        self.assertEqual(result['attempts'][-1]['http_status'], 200)
        self.assertNotIn(self.key, json.dumps(result))

    async def test_complete_and_custom_urls_are_tried_exactly(self):
        for endpoint, kind in [('/tenant/chat/completions', 'openai'), ('/tenant/messages', 'anthropic'), ('/inference', 'anthropic')]:
            self.endpoint, self.kind = endpoint, kind
            self.calls.clear()
            result = await connect(str(self.mock.make_url(endpoint)), model='local-model')
            self.assertEqual(result['protocol'], kind)
            self.assertTrue(all(path == endpoint for path, _ in self.calls))

    async def test_manual_override_is_persisted_for_custom_endpoint(self):
        self.endpoint, self.kind = '/generate', 'anthropic'
        response = await self.client.post('/api/targets', json={
            'base_url': str(self.mock.make_url('/generate')), 'model': 'local-model', 'protocol': 'anthropic'})
        target = await response.json()
        self.assertEqual(response.status, 201, target)
        self.calls.clear()
        response = await self.client.post(f"/api/targets/{target['id']}/connect", json={})
        self.assertEqual(response.status, 200)
        self.assertEqual([p for p, _ in self.calls], ['/generate'])

    async def test_failed_auth_html_and_rate_limit_never_save_targets(self):
        for failure in (401, 'html', 429):
            self.failure = failure
            response = await self.client.post('/api/targets', json={
                'base_url': str(self.mock.make_url('/v1/chat/completions')), 'model': 'local-model'})
            self.assertEqual(response.status, 400, await response.text())
            state = await (await self.client.get('/api/state')).json()
            self.assertEqual(state['targets'], [])

    async def test_existing_database_is_migrated_without_rewriting_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with closing(sqlite3.connect(path / 'cloudbench.sqlite')) as db:
                db.execute('CREATE TABLE targets (id TEXT PRIMARY KEY, name TEXT, base_url TEXT, model TEXT, created_at TEXT)')
                db.execute("INSERT INTO targets VALUES ('old', 'old', ?, 'local-model', '2026-01-01')", (str(self.mock.make_url('')),))
                db.commit()
            client = TestClient(TestServer(create_app(path)))
            await client.start_server()
            try:
                state = await (await client.get('/api/state')).json()
                self.assertIsNone(state['targets'][0]['endpoint_url'])
                response = await client.post('/api/targets/old/connect', json={})
                self.assertEqual(response.status, 200)
                state = await (await client.get('/api/state')).json()
                self.assertEqual(state['targets'][0]['protocol'], 'openai')
            finally:
                await client.close()
