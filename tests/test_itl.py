import asyncio
import json
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from pybench.results import export_csv, summarize
from pybench.runner import measure_request


class ITLTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_counts_only_output_events_and_same_read_has_zero_gap(self):
        async def respond(request):
            await request.json()
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            # Two output events in one read; empty and usage events are not output.
            first = [
                {'choices': [{'delta': {'role': 'assistant'}}]},
                {'choices': [{'delta': {'content': 'one'}}]},
                {'choices': [{'delta': {'content': 'two tokens'}}]},
                {'usage': {'prompt_tokens': 4, 'completion_tokens': 3}},
            ]
            await response.write(''.join('data: '+json.dumps(c)+'\n\n' for c in first).encode())
            await asyncio.sleep(.05)
            await response.write(('data: '+json.dumps({'choices': [{'delta': {'content': 'end'}}]})+'\n\n').encode())
            await response.write(b'data: [DONE]\n\n')
            return response

        app = web.Application()
        app.router.add_post('/v1/chat/completions', respond)
        server = TestServer(app)
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as session:
                config = {'target': {'base_url': str(server.make_url('/v1')), 'model': 'mock'},
                          'run': {'max_tokens': 8}}
                result = await measure_request(session, config, {'index': 0, 'messages': [{'role':'user','content':'hello'}]},
                                               1, asyncio.get_running_loop().time(), 0)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['itl_count'], 2)
            gaps = json.loads(result['itl_gaps_json'])
            self.assertEqual(gaps[0], 0)
            self.assertGreaterEqual(gaps[1], 35)
            self.assertEqual(result['completion_tokens'], 3)
        finally:
            await server.close()

    async def test_non_stream_response_has_no_itl(self):
        async def respond(request):
            return web.json_response({'choices': [{'message': {'content':'ok'}}], 'usage': {'completion_tokens': 2}})
        app = web.Application()
        app.router.add_post('/v1/chat/completions', respond)
        server = TestServer(app)
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as session:
                result = await measure_request(session, {'target': {'base_url': str(server.make_url('/v1')), 'model':'mock'},
                    'run': {'max_tokens': 8}}, {'index':0, 'messages':[{'role':'user','content':'hello'}]}, 1,
                    asyncio.get_running_loop().time(), 0)
            self.assertIsNone(result['itl_count'])
            self.assertIsNone(result['itl_p50_ms'])
        finally:
            await server.close()

    def test_summary_pools_gaps_not_request_percentiles_and_excludes_errors(self):
        run = {'started_at':'2026-11-04T00:00:00Z','finished_at':'2026-11-04T00:00:01Z',
               'gpu_count':None,'gpu_hour_price':None,'arrival_pattern':'burst','price_snapshot':None,
               'request_rate':0,'id':'r'}
        sample = {'status':'ok','cached_tokens':None,'prompt_tokens':4,'completion_tokens':3,
                  'estimated_cost':None,'sent_offset_ms':0,'queue_delay_ms':0,'ttft_ms':1,'tpot_ms':2,'e2e_ms':5}
        samples = [{**sample,'itl_gaps_json':'[10,20,30]','itl_count':3},
                   {**sample,'itl_gaps_json':'[100]','itl_count':1},
                   {**sample,'status':'error','itl_gaps_json':'[9999]','itl_count':1}]
        result = summarize(run, samples)
        self.assertEqual(result['itl'], {'p50':25.0,'p95':89.5,'p99':97.9,'n':4,'requests':2})
        old = summarize(run, [sample])
        self.assertEqual(old['itl'], {'p50':None,'p95':None,'p99':None,'n':0,'requests':0})
        self.assertIn('itl_p99_ms', export_csv(run, samples, 'summary').decode('utf-8-sig'))

