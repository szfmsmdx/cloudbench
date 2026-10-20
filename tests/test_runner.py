import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from pybench.datasets import eligible_count, length_distribution, load_local_items
from pybench.runner import run, schedule_requests
from pybench.telemetry import RunTelemetry


class SchedulerTests(unittest.TestCase):
    def test_unlimited_burst_starts_every_request_before_waiting_for_completion(self):
        async def scenario():
            release = asyncio.Event()
            started = []
            async def send(seq, planned):
                self.assertEqual(planned, 0)
                started.append(seq)
                if len(started) == 128:
                    release.set()
                await release.wait()
            await asyncio.wait_for(schedule_requests(128, 'async', 0, 0, send), 2)
            self.assertEqual(len(started), 128)
        asyncio.run(scenario())

    def test_sync_waits_for_each_request(self):
        async def scenario():
            starts = []
            in_flight = 0
            peak = 0

            async def send(seq, planned):
                nonlocal in_flight, peak
                self.assertIsNone(planned)
                starts.append(asyncio.get_running_loop().time())
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.04)
                in_flight -= 1

            await schedule_requests(3, "sync", None, 8, send)
            self.assertEqual(peak, 1)
            self.assertGreaterEqual(starts[1] - starts[0], 0.035)

        asyncio.run(scenario())

    def test_qps_five_targets_200ms_without_concurrency_limit(self):
        async def scenario():
            starts = []

            async def send(seq, planned):
                starts.append((asyncio.get_running_loop().time(), planned))
                await asyncio.sleep(0.3)

            await schedule_requests(4, "async", 5, 0, send)
            self.assertEqual([planned for _, planned in starts], [0.0, 200.0, 400.0, 600.0])
            for before, after in zip(starts, starts[1:]):
                self.assertAlmostEqual(after[0] - before[0], 0.2, delta=0.055)

        asyncio.run(scenario())

    def test_zero_qps_bursts_up_to_concurrency_limit(self):
        async def scenario():
            starts = []
            in_flight = 0
            peak = 0

            async def send(seq, planned):
                nonlocal in_flight, peak
                starts.append((asyncio.get_running_loop().time(), planned))
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.08)
                in_flight -= 1

            await schedule_requests(5, "async", 0, 2, send)
            self.assertEqual(peak, 2)
            self.assertTrue(all(planned == 0 for _, planned in starts))
            self.assertLess(starts[1][0] - starts[0][0], 0.04)
            self.assertGreaterEqual(starts[2][0] - starts[0][0], 0.06)

        asyncio.run(scenario())

    def test_fixed_qps_waits_at_concurrency_limit(self):
        async def scenario():
            starts = []

            async def send(seq, planned):
                starts.append((asyncio.get_running_loop().time(), planned))
                await asyncio.sleep(0.06)

            await schedule_requests(3, "async", 100, 1, send)
            self.assertEqual([planned for _, planned in starts], [0.0, 10.0, 20.0])
            for before, after in zip(starts, starts[1:]):
                self.assertGreaterEqual(after[0] - before[0], 0.05)

        asyncio.run(scenario())


class UnlimitedHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_unlimited_mode_has_no_hidden_http_connection_cap(self):
        count, peak, active, arrived = 128, 0, 0, 0
        release = asyncio.Event()
        events = []
        async def respond(request):
            nonlocal peak, active, arrived
            await request.json()
            active += 1
            arrived += 1
            peak = max(peak, active)
            if arrived == count:
                release.set()
            await release.wait()
            active -= 1
            return web.json_response({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
                                      'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
        app = web.Application()
        app.router.add_post('/v1/chat/completions', respond)
        server = TestServer(app)
        await server.start_server()
        config = {'target': {'base_url': str(server.make_url('/v1')), 'model': 'mock'},
                  'dataset': {'kind': 'public:hotpotqa', 'selected': [
                      {'id': str(i), 'index': i, 'turns': [{'messages': [{'role': 'user', 'content': str(i)}], 'input_tokens': 100}]}
                      for i in range(count)]},
                  'run': {'mode': 'async', 'request_rate': 0, 'concurrency': 0, 'request_count': count,
                          'unit_count': count, 'max_tokens': 10, 'request_timeout': 10, 'seed': 42}}
        try:
            with patch('pybench.runner.emit', events.append):
                await asyncio.wait_for(run(config), 10)
            self.assertEqual(peak, count)
            samples = [e['sample'] for e in events if e['type'] == 'sample']
            self.assertEqual(len(samples), count)
            self.assertTrue(all(s['status'] == 'ok' for s in samples))
        finally:
            release.set()
            await server.close()


class LocalDatasetTests(unittest.TestCase):
    def test_length_distribution_uses_binary_k_boundaries(self):
        rows = length_distribution([1023, 1024, 2047, 2048])
        self.assertEqual([row["count"] for row in rows[:3]], [1, 2, 1])
        self.assertEqual([row["percentage"] for row in rows[:3]], [25, 50, 25])
        self.assertEqual(sum(row["count"] for row in rows), 4)

    def test_v1_excludes_e_and_respects_token_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "datasets/raw/longbench/extracted/data"
            directory.mkdir(parents=True)
            (directory / "a.jsonl").write_text(
                '\n'.join(json.dumps({"context": str(i), "input": "question"}) for i in range(2)), encoding="utf-8"
            )
            (directory / "a_e.jsonl").write_text(json.dumps({"context": "exclude", "input": "me"}), encoding="utf-8")
            (root / "datasets/longbench-token-counts.json").write_text(json.dumps({"counts": [100, 2000]}), encoding="utf-8")
            items = load_local_items("local-longbench", root, 2048, 128)
            self.assertEqual(eligible_count("local-longbench", root, 2048, 128), len(items))
            self.assertEqual([item["index"] for item in items], [0])
            self.assertEqual(items[0]["messages"][0]["content"], "0\nquestion")

    def test_v2_filters_oversized_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "datasets/raw/longbench-v2"
            directory.mkdir(parents=True)
            rows = [{"context": "source", "question": "Q?", "choice_A": "yes"}, {"context": "too long", "question": "Q?"}]
            (directory / "data.json").write_text(json.dumps(rows), encoding="utf-8")
            (root / "datasets/longbench-v2-token-counts.json").write_text(json.dumps({"counts": [100, 2000]}), encoding="utf-8")
            items = load_local_items("local-longbench-v2", root, 2048, 128)
            self.assertEqual(eligible_count("local-longbench-v2", root, 2048, 128), len(items))
            self.assertEqual([item["index"] for item in items], [0])
            self.assertIn("A. yes", items[0]["messages"][0]["content"])


class TelemetryTests(unittest.TestCase):
    def test_rates_use_window_duration_and_usage_is_not_double_counted(self):
        counters = RunTelemetry(10)
        counters.snapshot(10)
        counters.request_started()
        counters.request_started()
        counters.progress(1, 4, True)
        counters.progress(1, 6, True)
        counters.progress(2, None, True)
        first = counters.snapshot(11)
        self.assertEqual(first["sent_qps"], 2)
        self.assertEqual(first["inflight"], 2)
        self.assertEqual(first["reported_output_tps"], 6)
        self.assertEqual(first["stream_events_per_s"], 3)
        counters.request_finished({"seq": 1, "status": "ok", "completion_tokens": 8, "ttft_ms": 100, "e2e_ms": 1000})
        counters.request_finished({"seq": 2, "status": "error", "completion_tokens": None, "ttft_ms": None, "e2e_ms": 1100})
        final = counters.snapshot(11.5, final=True)
        self.assertEqual(final["window_ms"], 500)
        self.assertEqual(final["rate_window_ms"], 1500)
        self.assertAlmostEqual(final["completed_qps"], 2 / 1.5, places=5)
        self.assertAlmostEqual(final["success_qps"], 1 / 1.5, places=5)
        self.assertAlmostEqual(final["failed_qps"], 1 / 1.5, places=5)
        self.assertAlmostEqual(final["reported_output_tps"], 8 / 1.5, places=5)
        self.assertEqual(final["reported_output_tokens"], 8)
        self.assertEqual(final["inflight"], 0)
        self.assertEqual(final["ttft_p95_ms"], 100)
        self.assertEqual((final["usage_known"], final["usage_missing"]), (1, 1))
        self.assertTrue(final["final"])

    def test_stream_chunks_are_not_assumed_to_be_tokens(self):
        counters = RunTelemetry(0)
        counters.request_started()
        counters.progress(1, None, True)
        point = counters.snapshot(1)
        self.assertIsNone(point["reported_output_tps"])
        self.assertEqual(point["stream_events_per_s"], 1)


if __name__ == "__main__":
    unittest.main()
