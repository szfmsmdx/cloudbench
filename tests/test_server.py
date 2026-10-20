from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from server import ROOT, create_app


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cloudbench-test-")
        self.directory = Path(self.temp.name)
        self.requests = []
        mock = web.Application()
        mock.router.add_post("/v1/chat/completions", self.mock_response)
        self.mock = TestServer(mock)
        await self.mock.start_server()
        self.client = TestClient(TestServer(create_app(self.directory)))
        await self.client.start_server()
        self.target = await self.post("/api/targets", {
            "name": "模拟模型", "base_url": str(self.mock.make_url("/v1")),
            "model": "mock-1", "api_key": "test-secret",
        }, 201)

    async def asyncTearDown(self):
        await self.client.close()
        await self.mock.close()
        self.temp.cleanup()

    async def post(self, path, body, status=200):
        response = await self.client.post(path, json=body)
        content = await response.json()
        self.assertEqual(response.status, status, content)
        return content

    async def mock_response(self, request):
        self.assertEqual(request.headers.get("Authorization"), "Bearer test-secret")
        payload = await request.json()
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.requests.append(payload)
        wait = .62 if len(self.requests) <= 2 else .018
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n')
        await asyncio.sleep(wait)
        chunk = {"choices": [{"delta": {"content": "你好 🧪"}}]}
        await response.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\r").encode())
        await asyncio.sleep(.002)
        await response.write(b"\n\r")
        await asyncio.sleep(.002)
        await response.write(b"\n")
        await asyncio.sleep(wait)
        await response.write('data: {"choices":[{"delta":{"content":"，世界"}}]}\n\n'.encode())
        await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_cache_hit_tokens":40,"prompt_cache_miss_tokens":60}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def wait_finished(self, run_id, observe=None):
        deadline = asyncio.get_running_loop().time() + 120
        while asyncio.get_running_loop().time() < deadline:
            response = await self.client.get(f"/api/runs/{run_id}")
            result = await response.json()
            self.assertEqual(response.status, 200, result)
            if observe:
                observe(result)
            if result["run"]["status"] != "running":
                return result
            await asyncio.sleep(.05)
        self.fail("压测进程未在规定时间内结束")

    async def test_python_api_worker_metrics_exports_and_history(self):
        response = await self.client.get("/")
        self.assertEqual(response.status, 200)
        page = await response.text()
        self.assertIn("模型端点", page)
        self.assertIn('name="context_limit_k"', page)
        self.assertNotIn('name="seed"', page)
        self.assertNotIn("dataset-import", page)
        state = await (await self.client.get("/api/state")).json()
        local = [d for d in state["datasets"] if d["kind"].startswith("local-")]
        if len(local) != 2:
            self.skipTest("完整端到端测试需要下载本地 LongBench 和 LongBench v2")
        self.assertEqual([d["id"] for d in local], ["local:longbench", "local:longbench-v2"])
        self.assertEqual(len([d for d in state["datasets"] if d["kind"].startswith("public:")]), 5)
        for dataset in local:
            self.assertEqual(sum(row["count"] for row in dataset["distribution"]), dataset["item_count"])
        one_to_two_k = next(row for row in local[0]["distribution"] if row["range"] == "1K–2K")
        self.assertEqual(one_to_two_k, {"range": "1K–2K", "count": 360, "percentage": 7.58})
        self.assertTrue(state["targets"][0]["has_key"])
        self.assertNotIn("test-secret", json.dumps(state))
        price = await self.post("/api/prices", {
            "provider": "Mock", "model": "mock-1", "currency": "USD", "input_price": 1,
            "cache_read_price": .1, "output_price": 2, "checked_at": "2026-10-19",
        }, 201)
        config = {"target_id": self.target["id"], "dataset_id": "local:longbench",
                  "mode": "sync", "max_tokens": 20, "max_input_tokens": 32768, "request_count": 2}
        run = await self.post("/api/runs", {**config, "price_id": price["id"], "gpu_count": 2,
                                             "gpu_hour_price": 10, "gpu_currency": "CNY"}, 202)
        await self.post("/api/runs", config, 409)
        observed_live = False

        def observe(result):
            nonlocal observed_live
            observed_live |= result["run"]["status"] == "running" and any(p["inflight"] > 0 for p in result["timeline"])

        result = await self.wait_finished(run["id"], observe)
        self.assertEqual(result["run"]["status"], "completed", result["run"]["error"])
        self.assertTrue(observed_live, "长请求尚未完成时应已有实时采样")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(result["summary"]["success"], 2)
        self.assertEqual(result["summary"]["cache_hit_rate"], 40)
        self.assertEqual(result["summary"]["estimated_cost"], .000208)
        self.assertEqual(result["summary"]["hardware_hourly_cost"], 20)
        self.assertEqual(result["run"]["seed"], 42)
        self.assertEqual(result["run"]["concurrency"], 1)
        self.assertGreater(result["summary"]["ttft"]["p50"], 0)
        self.assertGreater(result["summary"]["tpot"]["p50"], 0)
        final = result["timeline"][-1]
        self.assertTrue(final["final"])
        self.assertEqual((final["sent"], final["completed"], final["inflight"]), (2, 2, 0))
        self.assertEqual(final["reported_output_tokens"], 40)
        self.assertEqual(final["usage_known"], 2)
        for index, point in enumerate(result["timeline"]):
            self.assertEqual(point["seq"], index)
            if index:
                self.assertGreaterEqual(point["elapsed_ms"], result["timeline"][index - 1]["elapsed_ms"])
        for kind, expected in (("summary", "hardware_run_cost"), ("samples", "cached_tokens"), ("timeline", "rate_window_ms")):
            response = await self.client.get(f"/api/runs/{run['id']}/{kind}.csv")
            self.assertEqual(response.status, 200)
            content = await response.read()
            self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
            self.assertIn(expected, content.decode("utf-8-sig"))
            self.assertNotIn(b"test-secret", content)
        asynchronous = await self.post("/api/runs", {**config, "mode": "async", "request_rate": 5,
                                                        "request_count": 3, "concurrency": 2}, 202)
        async_result = await self.wait_finished(asynchronous["id"])
        self.assertEqual(async_result["run"]["status"], "completed", async_result["run"]["error"])
        self.assertEqual(async_result["summary"]["success"], 3)
        self.assertEqual(async_result["run"]["arrival_pattern"], "fixed")
        self.assertGreater(async_result["summary"]["actual_arrival_qps"], 0)
        self.assertEqual([s["scheduled_offset_ms"] for s in async_result["samples"]], [0, 200, 400])
        burst = await self.post("/api/runs", {**config, "dataset_id": "local:longbench-v2", "mode": "async",
                                                "request_rate": 0}, 202)
        burst_result = await self.wait_finished(burst["id"])
        self.assertEqual(burst_result["run"]["status"], "completed", burst_result["run"]["error"])
        self.assertEqual(burst_result["summary"]["success"], 2)
        self.assertEqual(burst_result["run"]["concurrency"], 0)
        self.assertEqual(burst_result["run"]["arrival_pattern"], "burst")
        self.assertIsNone(burst_result["summary"]["actual_arrival_qps"])
        self.assertTrue(all(s["scheduled_offset_ms"] == 0 for s in burst_result["samples"]))
        self.assertEqual(len(self.requests), 7)
        self.assertTrue(all("请从 A、B、C、D 中选择正确答案。" in payload["messages"][0]["content"] for payload in self.requests[-2:]))
        cancelled = await self.post("/api/runs", config, 202)
        await self.post(f"/api/runs/{cancelled['id']}/cancel", {})
        cancelled_result = await self.wait_finished(cancelled["id"])
        self.assertEqual(cancelled_result["run"]["status"], "cancelled")
        await self.client.close()
        self.client = TestClient(TestServer(create_app(self.directory)))
        await self.client.start_server()
        persisted = await (await self.client.get(f"/api/runs/{run['id']}")).json()
        self.assertEqual(persisted["timeline"], result["timeline"])
        self.assertEqual(persisted["samples"], result["samples"])
        self.assertEqual(persisted["summary"]["estimated_cost"], .000208)
        state = await (await self.client.get("/api/state")).json()
        self.assertFalse(state["targets"][0]["has_key"])

    async def test_blank_count_defaults_to_all_eligible_samples(self):
        datasets = [d for d in (await (await self.client.get("/api/state")).json())["datasets"] if d["kind"].startswith("local-")]
        if len(datasets) != 2:
            self.skipTest("全量条数测试需要本地离线索引")
        for dataset, index_file, blank in ((datasets[0], "longbench-token-counts.json", {}),
                                            (datasets[1], "longbench-v2-token-counts.json", {"request_count": ""})):
            counts = json.loads((ROOT / "datasets" / index_file).read_text(encoding="utf-8"))["counts"]
            expected_count = sum(length <= 131072 - 20 - 512 for length in counts)
            config = {"target_id": self.target["id"], "dataset_id": dataset["id"], "mode": "async",
                      "request_rate": 0, "max_tokens": 20, "max_input_tokens": 131072, **blank}
            run = await self.post("/api/runs", config, 202)
            await self.post(f"/api/runs/{run['id']}/cancel", {})
            result = await self.wait_finished(run["id"])
            self.assertEqual(result["run"]["status"], "cancelled")
            self.assertEqual(result["run"]["request_count"], expected_count)
            self.assertEqual(result["run"]["concurrency"], 0)
            if dataset["kind"] == "local-longbench-v2":
                self.assertLess(expected_count, 503)
            else:
                self.assertEqual(expected_count, 4750)
        await self.post("/api/runs", {**config, "request_count": 0}, 400)
        await self.post("/api/runs", {**config, "max_input_tokens": 2048, "request_count": ""}, 400)

    async def test_validation_key_management_and_access(self):
        response = await self.client.post("/api/targets", json={}, headers={"Origin": "https://example.org"})
        self.assertEqual(response.status, 403)
        await self.post(f"/api/targets/{self.target['id']}/key", {"api_key": "replacement-secret"})
        await self.post("/api/targets/missing/key", {"api_key": "x"}, 404)
        await self.post("/api/targets", {"name": "bad", "model": "x", "base_url": "file:///etc/passwd"}, 400)
        await self.post("/api/targets", {"name": "bad", "model": "x", "base_url": "https://host/v1?key=secret"}, 400)
        await self.post("/api/prices", {"provider": "mock", "model": "x", "input_price": -1, "output_price": 1}, 400)
        await self.post("/api/runs", {"target_id": "missing"}, 404)
        for path in ("/api/runs/missing", "/api/runs/missing/timeline.csv", "/api/missing", "/server.py"):
            response = await self.client.get(path)
            self.assertEqual(response.status, 404, path)
        response = await self.client.get("/charts.js")
        self.assertEqual(response.status, 200)
        self.assertIn("renderTimeline", await response.text())
        response = await self.client.delete(f"/api/targets/{self.target['id']}")
        self.assertEqual(response.status, 200)
        self.assertEqual((await (await self.client.get("/api/state")).json())["targets"], [])


if __name__ == "__main__":
    unittest.main()
