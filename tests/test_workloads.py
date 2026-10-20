from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from pybench.prepare import agent_item, prepare, rag_item
from pybench.sources import source_rows
from pybench.workloads import IndexedWriter, catalog, hydrate_workload, iter_workload_turns, load_workloads, paths, select_workloads, select_for_run, write_snapshot
from server import create_app


def trajectory():
    return {"trajectory_id": "trace-1", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "inspect repository"},
        {"role": "assistant", "content": "recorded action", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "content": "recorded observation"},
        {"role": "assistant", "content": "recorded answer"},
    ]}


class AdapterTests(unittest.TestCase):
    def test_parquet_auxiliary_fields_remain_in_intact_source_and_all_rows_are_indexable(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "datasets/raw/swe-smith"
            source_dir.mkdir(parents=True)
            raw = trajectory()
            source = source_dir / "0000.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"messages": json.dumps(raw["messages"]), "traj_id": str(i), "patch": "keep-original-patch" * 1000}
                for i in range(137)]), source)
            rows = list(source_rows(root, "swe-smith", {"format": "parquet", "files": [{"filename": source.name}]}))
            self.assertEqual(len(rows), 137)
            self.assertNotIn("patch", rows[136])
            self.assertEqual(rows[136]["_cloudbench_source"]["row"], 136)
            self.assertEqual(rows[136]["_cloudbench_source"]["fields_in_source"], ["patch"])
            original = pq.read_table(source, columns=["patch"]).to_pylist()
            self.assertEqual(original[136]["patch"], "keep-original-patch" * 1000)

    def test_default_preparation_keeps_full_source_without_tokenizing_or_indexing(self):
        class Counter:
            fingerprint, label = "fixture", "fixture tokens"
            def __init__(self, *args):
                pass
            def count_item(self, item):
                for turn in item["turns"]:
                    turn["input_tokens"] = 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"_id": str(i), "question": f"q{i}", "answer": f"gold{i}",
                "context": [["article", [f"evidence{i}"]]], "extra_original_field": i}) + "\n" for i in range(137)), encoding="utf-8")
            with patch("pybench.prepare.Counter", side_effect=AssertionError('Must not tokenize entire source')):
                info = prepare(root, "hotpotqa", source=source)
            self.assertEqual((info["raw_count"], info["item_count"]), (137, 137))
            self.assertTrue(info["complete"])
            self.assertFalse((root / 'datasets/prepared/hotpotqa.sqlite3').exists())
            with patch('pybench.prepare.Counter', Counter):
                units = select_for_run(root, 'public:hotpotqa', 4096, 128, 0, None, 42)
            self.assertEqual(len(units), 137)
            item = next(item for item in units if item['index'] == 136)
            self.assertIn("evidence136", json.dumps(item["turns"]))
            self.assertNotIn("gold136", json.dumps(item["turns"]))
            raw = list(map(json.loads, (root / 'datasets/raw/hotpotqa/source.jsonl').read_text().splitlines()))[136]
            self.assertEqual(raw["extra_original_field"], 136)
            self.assertEqual(raw["answer"], "gold136")

    def test_agent_default_keeps_more_than_16_turns(self):
        messages = []
        for i in range(23):
            messages.extend([{"role": "user", "content": str(i)}, {"role": "assistant", "content": "answer"}])
        item = agent_item({"messages": messages}, 0)
        self.assertEqual(len(item["turns"]), 23)

    def test_long_agent_import_and_replay_share_history_without_prefix_copies(self):
        messages = []
        for i in range(2000):
            messages.extend([{"role": "user", "content": str(i)}, {"role": "assistant", "content": "answer"}])
        item = agent_item({"messages": messages}, 0, compact=True)
        self.assertEqual(len(item["turns"]), 2000)
        self.assertEqual(len(item["messages"]), 4000)
        self.assertTrue(all("messages" not in turn for turn in item["turns"]))
        for turn in item["turns"]:
            turn["input_tokens"] = 100
        replay = iter_workload_turns(item)
        self.assertEqual(next(replay)["messages"], [{"role": "user", "content": "0"}])
        self.assertEqual(len(next(replay)["messages"]), 3)

    def test_excess_request_count_is_rejected_instead_of_repeating_samples(self):
        with self.assertRaisesRegex(ValueError, "不能通过重复"):
            select_workloads([{"id": "only-one"}], 2, 42)

    def test_unusable_record_is_retained_in_indexed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = IndexedWriter(root, "open-swe")
            writer.append({"messages": [], "original": "keep-me"}, error="no turns")
            item = agent_item(trajectory(), 1)
            for turn in item["turns"]:
                turn["input_tokens"] = 100
            writer.append(trajectory(), item)
            info = writer.finish({"complete": True})
            self.assertEqual((info["raw_count"], info["item_count"]), (2, 1))
            units = load_workloads(root, "public:open-swe", 4096, 128)
            self.assertEqual([unit["index"] for unit in units], [1])

    def test_agent_preserves_reference_prefix_and_matches_missing_ids(self):
        item = agent_item(trajectory(), 0, 16)
        self.assertEqual(len(item["turns"]), 2)
        first, second = item["turns"]
        self.assertEqual(first["messages"], second["messages"][:2])
        self.assertEqual(second["messages"][-1]["tool_call_id"], "call-1")
        self.assertEqual(first["tools"], second["tools"])
        self.assertEqual(json.loads(second["messages"][-2]["tool_calls"][0]["function"]["arguments"]), {"command": "ls"})
        self.assertNotIn("recorded answer", json.dumps(second))

    def test_swe_smith_string_messages_and_tool_call_ids(self):
        raw = trajectory()
        raw["messages"][3]["tool_call_ids"] = ["call-1"]
        raw["messages"] = json.dumps(raw["messages"])
        item = agent_item(raw, 0, 1)
        self.assertEqual(len(item["turns"]), 1)
        self.assertEqual(item["turns"][0]["tools"][0]["function"]["name"], "bash")

    def test_broken_tool_mapping_is_rejected(self):
        raw = trajectory()
        raw["messages"][3]["tool_call_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "不匹配"):
            agent_item(raw, 0, 16)

    def test_hotpot_keeps_distractors_but_not_answer(self):
        item = rag_item("hotpotqa", {"question": "question", "answer": "secret-gold",
            "context": {"title": ["support", "distractor"], "sentences": [["evidence"], ["noise"]]}}, 0)
        prompt = json.dumps(item["turns"])
        self.assertIn("noise", prompt)
        self.assertNotIn("secret-gold", prompt)
        self.assertEqual(item["reference_answer"], "secret-gold")

    def test_crag_html_is_text_and_empty_context_is_rejected(self):
        item = rag_item("crag", {"query": "q", "answer": "gold", "search_results": [
            {"page_result": "<p>evidence</p><script>hidden()</script>"}]}, 0)
        prompt = item["turns"][0]["messages"][0]["content"]
        self.assertIn("evidence", prompt)
        self.assertNotIn("hidden()", prompt)
        with self.assertRaises(ValueError):
            rag_item("crag", {"query": "q", "search_results": []}, 0)

    def test_crud_uses_article_not_reference_summary(self):
        item = rag_item("crud-rag", {"task": "event_summary", "event": "short event",
            "text": "full article", "summary": "secret-gold"}, 0)
        self.assertIn("full article", json.dumps(item["turns"]))
        self.assertNotIn("secret-gold", json.dumps(item["turns"]))

    def test_crud_qa_keeps_multiple_documents_without_answer_reasoning(self):
        item = rag_item("crud-rag", {"task": "questanswer_2docs", "ID": "42", "questions": "q",
            "news1": "article-one", "news2": "article-two", "thoughts": "secret-reasoning", "answers": "gold"}, 0)
        prompt = json.dumps(item["turns"])
        self.assertIn("article-one", prompt)
        self.assertIn("article-two", prompt)
        self.assertNotIn("secret-reasoning", prompt)
        self.assertNotIn("gold", prompt)
        self.assertEqual(item["reference_answer"], "gold")
        self.assertEqual(item["id"], "questanswer_2docs:42")

    def test_crud_continuation_does_not_include_future_text(self):
        item = rag_item("crud-rag", {"task": "continuing_writing", "beginning": "opening",
            "continuing": "future-gold", "summary": "summary-gold"}, 0)
        prompt = json.dumps(item["turns"])
        self.assertIn("opening", prompt)
        self.assertNotIn("future-gold", prompt)
        self.assertNotIn("summary-gold", prompt)

    def test_crud_hallucination_correction_keeps_evidence(self):
        item = rag_item("crud-rag", {"task": "hallu_modified", "newsBeginning": "opening",
            "newsRemainder": "factual evidence", "hallucinatedContinuation": "wrong claim",
            "realContinuation": "gold", "allKeywords": "secret-annotation"}, 0)
        prompt = json.dumps(item["turns"])
        self.assertIn("factual evidence", prompt)
        self.assertIn("wrong claim", prompt)
        self.assertNotIn("secret-annotation", prompt)
        self.assertNotIn("gold", prompt)

    def test_length_filter_keeps_cache_prefix_and_detects_modified_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = agent_item(trajectory(), 0, 16)
            for turn, size in zip(item["turns"], [100, 500]):
                turn["input_tokens"] = size
            write_snapshot(root, "open-swe", [item], {"length_label": "test"})
            loaded = load_workloads(root, "public:open-swe", 4096, 128, 400)
            self.assertEqual(len(loaded[0]["turns"]), 2)
            self.assertEqual(catalog(root)[0]["request_total"], 2)
            data, _ = paths(root, "open-swe")
            with data.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "变化"):
                load_workloads(root, "public:open-swe", 4096, 128)


class PublicWorkloadAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "public").mkdir()
        self.payloads = []
        item = agent_item(trajectory(), 0, 16)
        item["reference_answer"] = "never-send-reference"
        for turn in item["turns"]:
            turn["input_tokens"] = 100
        writer = IndexedWriter(self.root, "open-swe")
        writer.append(trajectory(), item)
        raw2 = trajectory()
        raw2["trajectory_id"] = "trace-2"
        item2 = {**item, "id": "trace-2"}
        writer.append(raw2, item2)
        writer.finish({"tokenizer": "test", "length_label": "test", "complete": True})

        async def respond(request):
            payload = await request.json()
            self.payloads.append(payload)
            await asyncio.sleep(.01)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            # Tool-call-only streams must count as valid output.
            chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "live-id", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}
            await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10}}\n\n')
            await response.write(b"data: [DONE]\n\n")
            return response

        mock = web.Application()
        mock.router.add_post("/v1/chat/completions", respond)
        self.mock = TestServer(mock)
        await self.mock.start_server()
        self.client = TestClient(TestServer(create_app(self.root / "data", self.root)))
        await self.client.start_server()
        response = await self.client.post("/api/targets", json={"name": "mock", "model": "mock", "base_url": str(self.mock.make_url("/v1"))})
        self.target = await response.json()

    async def asyncTearDown(self):
        await self.client.close()
        await self.mock.close()
        self.temp.cleanup()

    async def test_child_worker_replays_sessions_without_live_output_or_reference_leak(self):
        response = await self.client.post("/api/runs", json={"target_id": self.target["id"],
            "dataset_id": "public:open-swe", "mode": "async", "request_rate": 0,
            "request_count": 2, "concurrency": 2, "max_tokens": 128, "request_timeout": 900})
        run = await response.json()
        self.assertEqual(response.status, 202, run)
        for _ in range(200):
            result = await (await self.client.get(f"/api/runs/{run['id']}" )).json()
            if result["run"]["status"] != "running":
                break
            await asyncio.sleep(.05)
        self.assertEqual(result["run"]["status"], "completed", result["run"].get("error"))
        self.assertEqual(result["run"]["request_count"], 4)
        self.assertEqual(result["summary"]["success"], 4)
        self.assertEqual({s["turn_index"] for s in result["samples"]}, {1, 2})
        self.assertEqual(len({s["session_id"] for s in result["samples"]}), 2)
        self.assertTrue(all(s["finish_reason"] == "tool_calls" for s in result["samples"]))
        self.assertTrue(all(p.get("tools") for p in self.payloads))
        later = [p for p in self.payloads if len(p["messages"]) == 4]
        self.assertEqual(len(later), 2)
        self.assertEqual(later[0]["messages"][-1]["content"], "recorded observation")
        self.assertNotIn("live-id", json.dumps(later))
        self.assertNotIn("never-send-reference", json.dumps(self.payloads))
        exported = await self.client.get(f"/api/runs/{run['id']}/samples.csv")
        self.assertIn("session_id", await exported.text())

    async def test_unprepared_source_and_empty_range_fail_before_any_request(self):
        config = {"target_id": self.target["id"], "dataset_id": "public:crag", "mode": "sync", "max_tokens": 128}
        response = await self.client.post("/api/runs", json=config)
        self.assertEqual(response.status, 400)
        response = await self.client.post("/api/runs", json={**config, "dataset_id": "public:open-swe", "min_input_tokens": 1000})
        self.assertEqual(response.status, 400)
        response = await self.client.post("/api/runs", json={**config, "dataset_id": "public:open-swe", "request_count": 3})
        self.assertEqual(response.status, 400)
        self.assertIn("不能通过重复", (await response.json())["error"])
        self.assertEqual(self.payloads, [])


if __name__ == "__main__":
    unittest.main()
