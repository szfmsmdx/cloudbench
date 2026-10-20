"""Guard request equivalence and preparation boundaries when removing bulk work."""
import asyncio
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pybench.prepare import agent_item, prepare, rag_item
from pybench.workloads import IndexedWriter, iter_workload_turns, select_for_run
from tests import test_workloads
from tests.test_workloads import trajectory


class FixtureCounter:
    calls = 0
    fingerprint, label = 'fixture', 'fixture tokens'

    def __init__(self, *args):
        pass

    def count_item(self, item):
        type(self).calls += 1
        for index, turn in enumerate(item['turns']):
            turn['input_tokens'] = 100 + index * 400


class OnDemandTests(unittest.TestCase):
    def test_same_original_records_produce_identical_requests_for_all_five_datasets(self):
        sources = {
            'open-swe': trajectory(),
            'swe-smith': {**trajectory(), 'messages': json.dumps(trajectory()['messages'])},
            'hotpotqa': {'_id': 'rag', 'question': 'question', 'context': [['support', ['evidence']], ['distractor', ['noise']]], 'answer': 'gold'},
            'crag': {'interaction_id': 'rag', 'query': 'question', 'search_results': [{'page_name': 'p', 'page_result': '<p>evidence</p><script>hidden()</script>'}], 'answer': 'gold'},
            'crud-rag': {'task': 'questanswer_2docs', 'ID': 'rag', 'questions': 'question', 'news1': 'article1', 'news2': 'article2', 'answers': 'gold', 'thoughts': 'secret'},
        }
        for key, source in sources.items():
            with self.subTest(dataset=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old, new = root / 'old', root / 'new'
                raw_file = root / 'original.jsonl'
                records = [{**source, 'trajectory_id': f'trace-{i}', '_id': str(i)} for i in range(137)]
                raw_file.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
                original_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
                writer = IndexedWriter(old, key)
                for i, row in enumerate(records):
                    item = agent_item(row, i, compact=True) if key in {'open-swe', 'swe-smith'} else rag_item(key, row, i)
                    FixtureCounter().count_item(item)
                    writer.append(row, item)
                writer.finish({'complete': True})
                with patch('pybench.prepare.Counter', side_effect=AssertionError('No full-source tokenizing')):
                    prepare(new, key, source=raw_file)
                FixtureCounter.calls = 0
                with patch('pybench.prepare.Counter', FixtureCounter):
                    selected = select_for_run(new, 'public:' + key, 4096, 128, 0, 101, 42)
                expected = select_for_run(old, 'public:' + key, 4096, 128, 0, 101, 42)
                self.assertEqual(FixtureCounter.calls, 101)
                self.assertEqual([u['index'] for u in selected], [u['index'] for u in expected])
                self.assertEqual(len({u['index'] for u in selected}), 101)
                for before, after in zip(expected, selected):
                    self.assertEqual(list(iter_workload_turns(before)), list(iter_workload_turns(after)))
                self.assertEqual(hashlib.sha256(raw_file.read_bytes()).hexdigest(), original_hash)
                self.assertEqual(hashlib.sha256((new / 'datasets/raw' / key / raw_file.name).read_bytes()).hexdigest(), original_hash)
                self.assertFalse((new / 'datasets/prepared' / f'{key}.sqlite3').exists())
                with patch('pybench.prepare.Counter', FixtureCounter):
                    with self.assertRaisesRegex(ValueError, '不能通过重复'):
                        select_for_run(new, 'public:' + key, 4096, 128, 0, 138, 42)

    def test_length_filter_preserves_history_and_does_not_crop_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'original.jsonl'
            file.write_text(json.dumps(trajectory()) + '\n', encoding='utf-8')
            prepare(root, 'open-swe', source=file)
            with patch('pybench.prepare.Counter', FixtureCounter):
                selected = select_for_run(root, 'public:open-swe', 2048, 128, 400, 1, 42)
                self.assertEqual(len(selected[0]['turns']), 2)
                first, second = list(iter_workload_turns(selected[0]))
                self.assertEqual(second['messages'][:len(first['messages'])], first['messages'])
            class OversizeCounter(FixtureCounter):
                def count_item(self, item):
                    for turn in item['turns']:
                        turn['input_tokens'] = 2000
            with patch('pybench.prepare.Counter', OversizeCounter):
                with self.assertRaisesRegex(ValueError, '不能通过重复'):
                    select_for_run(root, 'public:open-swe', 2048, 128, 0, 1, 42)

    def test_parquet_direct_read_matches_previous_adapter_across_row_groups(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'original.parquet'
            rows = [{**trajectory(), 'trajectory_id': str(i), 'patch': 'auxiliary-not-for-request'} for i in range(137)]
            pq.write_table(pa.Table.from_pylist(rows), file, row_group_size=17)
            prepare(root, 'open-swe', source=file)
            FixtureCounter.calls = 0
            with patch('pybench.prepare.Counter', FixtureCounter):
                selected = select_for_run(root, 'public:open-swe', 4096, 128, 0, 7, 42)
            self.assertEqual(FixtureCounter.calls, 7)
            for unit in selected:
                expected = agent_item(rows[unit['index']], unit['index'], compact=True)
                FixtureCounter().count_item(expected)
                self.assertEqual(list(iter_workload_turns(unit)), list(iter_workload_turns(expected)))


class OnDemandWorkerTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_workloads.PublicWorkloadAPITests.asyncSetUp
    asyncTearDown = test_workloads.PublicWorkloadAPITests.asyncTearDown
    async def test_slow_input_preparation_finishes_before_measured_run_and_preserves_usage(self):
        source = self.root / 'original.jsonl'
        source.write_text(json.dumps(trajectory()) + '\n', encoding='utf-8')
        prepare(self.root, 'swe-smith', source=source)
        original_select = select_for_run
        preparation_done = []

        def delayed(*args):
            time.sleep(.4)
            with patch('pybench.prepare.Counter', FixtureCounter):
                selected = original_select(*args)
            preparation_done.append(time.time())
            return selected

        with patch('server.select_for_run', delayed):
            response = await self.client.post('/api/runs', json={'target_id': self.target['id'],
                'dataset_id': 'public:swe-smith', 'mode': 'sync', 'request_count': 1,
                'max_tokens': 128, 'request_timeout': 30})
        self.assertEqual(response.status, 202, await response.text())
        run = await response.json()
        for _ in range(200):
            result = await (await self.client.get(f"/api/runs/{run['id']}")).json()
            if result['run']['status'] != 'running':
                break
            await asyncio.sleep(.05)
        self.assertEqual(result['run']['status'], 'completed', result['run']['error'])
        from datetime import datetime
        measured_start = datetime.fromisoformat(result['run']['started_at'].replace('Z', '+00:00')).timestamp()
        self.assertGreaterEqual(measured_start + .001, preparation_done[0])
        self.assertEqual(result['summary']['success'], 2)
        self.assertEqual([s['prompt_tokens'] for s in result['samples']], [100, 100])
        self.assertEqual([s['completion_tokens'] for s in result['samples']], [10, 10])
        self.assertEqual(self.payloads[1]['messages'][-1]['content'], 'recorded observation')
        self.assertNotIn('live-id', json.dumps(self.payloads[1]))


if __name__ == '__main__':
    unittest.main()
