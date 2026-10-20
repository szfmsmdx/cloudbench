"""Read original public records on demand, before benchmark timing starts."""
from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from pathlib import Path

from pybench.datasets import prompt_budget


def manifest_path(root, key):
    return root / 'datasets' / 'raw' / key / 'source-manifest.json'


def source_info(root: Path, key: str):
    path = manifest_path(root, key)
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('key') != key or not manifest.get('expected_rows'):
        return None
    directory = path.parent
    for spec in manifest['files']:
        file = directory / spec['filename']
        if not file.is_file() or (spec.get('size') and file.stat().st_size != spec['size']):
            return None
    if manifest['format'] == 'parquet':
        import pyarrow.parquet as pq
        total = 0
        for spec in manifest['files']:
            rows = pq.ParquetFile(directory / spec['filename'], pre_buffer=False).metadata.num_rows
            if spec.get('num_rows') is not None and spec['num_rows'] != rows:
                raise ValueError('原始文件行数已变化')
            spec['num_rows'] = rows
            total += rows
        if total != manifest['expected_rows']:
            raise ValueError('原始文件总行数与来源清单不一致')
    signature = [{**spec, 'mtime_ns': (directory / spec['filename']).stat().st_mtime_ns}
                 for spec in manifest['files']]
    fingerprint = hashlib.sha256(json.dumps({**manifest, 'files': signature}, sort_keys=True).encode()).hexdigest()
    return {'format': 'raw-source-v1', 'source_manifest': manifest, 'sha256': fingerprint,
            'scope': manifest['scope'], 'source_total': manifest['expected_rows'],
            'raw_count': manifest['expected_rows'], 'item_count': manifest['expected_rows'],
            'complete': True, 'sampling': 'on-demand original source', 'request_total': None,
            'tokenizer': 'kimi-k3', 'length_label': '本次选中数据按原有 Kimi K3 口径筛选；全量分布未预计算',
            'distribution': [], 'raw_records_preserved': True}


def selected_rows(root: Path, key: str, manifest, indices):
    """Yield (global row index, original replay fields), preserving requested order."""
    directory = manifest_path(root, key).parent
    wanted = set(indices)
    found = {}
    if manifest['format'] == 'parquet':
        import pyarrow.parquet as pq
        file_ends, total = [], 0
        for spec in manifest['files']:
            total += spec['num_rows']
            file_ends.append(total)
        by_file = {}
        for index in indices:
            file_index = bisect_right(file_ends, index)
            by_file.setdefault(file_index, []).append(index)
        for file_index, requested in by_file.items():
            spec = manifest['files'][file_index]
            file = pq.ParquetFile(directory / spec['filename'], pre_buffer=False)
            columns = file.schema_arrow.names
            if key in {'open-swe', 'swe-smith'}:
                fields = {'messages', 'tools', 'trajectory_id', 'traj_id', 'instance_id', 'resolved', 'model'}
                columns = [name for name in columns if name in fields]
            start = file_ends[file_index - 1] if file_index else 0
            group_ends, group_total = [], 0
            for group in range(file.metadata.num_row_groups):
                group_total += file.metadata.row_group(group).num_rows
                group_ends.append(group_total)
            by_group = {}
            for index in requested:
                local = index - start
                group = bisect_right(group_ends, local)
                by_group.setdefault(group, []).append((index, local - (group_ends[group - 1] if group else 0)))
            for group, rows in by_group.items():
                table = file.read_row_group(group, columns=columns, use_threads=False)
                for index, local in rows:
                    found[index] = table.slice(local, 1).to_pylist()[0]
    else:
        from pybench.prepare import local_rows
        offset = 0
        last = max(wanted, default=-1)
        for spec in manifest['files']:
            for row in local_rows(directory / spec['filename']):
                if offset in wanted:
                    found[offset] = row
                offset += 1
                if offset > last:
                    break
            if offset > last:
                break
    if len(found) != len(wanted):
        raise ValueError('原始文件缺少选中的记录；未发起模型请求')
    for index in indices:
        yield index, found.pop(index)


def select_raw(root: Path, key: str, context: int, output: int, minimum: int, count, seed: int):
    from pybench.prepare import Counter, agent_item, rag_item
    from pybench.workloads import CATALOG
    info = source_info(root, key)
    if info is None:
        raise ValueError('完整原始数据尚未准备')
    total = info['source_total']
    if count is not None and count > total:
        raise ValueError(f'来源只有 {total} 条，不能通过重复样本补足 {count} 条')
    order = list(range(total))
    random.Random(seed).shuffle(order)
    counter = Counter(root, 'kimi-k3')
    limit = prompt_budget(context, output)
    selected = []
    failures = []
    # Only adapt/tokenize candidates needed by this run. Narrow length filters
    # may require checking more records; never substitute or repeat records.
    try:
        position = 0
        while position < total and (count is None or len(selected) < count):
            chunk = 32 if info['source_manifest']['format'] == 'parquet' else total
            size = min(chunk, total - position, (count - len(selected)) if count is not None else chunk)
            batch = order[position:position + size]
            position += size
            for index, raw in selected_rows(root, key, info['source_manifest'], batch):
                try:
                    item = agent_item(raw, index, compact=True) if CATALOG[key]['category'] == 'agent' else rag_item(key, raw, index)
                    counter.count_item(item)
                except (ValueError, KeyError, TypeError) as exc:
                    if len(failures) < 3:
                        failures.append(f'{index}: {str(exc)[:120]}')
                    continue
                turns = []
                for turn in item['turns']:
                    if turn['input_tokens'] > limit:
                        break
                    turns.append(turn)
                if turns and max(turn['input_tokens'] for turn in turns) >= minimum:
                    item.pop('reference_answer', None)
                    selected.append({**item, 'index': index, 'turns': turns})
        if not selected and count is None:
            raise ValueError('所选长度范围内没有可用会话/样本；未发起模型请求')
        if count is not None and len(selected) != count:
            raise ValueError(f'所选长度范围只有 {len(selected)} 个可用样本/会话，不能通过重复样本补足 {count} 条。'
                             + ('未适配记录示例：' + '; '.join(failures) if failures else ''))
        current = source_info(root, key)
        if current is None or current['sha256'] != info['sha256']:
            raise ValueError('数据在准备期间已变化；未发起模型请求')
        return selected
    finally:
        if hasattr(counter, 'message_pool'):
            counter.message_pool.shutdown()
