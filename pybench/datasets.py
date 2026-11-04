"""Load the two offline LongBench datasets for inference benchmarks."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import ijson


INDEX_FILES = {
    "local-longbench": "longbench-token-counts.json",
    "local-longbench-v2": "longbench-v2-token-counts.json",
    "local-longbench-v2-trunc": "longbench-v2-token-counts.json",
}
TRUNCATE_KINDS = {"local-longbench-v2-trunc": "local-longbench-v2"}
# K3 token counts can run ~13% below the deployed DeepSeek tokenizer on the same
# text. Keep 16% headroom so truncated prompts still fit the target model's 1M window.
TRUNCATE_RATIO = 0.84
LENGTH_BOUNDS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
LENGTH_LABELS = ["<1K", "1K–2K", "2K–4K", "4K–8K", "8K–16K", "16K–32K", "32K–64K", "64K–128K", "128K–256K", "256K–512K", "512K–1M", "≥1M"]


def token_counts(kind: str, root: Path) -> list[int]:
    filename = INDEX_FILES.get(kind)
    if filename is None:
        raise ValueError("未知本地数据集")
    return json.loads((root / "datasets" / filename).read_text(encoding="utf-8"))["counts"]


def length_distribution(counts: list[int]) -> list[dict[str, Any]]:
    bins = [0] * len(LENGTH_LABELS)
    for count in counts:
        bins[bisect_right(LENGTH_BOUNDS, count)] += 1
    return [{"range": label, "count": count, "percentage": round(count / len(counts) * 100, 2) if counts else 0}
            for label, count in zip(LENGTH_LABELS, bins)]


def prompt_budget(context_limit: int, output_limit: int) -> int:
    limit = context_limit - output_limit - 512
    if limit < 1024:
        raise ValueError("上下文上限需大于输出上限并留出 512 token 模板余量")
    return limit


def eligible_count(kind: str, root: Path, context_limit: int, output_limit: int, minimum: int = 0) -> int:
    limit = prompt_budget(context_limit, output_limit)
    counts = token_counts(kind, root)
    if kind in TRUNCATE_KINDS:
        return len(counts)
    count = sum(minimum <= length <= limit for length in counts)
    if count == 0:
        raise ValueError("当前上下文上限下没有可用样本")
    return count


def truncate_context(context: str, count: int, prompt_limit: int) -> str:
    """Keep a proportional head of the context; questions and choices stay intact."""
    target = max(1, int(prompt_limit * TRUNCATE_RATIO))
    keep = max(1, int(len(context) * target / max(1, count)))
    return context[:keep]


def load_local_items(kind: str, root: Path, context_limit: int, output_limit: int, minimum: int = 0) -> list[dict[str, Any]]:
    # The tokenizer counts omit the chat template. Keep a small reserve for it.
    prompt_limit = prompt_budget(context_limit, output_limit)
    truncate = kind in TRUNCATE_KINDS
    kind = TRUNCATE_KINDS.get(kind, kind)

    items: list[dict[str, Any]] = []
    if kind == "local-longbench":
        directory = root / "datasets" / "raw" / "longbench" / "extracted" / "data"
        index_file = root / "datasets" / "longbench-token-counts.json"
        if not directory.is_dir() or not index_file.is_file():
            raise ValueError("LongBench 文件或 token 索引不存在")
        counts = token_counts(kind, root)
        index = 0
        for file in sorted(directory.glob("*.jsonl")):
            if file.stem.endswith("_e"):
                continue
            with file.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if index >= len(counts):
                        raise ValueError("LongBench token 索引与原始文件不匹配")
                    text = f"{row.get('context') or ''}\n{row.get('input') or ''}".strip()
                    if text and minimum <= counts[index] <= prompt_limit:
                        items.append({"index": index, "messages": [{"role": "user", "content": text}]})
                    index += 1
        if index != len(counts):
            raise ValueError("LongBench token 索引与原始文件不匹配")
    elif kind == "local-longbench-v2":
        source = root / "datasets" / "raw" / "longbench-v2" / "data.json"
        index_file = root / "datasets" / "longbench-v2-token-counts.json"
        if not source.is_file() or not index_file.is_file():
            raise ValueError("LongBench v2 文件或 token 索引不存在")
        counts = token_counts(kind, root)
        row_count = 0
        with source.open("rb") as stream:
            for index, row in enumerate(ijson.items(stream, "item")):
                row_count += 1
                if index >= len(counts):
                    raise ValueError("LongBench v2 token 索引与原始文件不匹配")
                if counts[index] < minimum:
                    continue
                if not truncate and counts[index] > prompt_limit:
                    continue
                context = row.get('context') or ''
                if truncate and counts[index] > prompt_limit * TRUNCATE_RATIO:
                    context = truncate_context(context, counts[index], prompt_limit)
                choices = "\n".join(f"{letter}. {row.get(f'choice_{letter}', '')}" for letter in "ABCD")
                text = f"{context}\n\n{row.get('question') or ''}\n{choices}\n请从 A、B、C、D 中选择正确答案。"
                items.append({"index": index, "messages": [{"role": "user", "content": text}]})
        if row_count != len(counts):
            raise ValueError("LongBench v2 token 索引与原始文件不匹配")
    else:
        raise ValueError("未知本地数据集")
    if not items:
        raise ValueError("当前上下文上限下没有可用样本")
    return items
