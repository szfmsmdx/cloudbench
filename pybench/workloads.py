"""Prepared public workloads: immutable conversations and retrieved-context prompts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from bisect import bisect_right
from contextlib import closing
from pathlib import Path
from typing import Any

from pybench.datasets import LENGTH_BOUNDS, LENGTH_LABELS, length_distribution, prompt_budget

_VERIFIED = {}

CATALOG = {
    "open-swe": {"name": "Open-SWE-Traces", "category": "agent", "source": "https://huggingface.co/datasets/nvidia/Open-SWE-Traces", "description": "编码 Agent 固定轨迹；保留工具定义和逐轮历史，仅测性能。"},
    "swe-smith": {"name": "SWE-smith trajectories", "category": "agent", "source": "https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories", "description": "SWE-agent 编码轨迹；逐会话顺序回放。"},
    "hotpotqa": {"name": "HotpotQA", "category": "rag", "source": "https://hotpotqa.github.io/", "description": "问题＋官方 distractor 上下文；包含干扰段落。"},
    "crag": {"name": "CRAG", "category": "rag", "source": "https://github.com/facebookresearch/CRAG", "description": "问题＋清理后的网页检索材料；不包含在线检索耗时。"},
    "crud-rag": {"name": "CRUD-RAG", "category": "rag", "source": "https://github.com/IAAR-Shanghai/CRUD_RAG", "description": "中文任务＋原始关联文档；保留任务类型与参考答案。"},
}


def paths(root: Path, key: str) -> tuple[Path, Path]:
    if key not in CATALOG:
        raise ValueError("未知公开数据集")
    base = root / "datasets" / "prepared" / key
    meta = base.with_suffix(".meta.json")
    compact = meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("format") == "indexed-raw-v1"
    return base.with_suffix(".sqlite3" if compact else ".jsonl"), meta


def catalog(root: Path) -> list[dict[str, Any]]:
    result = []
    for key, spec in CATALOG.items():
        data, meta = paths(root, key)
        info = json.loads(meta.read_text(encoding="utf-8")) if data.is_file() and meta.is_file() else {}
        progress_file = root / "datasets" / "prepared" / f"{key}.progress.json"
        progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.is_file() else None
        result.append({**spec, **info, "id": f"public:{key}", "kind": f"public:{key}", "preparation": progress,
                       "ready": bool(info) and not (progress and progress.get("status") != "complete" and not info.get("complete")),
                       "source_total": info.get("source_total") or (progress or {}).get("expected_rows"),
                       "item_count": info.get("item_count", 0),
                       "distribution": info.get("distribution", []),
                       "length_label": info.get("length_label", "尚未准备"),
                       "unit": "会话" if spec["category"] == "agent" else "样本"})
    return result


def load_workloads(root: Path, kind: str, context: int, output: int, minimum: int = 0) -> list[dict[str, Any]]:
    data, meta = paths(root, kind.removeprefix("public:"))
    if not data.is_file() or not meta.is_file():
        raise ValueError("数据集尚未准备，请先运行 README 中的数据准备命令")
    info = json.loads(meta.read_text(encoding="utf-8"))
    stat = data.stat()
    signature = (str(data), stat.st_size, stat.st_mtime_ns, info["sha256"])
    if signature not in _VERIFIED:
        with data.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _file_hash(stream)
        if digest != info["sha256"]:
            raise ValueError("数据快照已变化，请重新准备后运行")
        if len(_VERIFIED) > 32:
            _VERIFIED.clear()
        _VERIFIED[signature] = True
    if info.get("format") == "indexed-raw-v1":
        result = []
        limit = prompt_budget(context, output)
        with closing(sqlite3.connect(f"file:{data.as_posix()}?mode=ro", uri=True)) as db:
            for index, identity, raw_sizes in db.execute("SELECT idx,source_id,turn_tokens FROM workloads ORDER BY idx"):
                count, longest = 0, 0
                for size in json.loads(raw_sizes):
                    if size > limit:
                        break
                    count += 1
                    longest = max(longest, size)
                if count and longest >= minimum:
                    result.append({"id": identity, "index": index, "turn_count": count, "storage": "indexed-raw-v1"})
        if not result:
            raise ValueError("所选长度范围内没有可用会话/样本")
        return result
    limit = prompt_budget(context, output)
    result = []
    with data.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            # Keep preceding turns even when only later prompts meet the lower bound.
            # Their shared prefix is required for the cache experiment.
            turns = []
            for turn in item["turns"]:
                if turn["input_tokens"] > limit:
                    break
                turns.append(turn)
            if turns and max(t["input_tokens"] for t in turns) >= minimum:
                result.append({**item, "index": index, "turns": turns})
    if not result:
        raise ValueError("所选长度范围内没有可用会话/样本")
    return result


def select_workloads(items: list[dict[str, Any]], count: int | None, seed: int) -> list[dict[str, Any]]:
    import random
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    if count is not None and count > len(order):
        raise ValueError(f"所选长度范围只有 {len(order)} 个可用样本/会话，不能通过重复样本补足 {count} 条")
    return order if count is None else order[:count]


def _file_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def turn_count(unit):
    return unit["turn_count"] if "turn_count" in unit else len(unit["turns"])


def hydrate_workload(root: Path, kind: str, unit: dict[str, Any]) -> dict[str, Any]:
    if unit.get("storage") != "indexed-raw-v1":
        return unit
    from pybench.prepare import agent_item, rag_item
    key = kind.removeprefix("public:")
    data, _ = paths(root, key)
    with closing(sqlite3.connect(f"file:{data.as_posix()}?mode=ro", uri=True)) as db:
        payload, raw_sizes = db.execute("SELECT raw_record,turn_tokens FROM workloads WHERE idx=?", (unit["index"],)).fetchone()
    raw = json.loads(zlib.decompress(payload))
    item = agent_item(raw, unit["index"], None, compact=True) if CATALOG[key]["category"] == "agent" else rag_item(key, raw, unit["index"])
    turns = item["turns"][:unit["turn_count"]]
    for turn, size in zip(turns, json.loads(raw_sizes)):
        turn["input_tokens"] = size
    return {**item, **unit, "turns": turns}


def iter_workload_turns(unit):
    for turn in unit["turns"]:
        if "end" in turn:
            yield {"messages": unit["messages"][:turn["end"]], "input_tokens": turn["input_tokens"],
                   **({"tools": unit["tools"]} if unit.get("tools") else {})}
        else:
            yield turn


class IndexedWriter:
    """Preserve every raw record; load conversation bodies only for selected sessions."""
    def __init__(self, root: Path, key: str):
        self.base = root / "datasets" / "prepared" / key
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.base.with_suffix(".sqlite3.build")
        if self.temp.exists():
            self.temp.unlink()
        self.db = sqlite3.connect(self.temp)
        self.db.execute("CREATE TABLE workloads(idx INTEGER PRIMARY KEY,source_id TEXT,turn_tokens TEXT,raw_record BLOB,error TEXT)")
        self.rows = self.accepted = self.requests = 0
        self.bins = [0] * len(LENGTH_LABELS)

    def append(self, raw, item=None, error=None):
        sizes = [turn["input_tokens"] for turn in item["turns"]] if item else []
        self.db.execute("INSERT INTO workloads VALUES(?,?,?,?,?)", (self.rows, item["id"] if item else str(self.rows),
            json.dumps(sizes, separators=(",", ":")), zlib.compress(json.dumps(raw, ensure_ascii=False).encode(), 3), error))
        self.rows += 1
        self.accepted += bool(sizes)
        self.requests += len(sizes)
        for size in sizes:
            self.bins[bisect_right(LENGTH_BOUNDS, size)] += 1
        if self.rows % 100 == 0:
            self.db.commit()

    def finish(self, metadata):
        self.db.commit()
        self.db.close()
        with self.temp.open("rb") as stream:
            digest = _file_hash(stream)
        info = {**metadata, "format": "indexed-raw-v1", "raw_count": self.rows, "item_count": self.accepted,
            "request_total": self.requests, "sha256": digest, "distribution": [
                {"range": label, "count": count, "percentage": round(count / self.requests * 100, 2) if self.requests else 0}
                for label, count in zip(LENGTH_LABELS, self.bins)]}
        self.temp.replace(self.base.with_suffix(".sqlite3"))
        meta = self.base.with_suffix(".meta.json")
        temp_meta = meta.with_suffix(".tmp")
        temp_meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_meta.replace(meta)
        return info

    def close(self):
        self.db.close()


def write_snapshot(root: Path, key: str, items: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    _, meta = paths(root, key)
    data = root / "datasets" / "prepared" / f"{key}.jsonl"
    if not items:
        raise ValueError("没有转换成功的样本；原快照未改动")
    data.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items)
    counts = [turn["input_tokens"] for item in items for turn in item["turns"]]
    info = {**metadata, "item_count": len(items), "request_total": len(counts),
            "distribution": length_distribution(counts),
            "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    temp = data.with_suffix(".tmp")
    # Hash and persist identical bytes on Windows as well as Unix.
    temp.write_bytes(payload.encode("utf-8"))
    temp.replace(data)
    temp_meta = meta.with_suffix(".tmp")
    temp_meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_meta.replace(meta)
    return info
