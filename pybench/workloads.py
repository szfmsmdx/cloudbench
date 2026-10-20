"""Prepared public workloads: immutable conversations and retrieved-context prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pybench.datasets import length_distribution, prompt_budget

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
    return base.with_suffix(".jsonl"), base.with_suffix(".meta.json")


def catalog(root: Path) -> list[dict[str, Any]]:
    result = []
    for key, spec in CATALOG.items():
        data, meta = paths(root, key)
        info = json.loads(meta.read_text(encoding="utf-8")) if data.is_file() and meta.is_file() else {}
        result.append({**spec, **info, "id": f"public:{key}", "kind": f"public:{key}",
                       "ready": bool(info), "item_count": info.get("item_count", 0),
                       "distribution": info.get("distribution", []),
                       "length_label": info.get("length_label", "尚未准备"),
                       "unit": "会话" if spec["category"] == "agent" else "样本"})
    return result


def load_workloads(root: Path, kind: str, context: int, output: int, minimum: int = 0) -> list[dict[str, Any]]:
    data, meta = paths(root, kind.removeprefix("public:"))
    if not data.is_file() or not meta.is_file():
        raise ValueError("数据集尚未准备，请先运行 README 中的数据准备命令")
    info = json.loads(meta.read_text(encoding="utf-8"))
    if hashlib.sha256(data.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("数据快照已变化，请重新准备后运行")
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
    # Default: every eligible session once. Explicit counts may cycle the pool.
    return [order[i % len(order)] for i in range(count if count is not None else len(order))]


def write_snapshot(root: Path, key: str, items: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    data, meta = paths(root, key)
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
