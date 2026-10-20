"""Download bounded public samples and prepare reproducible performance workloads.

python -m pybench.prepare all --limit 100 --max-turns 16
No dataset-provided code or recorded tool command is executed.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

from pybench.workloads import CATALOG, write_snapshot

ROOT = Path(__file__).resolve().parent.parent


def open_url(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CloudBench/1.0"}), timeout=120)


def json_url(url: str) -> Any:
    with open_url(url) as response:
        return json.load(response)


class Counter:
    def __init__(self, root: Path, tokenizer: str):
        import tiktoken
        if tokenizer == "kimi-k3":
            directory = root / "datasets" / "tokenizers" / "kimi-k3"
            directory.mkdir(parents=True, exist_ok=True)
            for filename in ("tiktoken.model", "tokenization_kimi.py"):
                dest = directory / filename
                if not dest.is_file():
                    with open_url(f"https://huggingface.co/moonshotai/Kimi-K3/resolve/main/{filename}") as response:
                        dest.write_bytes(response.read())
            # Read literal regex fragments; never import/execute downloaded Python.
            tree = ast.parse((directory / "tokenization_kimi.py").read_text(encoding="utf-8"))
            fragments = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in node.targets):
                    if isinstance(node.value, ast.Call) and node.value.args:
                        fragments = ast.literal_eval(node.value.args[0])
                        break
            if not isinstance(fragments, list) or not all(isinstance(x, str) for x in fragments):
                raise ValueError("无法读取 Kimi 分词规则")
            from tiktoken.load import load_tiktoken_bpe
            self.encoding = tiktoken.Encoding(name="cloudbench-kimi-k3", pat_str="|".join(fragments),
                mergeable_ranks=load_tiktoken_bpe(str(directory / "tiktoken.model")), special_tokens={})
            self.label = "Kimi K3 正文/工具定义 token（不含聊天模板）"
            self.fingerprint = hashlib.sha256((directory / "tiktoken.model").read_bytes()).hexdigest()
        else:
            self.encoding = tiktoken.get_encoding(tokenizer)
            self.label = f"{tokenizer} 参考 token（不含聊天模板）"
            self.fingerprint = None

    def __call__(self, turn: dict[str, Any]) -> int:
        text = json.dumps({k: turn[k] for k in ("messages", "tools") if k in turn}, ensure_ascii=False)
        return len(self.encoding.encode_ordinary(text))


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if any(isinstance(x, dict) and x.get("type") not in {"text", "input_text", "output_text"} for x in value):
            raise ValueError("当前仅转换纯文本轨迹")
        return "\n".join(x.get("text", "") for x in value if isinstance(x, dict))
    return "" if value is None else str(value)


def agent_item(row: dict[str, Any], index: int, max_turns: int) -> dict[str, Any]:
    messages = row.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)
    tools = [json.loads(t) if isinstance(t, str) else t for t in row.get("tools", [])]
    inferred_tools = []
    for original in messages:
        for call in original.get("tool_calls") or []:
            name = call["function"]["name"]
            if not any(t.get("function", {}).get("name") == name for t in tools):
                tools.append({"type": "function", "function": {"name": name,
                    "description": "Recorded trajectory tool", "parameters": {"type": "object", "additionalProperties": True}}})
                inferred_tools.append(name)
    history, turns, pending_calls = [], [], []
    for original in messages:
        role = original.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        msg = {"role": role, "content": text_content(original.get("content"))}
        if role == "assistant":
            if pending_calls:
                raise ValueError("原始工具返回不完整")
            if history and history[-1]["role"] in {"user", "tool"}:
                turns.append({"messages": list(history), **({"tools": list(tools)} if tools else {})})
                if len(turns) >= max_turns:
                    break
            calls = []
            for call in original.get("tool_calls") or []:
                fn = dict(call["function"])
                if not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                calls.append({"id": call["id"], "type": "function", "function": fn})
            if calls:
                msg["tool_calls"] = calls
                pending_calls = [call["id"] for call in calls]
            if original.get("reasoning_content"):
                msg["reasoning_content"] = original["reasoning_content"]
        if role == "tool":
            ids = original.get("tool_call_ids") or []
            call_id = original.get("tool_call_id") or (ids[0] if len(ids) == 1 else None)
            if not call_id and pending_calls:
                # Some Open-SWE exports omit IDs on tool messages; observations
                # follow the recorded tool-call order. Preserve this mapping.
                call_id = pending_calls[0]
            if not call_id:
                raise ValueError("工具返回缺少唯一 tool_call_id")
            if call_id not in pending_calls:
                raise ValueError("工具返回与原始调用不匹配")
            pending_calls.remove(call_id)
            msg["tool_call_id"] = call_id
        history.append(msg)
    if not turns:
        raise ValueError("轨迹没有可回放的 assistant 轮次")
    # Include the complete tool definitions in every prefix, including turn 1.
    for turn in turns:
        if tools:
            turn["tools"] = tools
    return {"id": str(row.get("trajectory_id") or row.get("traj_id") or row.get("instance_id") or index),
            "turns": turns, "metadata": {"resolved": row.get("resolved"), "replay": "recorded_prefixes", "inferred_tools": inferred_tools}}


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain_html(value: str) -> str:
    parser = PlainHTML()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def rag_item(key: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    if key == "hotpotqa":
        question = row["question"]
        context = row["context"]
        if isinstance(context, dict):
            context = zip(context["title"], context["sentences"])
        docs = [f"[{title}]\n{''.join(sentences)}" for title, sentences in context]
        reference = row.get("answer")
    elif key == "crag":
        question = row["query"]
        docs = [f"[{page.get('page_name', '')}]\n{plain_html(page.get('page_result') or page.get('page_snippet', ''))}"
                for page in row.get("search_results", [])]
        reference = row.get("answer")
    else:
        task = row.get("task", "")
        if task == "event_summary":
            question, content, reference = "请根据材料概括事件。", row.get("text"), row.get("summary")
        elif task == "continuing_writing":
            question, content, reference = "请接着材料中的新闻开头续写。", row.get("beginning"), row.get("continuing")
        elif task == "hallu_modified":
            question = "请根据材料纠正以下续写中的事实错误：\n" + str(row.get("hallucinatedContinuation") or "")
            content = [row.get("newsBeginning", ""), row.get("newsRemainder", "")]
            reference = row.get("realContinuation")
        elif task.startswith("questanswer_"):
            question = row.get("questions")
            content = [row[name] for name in ("news1", "news2", "news3") if row.get(name)]
            reference = row.get("answers")
        else:
            question = row.get("question") or row.get("query") or row.get("instruction")
            content = row.get("documents") or row.get("context") or row.get("news") or row.get("text")
            reference = row.get("answer") or row.get("output")
        if not question or not content:
            raise ValueError("CRUD 样本缺少问题或关联材料")
        if isinstance(content, list):
            docs = [text_content(x) if not isinstance(x, dict) else str(x.get("content") or x.get("text") or x) for x in content]
        else:
            docs = [str(content)]
    if not any(doc.strip() for doc in docs):
        raise ValueError("RAG 样本缺少材料")
    prompt = "请根据以下材料回答问题；材料不足时明确说明，不编造信息。\n\n材料：\n" + "\n\n".join(docs) + "\n\n问题：\n" + str(question)
    identity = str(row.get("_id") or row.get("interaction_id") or row.get("ID") or index)
    if key == "crud-rag":
        identity = f"{row.get('task', 'generic')}:{identity}"
    return {"id": identity,
            "turns": [{"messages": [{"role": "user", "content": prompt}]}],
            "reference_answer": reference, "metadata": {"task": row.get("task"), "document_count": len(docs)}}


def local_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if ".jsonl" in path.name:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            raw = json.load(stream)
            if isinstance(raw, list):
                yield from raw
            elif isinstance(raw, dict):
                for task, rows in raw.items():
                    if isinstance(rows, list):
                        for row in rows:
                            yield {**row, "task": task}


def remote_rows(key: str, limit: int, offset: int) -> Iterator[dict[str, Any]]:
    if key in {"open-swe", "swe-smith"}:
        dataset, config, split = ("nvidia/Open-SWE-Traces", "v1.2", "minisweagent") if key == "open-swe" else ("SWE-bench/SWE-smith-trajectories", "default", "tool")
        # Viewer pages are bounded. Persisted normalized snapshots do not depend on the viewer at runtime.
        for start in range(offset, offset + limit, 10):
            query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": start, "length": min(10, offset + limit - start)})
            page = json_url("https://datasets-server.huggingface.co/rows?" + query)
            if not page.get("rows"):
                return
            for record in page["rows"]:
                if record.get("truncated_cells"):
                    raise ValueError("HF Viewer 返回截断字段，请用 --input 提供完整原始 JSON/JSONL")
                yield record["row"]
        return
    if key == "hotpotqa":
        for start in range(offset, offset + limit, 100):
            query = urllib.parse.urlencode({"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation", "offset": start, "length": min(100, offset + limit - start)})
            for record in json_url("https://datasets-server.huggingface.co/rows?" + query)["rows"]:
                if record.get("truncated_cells"):
                    raise ValueError("HotpotQA 返回截断字段，请提供完整原始数据")
                yield record["row"]
        return
    if key == "crag":
        url = "https://media.githubusercontent.com/media/facebookresearch/CRAG/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
        with open_url(url) as response, bz2.BZ2File(response) as stream:
            for index, line in enumerate(stream):
                if index >= offset + limit:
                    break
                if index >= offset:
                    yield json.loads(line)
        return
    url = "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json"
    cache = ROOT / "datasets" / "raw" / "crud-rag" / "split_merged.json"
    if not cache.is_file():
        import shutil
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".download")
        with open_url(url) as response, temp.open("wb") as output:
            shutil.copyfileobj(response, output)
        # A failed/corrupt download must not become the cache.
        json.loads(temp.read_text(encoding="utf-8"))
        temp.replace(cache)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    # Interleave task families instead of taking only summaries from the first key.
    index = 0
    for position in range(max(len(rows) for rows in raw.values())):
        for task, rows in raw.items():
            if position >= len(rows):
                continue
            if index >= offset + limit:
                return
            if index >= offset:
                yield {**rows[position], "task": task}
            index += 1


def prepare(root: Path, key: str, limit: int = 100, max_turns: int = 16, tokenizer: str = "kimi-k3", source: Path | None = None, offset: int = 0) -> dict[str, Any]:
    if key not in CATALOG or not 1 <= limit <= 5000 or not 1 <= max_turns <= 128 or offset < 0:
        raise ValueError("无效的数据准备参数")
    counter = Counter(root, tokenizer)
    if source:
        from itertools import islice
        rows = islice(local_rows(source), offset, offset + limit)
    else:
        rows = remote_rows(key, limit, offset)
    items, skipped = [], []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        try:
            item = agent_item(row, index + offset, max_turns) if CATALOG[key]["category"] == "agent" else rag_item(key, row, index + offset)
            for turn in item["turns"]:
                turn["input_tokens"] = counter(turn)
            items.append(item)
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append({"row": index + offset, "reason": str(exc)[:160]})
    if not items:
        raise ValueError(f"没有转换成功的样本；前3条错误：{skipped[:3]}")
    return write_snapshot(root, key, items, {"source": CATALOG[key]["source"], "source_file": source.name if source else None,
        "prepared_at": datetime.now(timezone.utc).isoformat(), "sampling": "task-interleaved bounded slice" if key == "crud-rag" and not source else "source order bounded slice",
        "offset": offset, "requested_rows": limit, "max_turns": max_turns,
        "tokenizer": tokenizer, "tokenizer_sha256": counter.fingerprint, "length_label": counter.label,
        "skipped_count": len(skipped), "skipped": skipped, "reference_answers_sent": False})


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[*CATALOG, "all"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tokenizer", default="kimi-k3", help="kimi-k3 或 tiktoken encoding 名称")
    parser.add_argument("--input", type=Path, help="本地原始 JSON/JSONL/JSONL.bz2")
    args = parser.parse_args()
    if args.input and args.dataset == "all":
        parser.error("--input 需要指定单个数据集")
    failed = False
    for key in CATALOG if args.dataset == "all" else [args.dataset]:
        try:
            info = prepare(ROOT, key, args.limit, args.max_turns, args.tokenizer, args.input, args.offset)
            print(f"{key}: {info['item_count']} samples/sessions, {info['request_total']} requests, skipped={info['skipped_count']}", flush=True)
        except Exception as exc:
            failed = True
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
