"""Prepare complete source partitions; sampling requires an explicit --limit.

python -m pybench.prepare all
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
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

from pybench.workloads import CATALOG, IndexedWriter

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
        self.message_pool = ThreadPoolExecutor(max_workers=2)

    def __call__(self, turn: dict[str, Any]) -> int:
        text = json.dumps({k: turn[k] for k in ("messages", "tools") if k in turn}, ensure_ascii=False)
        return len(self.encoding.encode_ordinary(text))

    def count_item(self, item):
        turns = item["turns"]
        # Agent prefixes share history. Tokenize each message once rather than
        # repeatedly encoding every earlier message in a long conversation.
        if len(turns) == 1 and "end" not in turns[0]:
            turns[0]["input_tokens"] = self(turns[0])
            return
        history = item["messages"][:turns[-1]["end"]] if "messages" in item else turns[-1]["messages"]
        prefix = [0]
        def message_size(message):
            return len(self.encoding.encode_ordinary(json.dumps(message, ensure_ascii=False))) + 1
        for size in self.message_pool.map(message_size, history):
            prefix.append(prefix[-1] + size)
        tools = item.get("tools") or turns[0].get("tools")
        frame = self({"messages": [], **({"tools": tools} if tools else {})})
        for turn in turns:
            turn["input_tokens"] = frame + prefix[turn["end"] if "end" in turn else len(turn["messages"])]


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if any(isinstance(x, dict) and x.get("type") not in {"text", "input_text", "output_text"} for x in value):
            raise ValueError("当前仅转换纯文本轨迹")
        return "\n".join(x.get("text", "") for x in value if isinstance(x, dict))
    return "" if value is None else str(value)


def agent_item(row: dict[str, Any], index: int, max_turns: int | None = None, compact: bool = False) -> dict[str, Any]:
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
                turns.append({"end": len(history)} if compact else {"messages": list(history), **({"tools": tools} if tools else {})})
                if max_turns is not None and len(turns) >= max_turns:
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
        if tools and not compact:
            turn["tools"] = tools
    return {"id": str(row.get("trajectory_id") or row.get("traj_id") or row.get("instance_id") or index),
            "turns": turns, **({"messages": history, "tools": tools} if compact else {}),
            "metadata": {"resolved": row.get("resolved"), "replay": "recorded_prefixes", "inferred_tools": inferred_tools}}


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


def prepare(root: Path, key: str, limit: int | None = None, max_turns: int | None = None, tokenizer: str = "kimi-k3", source: Path | None = None, offset: int = 0, manifest_path: Path | None = None, endpoint: str | None = None) -> dict[str, Any]:
    if key not in CATALOG or (limit is not None and limit < 1) or (max_turns is not None and max_turns < 1) or offset < 0:
        raise ValueError("无效的数据准备参数")
    progress_file = root / "datasets" / "prepared" / f"{key}.progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_info = {}
    def progress(**values):
        progress_info.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temp = progress_file.with_suffix(".tmp")
        temp.write_text(json.dumps(progress_info, ensure_ascii=False), encoding="utf-8")
        temp.replace(progress_file)

    counter = Counter(root, tokenizer)
    manifest = None
    if source:
        from itertools import islice
        rows = islice(local_rows(source), offset, None if limit is None else offset + limit)
    elif limit is None:
        from pybench.sources import source_manifest, download_source, source_rows
        cached = root / "datasets" / "raw" / key / "source-manifest.json"
        manifest = json.loads((manifest_path or cached).read_text(encoding="utf-8")) if (manifest_path or cached).is_file() else source_manifest(key)
        if manifest["key"] != key:
            raise ValueError("来源清单与数据集不匹配")
        progress(scope=manifest["scope"], expected_rows=manifest.get("expected_rows"), status="downloading")
        manifest = download_source(root, key, manifest, endpoint, progress)
        rows = source_rows(root, key, manifest)
    else:
        rows = remote_rows(key, limit, offset)
    writer = IndexedWriter(root, key)
    skipped, errors = 0, []
    progress(status="indexing", rows=0, accepted=0)
    try:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            try:
                item = agent_item(row, index + offset, max_turns, compact=True) if CATALOG[key]["category"] == "agent" else rag_item(key, row, index + offset)
                counter.count_item(item)
                writer.append(row, item)
            except (ValueError, KeyError, TypeError) as exc:
                skipped += 1
                writer.append(row, error=str(exc)[:300])
                if len(errors) < 20:
                    errors.append({"row": index + offset, "reason": str(exc)[:160]})
            if writer.rows % 100 == 0:
                progress(status="indexing", rows=writer.rows, accepted=writer.accepted)
                print(f"{key}: indexed {writer.rows} raw records, {writer.accepted} usable", flush=True)
        if not writer.accepted:
            raise ValueError(f"没有转换成功的样本；前3条错误：{errors[:3]}")
        if manifest and manifest.get("expected_rows") is not None and writer.rows != manifest["expected_rows"]:
            raise ValueError(f"来源记录数不完整：{writer.rows}/{manifest['expected_rows']}；原快照未替换")
        info = writer.finish({"source": CATALOG[key]["source"], "source_file": source.name if source else None,
            "scope": manifest["scope"] if manifest else (str(source.name) if source else "显式抽样"),
            "source_manifest": manifest, "source_total": manifest.get("expected_rows", writer.rows) if manifest else writer.rows,
            "complete": limit is None and max_turns is None and offset == 0,
            "prepared_at": datetime.now(timezone.utc).isoformat(), "sampling": "complete source partition" if limit is None else "explicit bounded slice",
            "offset": offset, "requested_rows": limit, "max_turns": max_turns,
            "tokenizer": tokenizer, "tokenizer_sha256": counter.fingerprint, "length_label": counter.label + "；多轮按消息累加参考",
            "skipped_count": skipped, "skipped": errors, "raw_records_preserved": True,
            "raw_storage": "完整原始文件保留于 datasets/raw；索引保存回放字段和源记录位置" if manifest else "完整输入记录保留于索引",
            "reference_answers_sent": False})
        progress(status="complete", rows=info["raw_count"], accepted=info["item_count"])
        return info
    except BaseException as exc:
        writer.close()
        progress(status="error", rows=writer.rows, accepted=writer.accepted, error=str(exc)[:300])
        raise


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[*CATALOG, "all"])
    parser.add_argument("--limit", type=int, help="仅在显式指定时抽样；默认导入完整来源")
    parser.add_argument("--max-turns", type=int, help="仅在显式指定时限制会话轮数；默认保留全部轮次")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tokenizer", default="kimi-k3", help="kimi-k3 或 tiktoken encoding 名称")
    parser.add_argument("--input", type=Path, help="本地原始 JSON/JSONL/JSONL.bz2")
    parser.add_argument("--manifest", type=Path, help="官方完整文件清单，可在可访问 HF 的机器生成")
    parser.add_argument("--endpoint", help="HF 文件下载镜像，例如 https://hf-mirror.com")
    args = parser.parse_args()
    if (args.input or args.manifest) and args.dataset == "all":
        parser.error("--input / --manifest 需要指定单个数据集")
    failed = False
    for key in CATALOG if args.dataset == "all" else [args.dataset]:
        try:
            info = prepare(ROOT, key, args.limit, args.max_turns, args.tokenizer, args.input, args.offset, args.manifest, args.endpoint)
            print(f"{key}: {info['item_count']} samples/sessions, {info['request_total']} requests, skipped={info['skipped_count']}", flush=True)
        except Exception as exc:
            failed = True
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
