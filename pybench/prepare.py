"""Download complete original source partitions; adapt selected records when testing.

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
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

from pybench.workloads import CATALOG

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


def prepare(root: Path, key: str, source: Path | None = None, manifest_path: Path | None = None, endpoint: str | None = None) -> dict[str, Any]:
    """Keep originals and their manifest. Do not tokenize or convert the corpus."""
    import shutil
    from pybench.raw_workloads import manifest_path as source_path, source_info
    from pybench.sources import source_manifest, download_source, file_hash
    if key not in CATALOG:
        raise ValueError("未知公开数据集")
    if source:
        source = source.resolve()
        directory = root / "datasets" / "raw" / key
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / source.name
        if source != dest.resolve():
            shutil.copyfile(source, dest)
        if dest.suffix == ".parquet":
            import pyarrow.parquet as pq
            total = pq.ParquetFile(dest).metadata.num_rows
            format = "parquet"
        else:
            total = sum(1 for _ in local_rows(dest))
            format = "jsonl.bz2" if dest.suffix == ".bz2" else ("jsonl" if ".jsonl" in dest.name else "json")
        manifest = {"key": key, "scope": source.name, "format": format, "expected_rows": total,
                    "files": [{"filename": dest.name, "size": dest.stat().st_size, "sha256": file_hash(dest)}]}
        source_path(root, key).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        cached = source_path(root, key)
        path = manifest_path or cached
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else source_manifest(key)
        if manifest["key"] != key:
            raise ValueError("来源清单与数据集不匹配")
        manifest = download_source(root, key, manifest, endpoint)
        if not manifest.get("expected_rows"):
            # Older manifests for these exact public files predate row counts.
            manifest["expected_rows"] = sum(1 for spec in manifest["files"] for _ in
                                            local_rows(cached.parent / spec["filename"]))
            cached.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info = source_info(root, key)
    if not info:
        raise ValueError("完整原始文件尚未准备")
    return info


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[*CATALOG, "all"])
    parser.add_argument("--input", type=Path, help="导入完整本地原始 JSON/JSONL/JSONL.bz2/Parquet")
    parser.add_argument("--manifest", type=Path, help="官方完整文件清单")
    parser.add_argument("--endpoint", help="HF 文件下载镜像，例如 https://hf-mirror.com")
    args = parser.parse_args()
    if (args.input or args.manifest) and args.dataset == "all":
        parser.error("--input / --manifest 需要指定单个数据集")
    failed = False
    for key in CATALOG if args.dataset == "all" else [args.dataset]:
        try:
            info = prepare(ROOT, key, source=args.input, manifest_path=args.manifest, endpoint=args.endpoint)
            print(f"{key}: {info['source_total']} original records ready; no full token/index build", flush=True)
        except Exception as exc:
            failed = True
            print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
