"""Run OpenAI-compatible streaming benchmarks from a JSON configuration on stdin.

Each stdout line is one JSON event: start, sample, done, or error. Diagnostics go
to stderr. The browser/API process can persist samples without doing load work.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import math
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from pybench.datasets import load_local_items
from pybench.telemetry import RunTelemetry
from pybench.workloads import hydrate_workload, load_workloads, paths, select_workloads, turn_count


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def usage_from(raw: Any) -> dict[str, Any]:
    usage = raw if isinstance(raw, dict) else {}

    def nonnegative(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    prompt = nonnegative(usage.get("prompt_tokens", usage.get("input_tokens")))
    output = nonnegative(usage.get("completion_tokens", usage.get("output_tokens")))
    cached = nonnegative(
        usage.get("prompt_cache_hit_tokens")
        if usage.get("prompt_cache_hit_tokens") is not None
        else (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    if cached is None:
        cached = nonnegative((usage.get("input_tokens_details") or {}).get("cached_tokens"))
    writes = nonnegative(usage.get("cache_creation_input_tokens"))
    misses = nonnegative(usage.get("prompt_cache_miss_tokens"))
    uncached = misses if misses is not None else (
        max(prompt - cached - (writes or 0), 0) if prompt is not None and cached is not None else None
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "cached_tokens": cached,
        "uncached_tokens": uncached,
        "cache_write_tokens": writes,
        "usage_source": "provider" if prompt is not None or output is not None else "unavailable",
    }


def estimate_cost(usage: dict[str, Any], price: dict[str, Any] | None) -> float | None:
    if not price or usage["prompt_tokens"] is None or usage["completion_tokens"] is None:
        return None
    cached, writes = usage["cached_tokens"], usage["cache_write_tokens"]
    if price.get("cache_read_price") is not None and cached is None:
        return None
    if price.get("cache_write_price") is not None and writes is None:
        return None
    uncached = usage["uncached_tokens"]
    if uncached is None:
        uncached = max(usage["prompt_tokens"] - (cached or 0) - (writes or 0), 0)
    cost = (
        uncached * price["input_price"]
        + (cached or 0) * (price.get("cache_read_price") if price.get("cache_read_price") is not None else price["input_price"])
        + (writes or 0) * (price.get("cache_write_price") if price.get("cache_write_price") is not None else price["input_price"])
        + usage["completion_tokens"] * price["output_price"]
    ) / 1_000_000
    return round(cost, 8)


def has_output(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
            return True
    return bool(chunk.get("output"))


async def measure_request(
    session: aiohttp.ClientSession,
    config: dict[str, Any],
    item: dict[str, Any],
    seq: int,
    run_start: float,
    scheduled_offset_ms: float | None,
    on_progress: Callable[[int | None, bool], None] | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    started = loop.time()
    started_at = utc_now()
    sent_offset_ms = (started - run_start) * 1000
    first_output: float | None = None
    usage: dict[str, Any] | None = None
    status = "ok"
    http_status: int | None = None
    error: str | None = None
    output_seen = False
    finish_reason = None
    target, run = config["target"], config["run"]
    key = target.get("api_key") or ""
    payload = {
        "model": target["model"],
        "messages": item["messages"],
        "max_tokens": run["max_tokens"],
        "stream": True,
    }
    if item.get("tools"):
        payload["tools"] = item["tools"]
    if run.get("include_usage", True):
        payload["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with session.post(endpoint_url(target["base_url"]), json=payload, headers=headers) as response:
            http_status = response.status
            if response.status >= 400:
                detail = (await response.text())[:300]
                raise RuntimeError(f"HTTP {response.status}: {detail}")
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                decoder = codecs.getincrementaldecoder("utf-8")()
                buffer = ""

                def consume(event: str) -> None:
                    nonlocal first_output, usage, output_seen, finish_reason
                    data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                    if not data or data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        return
                    if chunk.get("error"):
                        detail = chunk["error"]
                        raise RuntimeError(str(detail.get("message", detail)) if isinstance(detail, dict) else str(detail))
                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        if on_progress:
                            on_progress(usage_from(usage)["completion_tokens"], False)
                    if has_output(chunk):
                        if on_progress:
                            on_progress(None, True)
                        output_seen = True
                        if first_output is None:
                            first_output = loop.time()

                async for data in response.content.iter_any():
                    buffer += decoder.decode(data)
                    while match := re.search(r"\r?\n\r?\n", buffer):
                        event, buffer = buffer[:match.start()], buffer[match.end():]
                        consume(event)
                buffer += decoder.decode(b"", final=True)
                if buffer.strip():
                    consume(buffer)
            else:
                body = await response.json()
                usage = body.get("usage")
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason")
                output_seen = bool(message.get("content") or message.get("tool_calls") or message.get("reasoning_content") or message.get("reasoning"))
            if not output_seen:
                raise RuntimeError("响应没有生成内容")
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
        status = "error"
        error = str(exc).replace(key, "[REDACTED]")[:400] if key else str(exc)[:400]
    ended = loop.time()
    parsed = usage_from(usage)
    ttft = (first_output - started) * 1000 if first_output is not None else None
    e2e = (ended - started) * 1000
    output_tokens = parsed["completion_tokens"]
    tpot = (e2e - ttft) / (output_tokens - 1) if ttft is not None and output_tokens is not None and output_tokens >= 2 else None
    return {
        "id": str(uuid.uuid4()),
        "seq": seq,
        "status": status,
        "http_status": http_status,
        "started_at": started_at,
        "ttft_ms": round(ttft, 2) if ttft is not None else None,
        "e2e_ms": round(e2e, 2),
        "tpot_ms": round(tpot, 2) if tpot is not None else None,
        "dataset_item_index": item["index"],
        "session_id": item.get("session_id"), "turn_index": item.get("turn_index"),
        "source_item_id": item.get("source_item_id"), "finish_reason": finish_reason,
        "scheduled_offset_ms": round(scheduled_offset_ms, 2) if scheduled_offset_ms is not None else None,
        "sent_offset_ms": round(sent_offset_ms, 2),
        "queue_delay_ms": round(max(0, sent_offset_ms - scheduled_offset_ms), 2) if scheduled_offset_ms is not None else None,
        **parsed,
        "estimated_cost": estimate_cost(parsed, config.get("price")) if status == "ok" else None,
        "error": error,
    }


async def schedule_requests(
    count: int,
    mode: str,
    qps: float | None,
    concurrency: int,
    send: Callable[[int, float | None], Awaitable[None]],
) -> None:
    """Schedule at fixed intervals; QPS 0 fills the concurrency window ASAP."""
    if mode not in {"sync", "async"} or count < 1 or concurrency < 1:
        raise ValueError("无效的测试调度参数")
    if mode == "async" and (qps is None or not math.isfinite(qps) or qps < 0):
        raise ValueError("异步 QPS 必须是非负数")
    if mode == "sync":
        for seq in range(1, count + 1):
            await send(seq, None)
        return
    loop = asyncio.get_running_loop()
    start = loop.time()
    pending: set[asyncio.Task[None]] = set()
    for seq in range(1, count + 1):
        scheduled_ms = (seq - 1) * 1000 / qps if qps else 0.0
        if qps:
            delay = start + scheduled_ms / 1000 - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        while len(pending) >= concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        task = asyncio.create_task(send(seq, scheduled_ms))
        pending.add(task)
    if pending:
        await asyncio.gather(*pending)


async def run(config: dict[str, Any]) -> None:
    run_config = config["run"]
    root = Path(config["dataset"].get("root") or Path(__file__).resolve().parent.parent)
    kind = config["dataset"]["kind"]
    prepared = kind.startswith("public:")
    if prepared:
        _, meta = paths(root, kind.removeprefix("public:"))
        if json.loads(meta.read_text(encoding="utf-8"))["sha256"] != config["dataset"]["sha256"]:
            raise ValueError("数据快照在创建测试后已变化，请重新创建测试")
        units = select_workloads(load_workloads(root, kind, run_config["max_input_tokens"], run_config["max_tokens"], run_config.get("min_input_tokens", 0)), run_config["unit_count"], run_config["seed"])
        items = units
    else:
        items = load_local_items(kind, root, run_config["max_input_tokens"], run_config["max_tokens"], run_config.get("min_input_tokens", 0))
    rng = random.Random(run_config.get("seed", 42))
    order = list(range(len(items)))
    rng.shuffle(order)
    timeout = aiohttp.ClientTimeout(total=run_config.get("request_timeout", 120))
    connector = aiohttp.TCPConnector(limit=run_config["concurrency"])
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        loop = asyncio.get_running_loop()
        run_start = loop.time()
        emit({"type": "start", "started_at": utc_now()})
        telemetry = RunTelemetry(run_start)

        def record_tick(final: bool = False) -> None:
            emit({"type": "tick", "point": {**telemetry.snapshot(loop.time(), final), "recorded_at": utc_now()}})

        record_tick()
        stop_sampling = asyncio.Event()

        async def collect_timeline() -> None:
            while True:
                try:
                    await asyncio.wait_for(stop_sampling.wait(), timeout=1)
                    return
                except asyncio.TimeoutError:
                    record_tick()

        sampler = asyncio.create_task(collect_timeline())

        def choose_item(seq: int) -> dict[str, Any]:
            if seq > 1 and (seq - 1) % len(order) == 0:
                rng.shuffle(order)
            index = order[(seq - 1) % len(order)]
            return {"index": items[index].get("index", index), "messages": items[index]["messages"]}

        async def send(seq: int, scheduled_ms: float | None) -> None:
            item = choose_item(seq)
            telemetry.request_started()
            sample = await measure_request(session, config, item, seq, run_start, scheduled_ms,
                                           lambda tokens, output: telemetry.progress(seq, tokens, output))
            telemetry.request_finished(sample)
            emit({"type": "sample", "sample": sample})

        offsets = []
        if prepared:
            total = 0
            for unit in units:
                offsets.append(total)
                total += turn_count(unit)
            if total != run_config["request_count"]:
                raise ValueError("数据快照请求数与测试配置不一致")

        async def replay(unit_seq: int, scheduled_ms: float | None) -> None:
            unit = hydrate_workload(root, kind, units[unit_seq - 1])
            for turn_index, turn in enumerate(unit["turns"], 1):
                if turn_index > 1 and run_config.get("turn_delay_ms", 0):
                    await asyncio.sleep(run_config["turn_delay_ms"] / 1000)
                seq = offsets[unit_seq - 1] + turn_index
                item = {**turn, "index": unit["index"], "source_item_id": unit["id"],
                        "session_id": f"{unit_seq}:{unit['id']}", "turn_index": turn_index}
                telemetry.request_started()
                sample = await measure_request(session, config, item, seq, run_start,
                    scheduled_ms if turn_index == 1 else None,
                    lambda tokens, output: telemetry.progress(seq, tokens, output))
                telemetry.request_finished(sample)
                emit({"type": "sample", "sample": sample})

        try:
            await schedule_requests(
                len(units) if prepared else run_config["request_count"],
                run_config["mode"],
                run_config.get("request_rate"),
                run_config["concurrency"],
                replay if prepared else send,
            )
        finally:
            stop_sampling.set()
            await sampler
            record_tick(final=True)
        emit({"type": "done", "finished_at": utc_now()})


def main() -> None:
    # The API process sends and reads UTF-8 even on Windows with a GBK locale.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        config = json.load(sys.stdin)
        asyncio.run(run(config))
    except Exception as exc:
        emit({"type": "error", "error": str(exc)[:400]})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
