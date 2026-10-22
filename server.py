"""CloudBench's Python web server, management API and benchmark supervisor."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web

from pybench.connection import connect
from pybench.datasets import eligible_count, length_distribution, token_counts
from pybench.workloads import catalog, select_for_run, turn_count
from pybench.results import TIMELINE_COLUMNS, csv_bytes, export_csv, summarize, utc_now

ROOT = Path(__file__).resolve().parent
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS targets (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL,
  model TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
  id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL,
  currency TEXT NOT NULL, input_price REAL NOT NULL,
  cache_read_price REAL, cache_write_price REAL, output_price REAL NOT NULL,
  source_url TEXT, checked_at TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, target_id TEXT NOT NULL, target_name TEXT NOT NULL,
  model TEXT NOT NULL, base_url TEXT NOT NULL, price_snapshot TEXT,
  prompt TEXT NOT NULL, max_tokens INTEGER NOT NULL, request_count INTEGER NOT NULL,
  concurrency INTEGER NOT NULL, include_usage INTEGER NOT NULL,
  gpu_count INTEGER, gpu_hour_price REAL, gpu_currency TEXT,
  dataset_id TEXT, dataset_name TEXT, mode TEXT, request_rate REAL,
  arrival_pattern TEXT, seed INTEGER, input_words INTEGER, max_input_tokens INTEGER,
  status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS run_timeline (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, data_json TEXT NOT NULL,
  PRIMARY KEY (run_id,seq), FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS samples (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
  status TEXT NOT NULL, http_status INTEGER, started_at TEXT NOT NULL,
  ttft_ms REAL, e2e_ms REAL, tpot_ms REAL,
  prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER,
  uncached_tokens INTEGER, cache_write_tokens INTEGER,
  usage_source TEXT NOT NULL, estimated_cost REAL, error TEXT,
  dataset_item_index INTEGER, scheduled_offset_ms REAL, sent_offset_ms REAL, queue_delay_ms REAL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS samples_run_seq ON samples(run_id,seq);
"""


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def json_response(value: Any, status: int = 200) -> web.Response:
    return web.json_response(value, status=status, dumps=lambda v: json.dumps(v, ensure_ascii=False),
                             headers={"Cache-Control": "no-store"})


@web.middleware
async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        if request.path.startswith("/api/") and request.method != "GET":
            origin = request.headers.get("Origin")
            if origin and urlsplit(origin).netloc != request.host:
                raise ApiError("跨站请求已拒绝", 403)
        response = await handler(request)
        if not request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    except ApiError as exc:
        return json_response({"error": str(exc)}, exc.status)
    except (ValueError, UnicodeError) as exc:
        return json_response({"error": str(exc)}, 400)
    except web.HTTPException as exc:
        if request.path.startswith("/api/"):
            return json_response({"error": exc.reason}, exc.status)
        raise
    except Exception:
        logging.exception("CloudBench API 异常")
        return json_response({"error": "服务器内部错误"}, 500)


async def body_json(request: web.Request) -> dict[str, Any]:
    if request.content_type != "application/json":
        raise ApiError("请求必须是 JSON")
    try:
        body = await request.json()
    except (ValueError, UnicodeError):
        raise ApiError("JSON 格式无效") from None
    if not isinstance(body, dict):
        raise ApiError("请求体必须是 JSON 对象")
    return body


def required_string(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ApiError(f"{label}无效")
    return value.strip()


def number(value: Any, label: str, minimum: float, maximum: float, integer: bool = False) -> float | int:
    try:
        if value is None or value == "" or isinstance(value, bool):
            raise ValueError
        result = float(value)
        if not math.isfinite(result) or result < minimum or result > maximum or (integer and not result.is_integer()):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ApiError(f"{label}需在 {minimum}–{maximum} 之间") from None
    return int(result) if integer else result


def nullable_price(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    return float(number(value, label, 0, float("inf")))


def http_url(value: Any, label: str, base: bool = False) -> str:
    text = required_string(value, label, 2000)
    parsed = urlsplit(text)
    try:
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and not any(c.isspace() for c in text)
        parsed.port
    except ValueError:
        valid = False
    if not valid or (base and (parsed.username or parsed.password or parsed.query or parsed.fragment)):
        raise ApiError(f"{label}须为有效 HTTP(S) 地址；Base URL 不支持账号、查询参数或片段")
    return text.rstrip("/") if base else text


@dataclass
class ActiveRun:
    task: asyncio.Task[None] | None = None
    process: asyncio.subprocess.Process | None = None
    cancelled: bool = False


class CloudBench:
    def __init__(self, data_dir: Path, root: Path = ROOT):
        self.root = root.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(data_dir / "cloudbench.sqlite", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        migrations = {
            "targets": {"protocol": "TEXT", "endpoint_url": "TEXT", "checked_at": "TEXT"},
            "runs": {"dataset_id": "TEXT", "dataset_name": "TEXT", "mode": "TEXT", "request_rate": "REAL",
                     "arrival_pattern": "TEXT", "seed": "INTEGER", "input_words": "INTEGER", "max_input_tokens": "INTEGER",
                     "unit_count": "INTEGER", "min_input_tokens": "INTEGER", "request_timeout": "REAL",
                     "turn_delay_ms": "REAL", "dataset_snapshot": "TEXT"},
            "samples": {"dataset_item_index": "INTEGER", "scheduled_offset_ms": "REAL", "sent_offset_ms": "REAL", "queue_delay_ms": "REAL",
                        "session_id": "TEXT", "turn_index": "INTEGER", "source_item_id": "TEXT", "finish_reason": "TEXT"},
        }
        for table, columns in migrations.items():
            present = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns.items():
                if name not in present:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
        self.db.execute("UPDATE runs SET status='error',finished_at=?,error=? WHERE status='running'",
                        (utc_now(), "后台进程已重启，此测试已中断"))
        self.credentials: dict[str, str] = {}
        self.active: dict[str, ActiveRun] = {}
        self.app = web.Application(middlewares=[errors], client_max_size=1_000_000)
        self.app.add_routes([
            web.get("/api/state", self.state), web.post("/api/targets", self.create_target),
            web.post("/api/targets/{id}/connect", self.reconnect_target),
            web.post("/api/targets/{id}/key", self.set_key), web.delete("/api/targets/{id}", self.delete_target),
            web.post("/api/prices", self.create_price), web.post("/api/runs", self.create_run),
            web.get("/api/runs/{id}", self.get_run), web.post("/api/runs/{id}/cancel", self.cancel_run),
            web.get("/api/runs/{id}/{kind:summary|samples|timeline}.csv", self.download_csv),
            web.get("/", self.index), web.static("/", self.root / "public", follow_symlinks=False),
        ])
        self.app.on_cleanup.append(self.close)

    def rows(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute(sql, parameters).fetchall()]

    def one(self, sql: str, parameters: tuple[Any, ...]) -> dict[str, Any] | None:
        row = self.db.execute(sql, parameters).fetchone()
        return dict(row) if row is not None else None

    def insert(self, table: str, record: dict[str, Any]) -> None:
        if table not in {"targets", "prices", "runs", "samples"}:
            raise ValueError("未知数据表")
        columns = ",".join(record)
        values = ",".join("?" for _ in record)
        self.db.execute(f"INSERT INTO {table} ({columns}) VALUES ({values})", tuple(record.values()))

    def datasets(self) -> list[dict[str, Any]]:
        specifications = [
            ("local:longbench", "LongBench", "local-longbench", 4750, "zai-org/LongBench",
             "官方原始集，4,750 条；1K–64K token，排除 LongBench-E。", "raw/longbench/extracted/data/multi_news.jsonl", "longbench-token-counts.json"),
            ("local:longbench-v2", "LongBench v2", "local-longbench-v2", 503, "zai-org/LongBench-v2",
             "官方原始集，503 条；约 10K–4M token，按上下文上限筛选。", "raw/longbench-v2/data.json", "longbench-v2-token-counts.json"),
        ]
        directory = self.root / "datasets"
        return [{"id": key, "name": name, "kind": kind, "item_count": count, "source": source, "description": description,
                 "distribution": length_distribution(token_counts(kind, self.root)), "ready": True,
                 "category": "long-context", "unit": "样本", "length_label": "Kimi K3 token（不含聊天模板）"}
                for key, name, kind, count, source, description, raw, index in specifications
                if (directory / raw).is_file() and (directory / index).is_file()] + catalog(self.root)

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.root / "public" / "index.html")

    async def state(self, request: web.Request) -> web.Response:
        targets = [{**target, "has_key": target["id"] in self.credentials}
                   for target in self.rows("SELECT * FROM targets ORDER BY created_at DESC")]
        return json_response({"targets": targets, "prices": self.rows("SELECT * FROM prices ORDER BY created_at DESC"),
                              "datasets": self.datasets(), "runs": self.rows("SELECT id,target_name,model,dataset_name,mode,request_rate,status,started_at,finished_at,request_count FROM runs ORDER BY started_at DESC LIMIT 30")})

    async def create_target(self, request: web.Request) -> web.Response:
        body = await body_json(request)
        url = http_url(body.get("base_url"), "服务地址", base=True)
        model = required_string(body['model'], "模型 ID", 200) if body.get('model') else None
        name = required_string(body['name'], "名称", 100) if body.get('name') else None
        protocol = body.get('protocol') or None
        if protocol not in (None, 'openai', 'anthropic'):
            raise ApiError('接口格式无效')
        key = required_string(body["api_key"], "API Key", 4000) if body.get("api_key") else None
        result = await connect(url, key, model, protocol)
        if result['status'] != 'connected':
            return json_response(result)
        target = {"id": str(uuid.uuid4()), "name": name or result['model'][:100],
                  "base_url": url, "model": result['model'], "created_at": utc_now(),
                  **{k: result[k] for k in ('protocol', 'endpoint_url', 'checked_at')}}
        self.insert("targets", target)
        if key:
            self.credentials[target["id"]] = key
        return json_response({**target, "has_key": bool(key), "status": "connected"}, 201)

    async def verify_target(self, target):
        result = await connect(target.get('endpoint_url') or target['base_url'],
                               self.credentials.get(target['id']), target['model'],
                               target.get('protocol') if target.get('endpoint_url') else None)
        if not self.one('SELECT id FROM targets WHERE id=?', (target['id'],)):
            raise ApiError('目标不存在', 404)
        self.db.execute('UPDATE targets SET protocol=?,endpoint_url=?,checked_at=? WHERE id=?',
                        (result['protocol'], result['endpoint_url'], result['checked_at'], target['id']))
        target.update({k: result[k] for k in ('protocol', 'endpoint_url', 'checked_at')})
        return result

    async def reconnect_target(self, request):
        target = self.one('SELECT * FROM targets WHERE id=?', (request.match_info['id'],))
        if target is None:
            raise ApiError('目标不存在', 404)
        return json_response(await self.verify_target(target))

    async def set_key(self, request: web.Request) -> web.Response:
        target_id = request.match_info["id"]
        if not self.one("SELECT id FROM targets WHERE id=?", (target_id,)):
            raise ApiError("目标不存在", 404)
        body = await body_json(request)
        self.credentials[target_id] = required_string(body.get("api_key"), "API Key", 4000)
        return json_response({"ok": True})

    async def delete_target(self, request: web.Request) -> web.Response:
        target_id = request.match_info["id"]
        self.db.execute("DELETE FROM targets WHERE id=?", (target_id,))
        self.credentials.pop(target_id, None)
        return json_response({"ok": True})

    async def create_price(self, request: web.Request) -> web.Response:
        body = await body_json(request)
        price = {"id": str(uuid.uuid4()), "provider": required_string(body.get("provider"), "供应商", 100),
                 "model": required_string(body.get("model"), "模型 ID", 200),
                 "currency": required_string(body.get("currency") or "USD", "币种", 8).upper(),
                 "input_price": nullable_price(body.get("input_price"), "输入价格"),
                 "cache_read_price": nullable_price(body.get("cache_read_price"), "缓存读取价格"),
                 "cache_write_price": nullable_price(body.get("cache_write_price"), "缓存写入价格"),
                 "output_price": nullable_price(body.get("output_price"), "输出价格"),
                 "source_url": http_url(body["source_url"], "来源链接") if body.get("source_url") else None,
                 "checked_at": required_string(body.get("checked_at") or utc_now()[:10], "核对日期", 40),
                 "created_at": utc_now()}
        if price["input_price"] is None or price["output_price"] is None:
            raise ApiError("输入和输出价格必填")
        self.insert("prices", price)
        return json_response(price, 201)

    async def create_run(self, request: web.Request) -> web.Response:
        if self.active or getattr(self, 'preparing', False):
            raise ApiError("已有测试正在运行，请等待或取消", 409)
        body = await body_json(request)
        target = self.one("SELECT * FROM targets WHERE id=?", (body.get("target_id"),))
        if target is None:
            raise ApiError("目标不存在", 404)
        if not target.get('endpoint_url'):
            self.preparing = True
            try:
                await self.verify_target(target)
            finally:
                self.preparing = False
        price = self.one("SELECT * FROM prices WHERE id=?", (body["price_id"],)) if body.get("price_id") else None
        if body.get("price_id") and price is None:
            raise ApiError("价格版本不存在")
        dataset = next((d for d in self.datasets() if d["id"] == body.get("dataset_id")), None)
        if dataset is None:
            raise ApiError("请选择有效数据集")
        if not dataset.get("ready", True):
            raise ApiError("数据集尚未准备，见 README 中的数据准备命令")
        mode = body.get("mode")
        if mode not in {"sync", "async"}:
            raise ApiError("请选择同步或异步模式")
        qps = number(body.get("request_rate"), "目标 QPS", 0, 1000) if mode == "async" else None
        if mode == "async" and body.get("arrival_pattern") not in {None, "", "fixed", "burst"}:
            raise ApiError("当前仅支持固定间隔或 QPS=0 突发模式")
        context_limit = number(body.get("max_input_tokens") or 131072, "模型上下文上限", 2048, 4194304, True)
        output_limit = number(body.get("max_tokens"), "最大输出 token", 1, 8192, True)
        minimum = number(body.get("min_input_tokens") or 0, "最短输入 token", 0, context_limit - 1, True)
        seed = 42 if body.get("seed") in (None, "") else number(body["seed"], "随机种子", 0, 4294967295, True)
        timeout = number(120 if body.get("request_timeout") in (None, "") else body["request_timeout"], "单请求超时", 1, 7200)
        turn_delay = number(body.get("turn_delay_ms") or 0, "轮间等待", 0, 60000)
        requested_count = body.get("request_count")
        units = None if requested_count in (None, "") else number(requested_count, "测试条数/会话数", 1, 1000000000, True)
        if self.active or getattr(self, 'preparing', False):
            raise ApiError("已有测试正在准备或运行，请等待或取消", 409)
        if dataset["kind"].startswith("public:"):
            self.preparing = True
            try:
                selected = await asyncio.to_thread(select_for_run, self.root, dataset["kind"], context_limit, output_limit, minimum, units, seed)
            finally:
                self.preparing = False
            units = len(selected)
            count = sum(turn_count(item) for item in selected)
            dataset = {**dataset, "selected": selected}
        else:
            available = eligible_count(dataset["kind"], self.root, context_limit, output_limit, minimum)
            if units is not None and units > available:
                raise ApiError(f"所选长度范围只有 {available} 条可用样本，不能通过重复样本补足 {units} 条")
            units = available if units is None else units
            count = units
        concurrency = body.get("concurrency")
        run = {
            "id": str(uuid.uuid4()), "target_id": target["id"], "target_name": target["name"],
            "model": target["model"], "base_url": target["base_url"],
            "price_snapshot": json.dumps(price, ensure_ascii=False) if price else None,
            "prompt": dataset["name"], "max_tokens": output_limit,
            "request_count": count, "concurrency": 1 if mode == "sync" else (
                0 if concurrency is None or concurrency == "" else number(concurrency, "并发数", 1, 1000000000, True)),
            "unit_count": units, "min_input_tokens": minimum, "request_timeout": timeout, "turn_delay_ms": turn_delay,
            "dataset_snapshot": json.dumps({k: dataset.get(k) for k in ("sha256", "tokenizer", "length_label", "prepared_at", "sampling", "source", "offset", "max_turns")}, ensure_ascii=False),
            "include_usage": 0 if body.get("include_usage") is False else 1,
            "dataset_id": dataset["id"], "dataset_name": dataset["name"], "mode": mode,
            "request_rate": qps, "arrival_pattern": ("burst" if qps == 0 else "fixed") if mode == "async" else None,
            "seed": seed,
            "input_words": None,
            "max_input_tokens": context_limit,
            "gpu_count": number(body["gpu_count"], "GPU 数量", 1, 128, True) if body.get("gpu_count") else None,
            "gpu_hour_price": nullable_price(body.get("gpu_hour_price"), "每卡每小时成本"),
            "gpu_currency": required_string(body["gpu_currency"], "硬件成本币种", 8).upper() if body.get("gpu_currency") else None,
            "status": "running", "started_at": utc_now(), "finished_at": None, "error": None,
        }
        if (run["gpu_count"] is None) != (run["gpu_hour_price"] is None):
            raise ApiError("GPU 数量和每卡每小时成本需同时填写")
        if run["gpu_count"] and not run["gpu_currency"]:
            run["gpu_currency"] = "CNY"
        self.insert("runs", run)
        active = ActiveRun()
        self.active[run["id"]] = active
        active.task = asyncio.create_task(self.execute_run(run, target, price, dataset, active))
        return json_response({"id": run["id"]}, 202)

    async def execute_run(self, run: dict[str, Any], target: dict[str, Any], price: dict[str, Any] | None,
                          dataset: dict[str, Any], active: ActiveRun) -> None:
        key = self.credentials.get(target["id"], "")
        done_at = None
        worker_error = None
        stderr = ""
        stderr_task = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pybench.runner", cwd=ROOT,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            active.process = process

            async def read_stderr() -> None:
                nonlocal stderr
                while chunk := await process.stderr.read(4096):
                    stderr = (stderr + chunk.decode("utf-8", errors="replace"))[-1000:]

            stderr_task = asyncio.create_task(read_stderr())
            if active.cancelled:
                process.terminate()
            else:
                config = {"target": {"base_url": target["base_url"], "model": target["model"], "api_key": key or None,
                                     "endpoint_url": target.get('endpoint_url'), "protocol": target.get('protocol')},
                          "run": {field: run[field] for field in ("mode", "request_rate", "concurrency", "request_count", "max_tokens", "seed", "max_input_tokens", "unit_count", "min_input_tokens", "request_timeout", "turn_delay_ms")},
                          "dataset": {"kind": dataset["kind"], "sha256": dataset.get("sha256"), "root": str(self.root)}, "price": price}
                if "selected" in dataset:
                    config["dataset"]["selected"] = dataset["selected"]
                config["run"]["include_usage"] = bool(run["include_usage"])
                process.stdin.write(json.dumps(config, ensure_ascii=False).encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()
            async for line in process.stdout:
                try:
                    event = json.loads(line)
                    if event["type"] == "start":
                        self.db.execute("UPDATE runs SET started_at=? WHERE id=?", (event["started_at"], run["id"]))
                    elif event["type"] == "sample":
                        self.insert("samples", {**event["sample"], "run_id": run["id"]})
                    elif event["type"] == "tick":
                        point = event["point"]
                        self.db.execute("INSERT INTO run_timeline (run_id,seq,data_json) VALUES (?,?,?)",
                                        (run["id"], point["seq"], json.dumps(point, ensure_ascii=False)))
                    elif event["type"] == "done":
                        done_at = event["finished_at"]
                    elif event["type"] == "error":
                        worker_error = event["error"]
                except (ValueError, KeyError, sqlite3.Error) as exc:
                    worker_error = f"Python 结果无法处理：{exc}"
            await process.wait()
            await stderr_task
            if process.returncode != 0 or not done_at:
                worker_error = worker_error or stderr or "Python 测试进程异常退出"
        except Exception as exc:
            worker_error = str(exc)
        finally:
            process = active.process
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            if stderr_task is not None:
                await stderr_task
            status = "cancelled" if active.cancelled else "error" if worker_error else "completed"
            message = str(worker_error).replace(key, "[REDACTED]")[:400] if key and worker_error else str(worker_error)[:400] if worker_error else None
            self.db.execute("UPDATE runs SET status=?,finished_at=?,error=? WHERE id=?",
                            (status, done_at or utc_now(), None if active.cancelled else message, run["id"]))
            self.active.pop(run["id"], None)

    def read_run(self, run_id: str) -> dict[str, Any]:
        run = self.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if run is None:
            raise ApiError("测试不存在", 404)
        return run

    def samples(self, run_id: str) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM samples WHERE run_id=? ORDER BY seq", (run_id,))

    def timeline(self, run_id: str) -> list[dict[str, Any]]:
        return [json.loads(row["data_json"]) for row in self.db.execute("SELECT data_json FROM run_timeline WHERE run_id=? ORDER BY seq", (run_id,))]

    async def get_run(self, request: web.Request) -> web.Response:
        run = self.read_run(request.match_info["id"])
        samples = self.samples(run["id"])
        return json_response({"run": {key: value for key, value in run.items() if key != "prompt"}, "samples": samples,
                              "timeline": self.timeline(run["id"]), "summary": summarize(run, samples)})

    def stop(self, active: ActiveRun) -> None:
        active.cancelled = True
        if active.process is not None and active.process.returncode is None:
            try:
                active.process.terminate()
            except ProcessLookupError:
                pass

    async def cancel_run(self, request: web.Request) -> web.Response:
        active = self.active.get(request.match_info["id"])
        if active is None:
            raise ApiError("运行中的测试不存在", 404)
        self.stop(active)
        return json_response({"ok": True})

    async def download_csv(self, request: web.Request) -> web.Response:
        run_id, kind = request.match_info["id"], request.match_info["kind"]
        run = self.read_run(run_id)
        content = csv_bytes([{"run_id": run_id, **point} for point in self.timeline(run_id)], TIMELINE_COLUMNS) if kind == "timeline" else export_csv(run, self.samples(run_id), kind)
        return web.Response(body=content, content_type="text/csv", charset="utf-8",
                            headers={"Content-Disposition": f'attachment; filename="cloudbench-{run_id}-{kind}.csv"', "Cache-Control": "no-store"})

    async def close(self, app: web.Application) -> None:
        active_runs = list(self.active.values())
        for active in active_runs:
            self.stop(active)
        await asyncio.gather(*(active.task for active in active_runs if active.task is not None), return_exceptions=True)
        self.credentials.clear()
        self.db.close()


def create_app(data_dir: Path | None = None, root: Path = ROOT) -> web.Application:
    directory = data_dir or Path(os.environ.get("CLOUDBENCH_DATA_DIR", ROOT / "data")).resolve()
    return CloudBench(directory, root).app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=int(os.environ.get("CLOUDBENCH_PORT", "3876")),
                print=lambda message: print(message, flush=True), access_log=None)
