"""Whole-run summaries and CSV exports shared by the Python API."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from pybench.telemetry import percentile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def summarize(run: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [sample for sample in samples if sample["status"] == "ok"]

    def metric(key: str) -> dict[str, Any]:
        values = [s[key] for s in ok if s.get(key) is not None]
        return {"p50": percentile(values, .5, 2), "p95": percentile(values, .95, 2),
                "p99": percentile(values, .99, 2), "n": len(values)}

    known_cache = [s for s in ok if s["cached_tokens"] is not None and s["prompt_tokens"] is not None]
    cache_input = sum(s["prompt_tokens"] for s in known_cache)
    cache_hits = sum(s["cached_tokens"] for s in known_cache)
    start = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
    end = datetime.fromisoformat((run["finished_at"] or utc_now()).replace("Z", "+00:00"))
    duration = max((end - start).total_seconds(), .001)
    output = sum(s["completion_tokens"] or 0 for s in ok)
    sent = sorted(s["sent_offset_ms"] for s in samples if s["sent_offset_ms"] is not None)
    priced = [s for s in ok if s["estimated_cost"] is not None]
    hourly = run["gpu_count"] * run["gpu_hour_price"] if run["gpu_count"] and run["gpu_hour_price"] is not None else None
    hardware_cost = hourly * duration / 3600 if hourly is not None else None
    arrival_qps = (len(sent) - 1) * 1000 / (sent[-1] - sent[0]) if (
        run["arrival_pattern"] != "burst" and len(sent) >= 2 and sent[-1] > sent[0]
    ) else None
    return {
        "completed": len(samples), "success": len(ok), "failed": len(samples) - len(ok),
        "success_rate": rounded(len(ok) / len(samples) * 100, 1) if samples else None,
        "ttft": metric("ttft_ms"), "tpot": metric("tpot_ms"), "e2e": metric("e2e_ms"),
        "output_tps": rounded(output / duration, 2) if output else None,
        "request_tps": rounded(len(ok) / duration, 2),
        "actual_arrival_qps": rounded(arrival_qps, 2),
        "queue_delay_p50_ms": percentile([s["queue_delay_ms"] for s in samples if s["queue_delay_ms"] is not None], .5, 2),
        "cache_hit_rate": rounded(cache_hits / cache_input * 100, 1) if cache_input else None,
        "cache_coverage": len(known_cache),
        "estimated_cost": rounded(sum(s["estimated_cost"] for s in priced), 8) if ok and len(priced) == len(ok) else None,
        "price_coverage": len(priced), "hardware_hourly_cost": rounded(hourly, 4),
        "hardware_run_cost": rounded(hardware_cost, 6),
        "hardware_per_1m_output": rounded(hardware_cost / output * 1_000_000, 4) if hardware_cost is not None and output else None,
    }


RUN_COLUMNS = "schema_version run_id target_name model dataset_id dataset_name mode request_rate_qps arrival_pattern seed input_words max_input_tokens status started_at finished_at request_count concurrency completed success failed success_rate_pct ttft_p50_ms ttft_p95_ms ttft_p99_ms tpot_p50_ms tpot_p95_ms tpot_p99_ms e2e_p50_ms e2e_p95_ms e2e_p99_ms output_tps request_tps actual_arrival_qps queue_delay_p50_ms cache_hit_rate_pct cache_coverage estimated_cost currency price_source price_checked_at gpu_count gpu_hour_price gpu_currency hardware_hourly_cost hardware_run_cost hardware_per_1m_output".split()
SAMPLE_COLUMNS = "schema_version run_id seq dataset_item_index scheduled_offset_ms sent_offset_ms queue_delay_ms status http_status started_at ttft_ms tpot_ms e2e_ms prompt_tokens completion_tokens cached_tokens uncached_tokens cache_write_tokens usage_source estimated_cost error".split()
TIMELINE_COLUMNS = "run_id seq recorded_at elapsed_ms window_ms rate_window_ms sent_qps completed_qps success_qps failed_qps reported_output_tps stream_events_per_s sent completed success failed inflight reported_output_tokens usage_known usage_missing ttft_p50_ms ttft_p95_ms e2e_p95_ms final".split()


def csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    def cell(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(("=", "+", "@", "-", "\t", "\r")):
            return "'" + value
        if isinstance(value, bool):
            return str(value).lower()
        return value

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(columns)
    writer.writerows([cell(row.get(column)) for column in columns] for row in rows)
    return output.getvalue().encode("utf-8-sig")


def export_csv(run: dict[str, Any], samples: list[dict[str, Any]], kind: str) -> bytes:
    if kind == "samples":
        return csv_bytes([{"schema_version": 2, **sample} for sample in samples], SAMPLE_COLUMNS)
    summary = summarize(run, samples)
    price = json.loads(run["price_snapshot"]) if run["price_snapshot"] else {}
    row = {**run, **summary, "schema_version": 2, "run_id": run["id"],
           "request_rate_qps": run["request_rate"], "success_rate_pct": summary["success_rate"],
           "cache_hit_rate_pct": summary["cache_hit_rate"], "currency": price.get("currency"),
           "price_source": price.get("source_url"), "price_checked_at": price.get("checked_at")}
    for metric in ("ttft", "tpot", "e2e"):
        for p in ("p50", "p95", "p99"):
            row[f"{metric}_{p}_ms"] = summary[metric][p]
    return csv_bytes([row], RUN_COLUMNS)
