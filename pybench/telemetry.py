"""Small in-memory counters sampled by the runner, never one IPC event per token."""

from __future__ import annotations

from typing import Any


def percentile(values: list[float], fraction: float, digits: int = 3) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo), digits)


class RunTelemetry:
    def __init__(self, start: float):
        self.start = self.previous_time = start
        self.seq = 0
        self.sent = self.completed = self.success = self.failed = 0
        self.output_tokens = self.stream_events = self.usage_known = self.usage_missing = 0
        self.usage_seen = False
        self.request_output: dict[int, int] = {}
        self.previous = (0, 0, 0, 0, 0, 0)
        self.before_previous = (start, self.previous)
        self.ttft: list[float] = []
        self.e2e: list[float] = []

    def request_started(self) -> None:
        self.sent += 1

    def progress(self, seq: int, output_tokens: int | None, has_output: bool) -> None:
        if has_output:
            self.stream_events += 1
        if output_tokens is not None:
            self.usage_seen = True
            previous = self.request_output.get(seq, 0)
            self.output_tokens += max(0, output_tokens - previous)
            self.request_output[seq] = max(previous, output_tokens)

    def request_finished(self, sample: dict[str, Any]) -> None:
        self.completed += 1
        if sample["status"] == "ok":
            self.success += 1
            if sample["ttft_ms"] is not None:
                self.ttft.append(sample["ttft_ms"])
            self.e2e.append(sample["e2e_ms"])
        else:
            self.failed += 1
        output_tokens = sample["completion_tokens"]
        self.progress(sample["seq"], output_tokens, False)
        if output_tokens is None:
            self.usage_missing += 1
        else:
            self.usage_known += 1

    def snapshot(self, now: float, final: bool = False) -> dict[str, Any]:
        duration = max(0, now - self.previous_time)
        current = (self.sent, self.completed, self.success, self.failed, self.output_tokens, self.stream_events)
        rate_start, rate_counts = self.previous_time, self.previous
        # A final 20 ms tail can otherwise turn one completion into a 50 QPS
        # spike. Keep the point, but report its rates over a longer real window.
        if final and duration < 1 and self.seq > 1:
            rate_start, rate_counts = self.before_previous
        rate_duration = max(0, now - rate_start)
        rates = [round((value - previous) / rate_duration, 6) if rate_duration else 0.0 for value, previous in zip(current, rate_counts)]
        point = {
            "seq": self.seq,
            "elapsed_ms": round((now - self.start) * 1000, 3),
            "window_ms": round(duration * 1000, 3),
            "rate_window_ms": round(rate_duration * 1000, 3),
            "sent_qps": rates[0], "completed_qps": rates[1],
            "success_qps": rates[2], "failed_qps": rates[3],
            "reported_output_tps": rates[4] if self.usage_seen else None,
            "stream_events_per_s": rates[5],
            "sent": self.sent, "completed": self.completed,
            "success": self.success, "failed": self.failed,
            "inflight": self.sent - self.completed,
            "reported_output_tokens": self.output_tokens,
            "usage_known": self.usage_known, "usage_missing": self.usage_missing,
            "ttft_p50_ms": percentile(self.ttft, .5),
            "ttft_p95_ms": percentile(self.ttft, .95),
            "e2e_p95_ms": percentile(self.e2e, .95),
            "final": final,
        }
        self.seq += 1
        self.before_previous = (self.previous_time, self.previous)
        self.previous_time, self.previous = now, current
        self.ttft.clear()
        self.e2e.clear()
        return point
