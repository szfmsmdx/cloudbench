"""Full, explicitly named source partitions. Raw downloads are retained on disk."""
from __future__ import annotations

import bz2
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_SCOPES = {
    "open-swe": ("nvidia/Open-SWE-Traces", "v1.2", "minisweagent"),
    "swe-smith": ("SWE-bench/SWE-smith-trajectories", "default", "tool"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
}


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def source_manifest(key):
    if key in HF_SCOPES:
        dataset, config, split = HF_SCOPES[key]
        parquet = fetch_json(f"https://datasets-server.huggingface.co/parquet?dataset={dataset}")
        sizes = fetch_json(f"https://datasets-server.huggingface.co/size?dataset={dataset}")
        files = [f for f in parquet["parquet_files"] if f["config"] == config and f["split"] == split]
        size, = [s for s in sizes["size"]["splits"] if s["config"] == config and s["split"] == split]
        if not files or parquet.get("partial") or sizes.get("partial"):
            raise ValueError("上游未提供完整文件清单，不能将部分文件标记为完整数据")
        return {"key": key, "scope": f"{dataset} / {config} / {split}", "expected_rows": size["num_rows"],
                "format": "parquet", "files": files}
    if key == "crag":
        return {"key": key, "scope": "CRAG Task 1/2 dev v4（完整公开文件）", "format": "jsonl.bz2",
                "files": [{"filename": "crag_task_1_and_2_dev_v4.jsonl.bz2", "size": 739384484,
                    "sha256": "afa29f2b3facfb5d15aa9cded00d5ec90ff76f3e67279e7b99cfe86659a641ca",
                    "url": "https://media.githubusercontent.com/media/facebookresearch/CRAG/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"}]}
    return {"key": key, "scope": "CRUD-RAG crud_split/split_merged.json（完整六类任务）", "format": "json",
            "files": [{"filename": "split_merged.json", "url": "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json"}]}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(directory, spec, endpoint):
    dest = directory / spec["filename"]
    expected = spec.get("size")
    if not dest.is_file() or (expected and dest.stat().st_size != expected):
        partial = dest.with_suffix(dest.suffix + ".download")
        url = spec["url"]
        if endpoint and url.startswith("https://huggingface.co/"):
            url = endpoint.rstrip("/") + url.removeprefix("https://huggingface.co")
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                headers = {"User-Agent": "CloudBench/1.0", **({"Range": f"bytes={offset}-"} if offset else {})}
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("下载断点与服务器响应不一致")
                    with partial.open("ab" if resumed else "wb") as stream:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(block)
                if expected and partial.stat().st_size != expected:
                    raise ValueError(f"文件长度不完整：{partial.name}，{partial.stat().st_size}/{expected}")
                partial.replace(dest)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    digest = file_hash(dest)
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError(f"源文件 SHA-256 不一致：{dest.name}")
    return {**spec, "sha256": digest, "size": dest.stat().st_size}


def download_source(root, key, manifest, endpoint=None, progress=None):
    directory = root / "datasets" / "raw" / key
    directory.mkdir(parents=True, exist_ok=True)
    specs = manifest["files"]
    completed = []
    workers = max(1, min(32, int(os.environ.get("CLOUDBENCH_DOWNLOAD_WORKERS", "8"))))
    with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        futures = {pool.submit(download_file, directory, spec, endpoint): spec for spec in specs}
        for future in as_completed(futures):
            completed.append(future.result())
            if progress:
                progress(status="downloading", files_done=len(completed), files_total=len(specs))
            print(f"{key}: downloaded {len(completed)}/{len(specs)} files", flush=True)
    manifest = {**manifest, "files": sorted(completed, key=lambda f: f["filename"])}
    (directory / "source-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def source_rows(root, key, manifest):
    directory = root / "datasets" / "raw" / key
    for spec in manifest["files"]:
        path = directory / spec["filename"]
        if manifest["format"] == "parquet":
            import pyarrow.parquet as pq
            # Keep only a few complete rows in memory; agent rows may be large.
            for batch in pq.ParquetFile(path, pre_buffer=False).iter_batches(batch_size=1, use_threads=False):
                yield from batch.to_pylist()
        elif manifest["format"] == "jsonl.bz2":
            with bz2.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        else:
            from pybench.prepare import local_rows
            yield from local_rows(path)
