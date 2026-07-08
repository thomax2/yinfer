#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark helper for the LLM serving engine.

This script intentionally uses only the Python standard library so it can run
on the RK3588 board without installing extra packages.
"""

import argparse
import concurrent.futures
import csv
import datetime as _dt
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


TOPICS = [
    "鲁迅",
    "李大钊",
    "陈独秀",
    "钱玄同",
    "胡适",
    "蔡元培",
    "闻一多",
    "朱自清",
]

REQUEST_FIELDS = [
    "run_id", "suite", "scenario", "mode", "repeat", "request_index",
    "concurrency", "stream", "max_tokens", "temperature", "prompt_kind",
    "topic", "shared_prefix_kind", "success", "error", "http_status",
    "response_id", "client_start_ts", "client_ttft_ms", "client_total_ms",
    "client_chunks", "client_chars", "server_prompt_tokens",
    "server_completion_tokens", "server_total_tokens", "jsonl_request_id",
    "jsonl_session_id", "jsonl_prompt_tokens", "jsonl_generated_tokens",
    "jsonl_cached_prefix_tokens", "jsonl_cached_prefix_blocks",
    "jsonl_computed_prefill_tokens", "jsonl_prefill_ms", "jsonl_decode_ms",
    "jsonl_first_token_ms", "jsonl_total_ms", "jsonl_tokens_per_second",
    "jsonl_paged_attention_calls", "jsonl_paged_attention_fallbacks",
    "jsonl_decode_batch_size_avg", "jsonl_decode_batch_size_max",
    "jsonl_prefill_microbatch_size_avg",
    "jsonl_prefill_microbatch_size_max",
    "jsonl_prefill_microbatch_executor", "match_method",
]

SUMMARY_FIELDS = [
    "run_id", "suite", "scenario", "mode", "repeat", "concurrency",
    "num_requests", "success_count", "fail_count", "abort_count",
    "total_wall_ms", "aggregate_output_tokens", "aggregate_output_tok_s",
    "p50_client_ttft_ms", "p95_client_ttft_ms", "p50_client_total_ms",
    "p95_client_total_ms", "avg_server_tps", "avg_prefill_ms",
    "avg_decode_ms", "avg_first_token_ms", "prefix_hit_requests",
    "prefix_cached_tokens_total", "prefix_cached_blocks_total",
    "paged_attention_calls", "paged_attention_fallbacks",
    "decode_batch_size_max", "avg_decode_batch_size",
    "prefill_batching_enabled", "prefill_microbatch_size_max",
    "avg_prefill_microbatch_size", "prefill_microbatch_tokens_total",
    "server_requests_total", "server_requests_finished",
    "server_requests_failed", "server_requests_aborted",
    "server_tokens_generated_total", "raw_log_path", "metrics_jsonl_path",
    "server_metrics_path",
]


def now_ms():
    return time.perf_counter() * 1000.0


def safe_mean(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def percentile(values, p):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (p / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def fmt(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compact_error(exc):
    text = str(exc)
    return text[:500] if len(text) > 500 else text


def http_json(url, timeout_s=10.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


def read_jsonl(path):
    items = []
    path = Path(path)
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def make_short_prompt(topic):
    return f"请用简洁的语言介绍{topic}。"


def make_medium_prompt(topic):
    return f"请详细介绍{topic}的生平、代表作品、主要思想、历史影响，并分点说明。"


def make_shared_system_prompt(repeat=20):
    base = (
        "你是一名严谨的中文助手，需要根据给定背景进行回答。"
        "请保持事实清楚、结构完整、语言简明。"
        "以下是背景说明：现代中国文学与思想史涉及多个重要人物、"
        "历史阶段、社会议题和文化转型。回答时需要先给出概括，"
        "再分点说明人物经历、主要贡献、代表作品、思想影响和历史评价。"
        "请避免空泛表达。"
    )
    return base * max(1, repeat)


def make_long_prompt(topic, repeat=20):
    return (
        make_shared_system_prompt(repeat)
        + f"\n请围绕{topic}展开，给出清晰、可核对、分层的说明。"
    )


def make_chat_messages(system_text=None, user_text="", history_turns=0):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for i in range(history_turns):
        messages.append({"role": "user", "content": f"第{i + 1}轮背景问题。"})
        messages.append({"role": "assistant", "content": "这是上一轮的简短回答。"})
    messages.append({"role": "user", "content": user_text})
    return messages


class ServerRunner:
    def __init__(self, binary, host, port, env, log_path, taskset=None, verbose=False):
        self.binary = str(binary)
        self.host = host
        self.port = int(port)
        self.env = dict(env)
        self.log_path = Path(log_path)
        self.taskset = taskset
        self.verbose = verbose
        self.proc = None
        self.log_file = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8", errors="replace")
        cmd = [self.binary]
        if self.taskset:
            cmd = ["taskset", "-c", self.taskset] + cmd
        if self.verbose:
            print("[BENCH] starting server:", " ".join(shlex.quote(x) for x in cmd))
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.env.items()})
        if os.name != "nt":
            self.proc = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        else:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

    def wait_health(self, timeout_s=60):
        deadline = time.time() + timeout_s
        last_error = None
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early returncode={self.proc.returncode} "
                    f"log={self.log_path}"
                )
            try:
                data = http_json(f"{self.base_url}/health", timeout_s=2.0)
                if isinstance(data, dict):
                    if self.verbose:
                        print("[BENCH] health ready", data)
                    return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.5)
        raise TimeoutError(f"health not ready: {last_error}; log={self.log_path}")

    def fetch_metrics(self):
        return http_json(f"{self.base_url}/metrics", timeout_s=5.0)

    def stop(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=8)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        if self.log_file:
            self.log_file.close()


class HttpClient:
    def __init__(self, base_url, request_timeout=180, raw_dir=None):
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.raw_dir = Path(raw_dir) if raw_dir else None

    def post_chat(
        self,
        messages,
        max_tokens,
        stream,
        temperature=0.0,
        top_p=1.0,
        seed=None,
        raw_name=None,
    ):
        payload = {
            "model": "qwen",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start_perf = now_ms()
        start_ts = _dt.datetime.now().isoformat(timespec="milliseconds")
        rec = {
            "success": False,
            "error": "",
            "http_status": "",
            "response_id": "",
            "client_start_ts": start_ts,
            "client_ttft_ms": None,
            "client_total_ms": None,
            "client_chunks": 0,
            "client_chars": 0,
            "server_prompt_tokens": "",
            "server_completion_tokens": "",
            "server_total_tokens": "",
        }
        raw_lines = []
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                rec["http_status"] = getattr(resp, "status", "")
                if stream:
                    first_perf = None
                    for raw in resp:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        raw_lines.append(line)
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            rec["success"] = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if first_perf is None:
                            first_perf = now_ms()
                            rec["client_ttft_ms"] = first_perf - start_perf
                        if not rec["response_id"]:
                            rec["response_id"] = chunk.get("id", "")
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            text = delta.get("content") or ""
                            rec["client_chars"] += len(text)
                            rec["client_chunks"] += 1
                    if not rec["success"]:
                        rec["success"] = rec["http_status"] == 200 and rec["client_chunks"] > 0
                    if rec["client_ttft_ms"] is None:
                        rec["client_ttft_ms"] = now_ms() - start_perf
                else:
                    text = resp.read().decode("utf-8", errors="replace")
                    raw_lines.append(text)
                    obj = json.loads(text)
                    rec["success"] = True
                    rec["response_id"] = obj.get("id", "")
                    usage = obj.get("usage") or {}
                    rec["server_prompt_tokens"] = usage.get("prompt_tokens", "")
                    rec["server_completion_tokens"] = usage.get("completion_tokens", "")
                    rec["server_total_tokens"] = usage.get("total_tokens", "")
                    choices = obj.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        rec["client_chars"] = len(msg.get("content") or "")
                    rec["client_ttft_ms"] = None
        except urllib.error.HTTPError as exc:
            rec["http_status"] = exc.code
            rec["error"] = compact_error(exc)
            try:
                raw_lines.append(exc.read().decode("utf-8", errors="replace"))
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            rec["error"] = compact_error(exc)
        finally:
            rec["client_total_ms"] = now_ms() - start_perf
            if raw_name and self.raw_dir:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = self.raw_dir / raw_name
                raw_path.write_text("\n".join(raw_lines), encoding="utf-8")
        return rec


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_root = Path(args.out_dir) / self.run_id
        self.raw_dir = self.out_root / "raw"
        self.csv_dir = self.out_root / "csv"
        self.json_dir = self.out_root / "json"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.summaries = []
        self.server_metrics_rows = []
        self.next_port = int(args.base_port)

    def log(self, *parts):
        if self.args.verbose:
            print("[BENCH]", *parts)

    def common_env(self, port, metrics_path):
        server_timeout_s = (
            self.args.server_request_timeout
            if self.args.server_request_timeout is not None
            else self.args.request_timeout
        )
        return {
            "LLM_ENABLE_SERVICE": "1",
            "LLM_ENABLE_SCHEDULER": "1",
            "LLM_ENABLE_HTTP_SERVER": "1",
            "LLM_HTTP_HOST": self.args.host,
            "LLM_HTTP_PORT": str(port),
            "LLM_NUM_THREADS": str(self.args.threads),
            "LLM_DEBUG_METRICS": "1",
            "LLM_METRICS_JSONL": str(metrics_path),
            "LLM_MAX_NEW_TOKENS": str(self.args.max_new_tokens),
            "LLM_HTTP_PREFIX_CACHE_FRIENDLY": "1",
            "LLM_OPENAI_STATELESS": "1",
            "LLM_SERVER_REQUEST_TIMEOUT_MS": str(max(1, int(server_timeout_s * 1000))),
        }

    def mode_env(self, mode, port, metrics_path):
        env = self.common_env(port, metrics_path)
        if mode in (
            "paged_session",
            "paged_attention",
            "prefix_cache",
            "continuous_batching",
            "prefill_batching",
            "full_stack",
        ):
            env.update({"LLM_PAGED_KV": "1", "LLM_ENABLE_SESSION_CACHE": "1"})
        if mode in (
            "paged_attention",
            "prefix_cache",
            "continuous_batching",
            "prefill_batching",
            "full_stack",
        ):
            env["LLM_ENABLE_PAGED_ATTENTION"] = "1"
        if mode in ("prefix_cache", "full_stack"):
            env.update({
                "LLM_ENABLE_PREFIX_CACHE": "1",
                "LLM_OPENAI_STATELESS": "1",
                "LLM_HTTP_PREFIX_CACHE_FRIENDLY": "1",
            })
        if mode in ("continuous_batching", "prefill_batching", "full_stack"):
            env.update({
                "LLM_ENABLE_CONTINUOUS_BATCHING": "1",
                "LLM_CONT_BATCH_MAX_ACTIVE_DECODE": "8",
                "LLM_CONT_BATCH_DECODE_FIRST": "1",
                "LLM_CONT_BATCH_PREFILL_WHEN_DECODE_EMPTY": "1",
                "LLM_CONT_BATCH_PREFILL_AFTER_DECODE": "0",
            })
        if mode in ("prefill_batching", "full_stack"):
            env.update({
                "LLM_ENABLE_PREFILL_BATCHING": "1",
                "LLM_PREFILL_MICROBATCH_MAX_REQUESTS": "4",
                "LLM_PREFILL_MICROBATCH_CHUNK_SIZE": "8",
                "LLM_PREFILL_MICROBATCH_EXECUTOR": "conservative",
                "LLM_PREFILL_MICROBATCH_AFTER_DECODE": "0",
            })
        if self.args.verbose:
            env.update({
                "LLM_CONT_BATCH_DEBUG": "1",
                "LLM_PREFILL_BATCH_DEBUG": "1",
                "LLM_DEBUG_PREFIX_CACHE": "1",
                "LLM_DEBUG_PAGED_ATTENTION": "1",
            })
        return env

    def run_server_scenario(
        self,
        suite,
        scenario,
        mode,
        repeat,
        concurrency,
        requests,
        stream,
    ):
        port = self.next_port
        self.next_port += 1
        metrics_path = self.raw_dir / f"{scenario}_r{repeat}_c{concurrency}_metrics.jsonl"
        log_path = self.raw_dir / f"{scenario}_r{repeat}_c{concurrency}_service.log"
        server_metrics_path = self.raw_dir / f"{scenario}_r{repeat}_c{concurrency}_server_metrics.json"
        env = self.mode_env(mode, port, metrics_path)
        runner = ServerRunner(
            self.args.binary,
            self.args.host,
            port,
            env,
            log_path,
            taskset=self.args.taskset,
            verbose=self.args.verbose,
        )
        wall_start = now_ms()
        server_metrics = {}
        try:
            self.log(f"starting scenario={scenario} mode={mode} port={port}")
            runner.start()
            runner.wait_health(timeout_s=90)
            client = HttpClient(
                runner.base_url,
                request_timeout=self.args.request_timeout,
                raw_dir=self.raw_dir,
            )
            scenario_records = self.send_requests(
                client, suite, scenario, mode, repeat, concurrency, requests, stream
            )
            try:
                server_metrics = runner.fetch_metrics()
            except Exception as exc:  # noqa: BLE001
                server_metrics = {"error": compact_error(exc)}
            write_json(server_metrics_path, server_metrics)
        except KeyboardInterrupt:
            runner.stop()
            raise
        except Exception as exc:  # noqa: BLE001
            scenario_records = [self.failure_record(
                suite, scenario, mode, repeat, concurrency, compact_error(exc)
            )]
        finally:
            if not self.args.no_kill:
                runner.stop()
        wall_ms = now_ms() - wall_start
        jsonl_items = read_jsonl(metrics_path)
        self.attach_jsonl_metrics(scenario_records, jsonl_items)
        self.records.extend(scenario_records)
        summary = self.summarize_scenario(
            suite,
            scenario,
            mode,
            repeat,
            concurrency,
            scenario_records,
            server_metrics,
            wall_ms,
            log_path,
            metrics_path,
            server_metrics_path,
        )
        self.summaries.append(summary)
        metrics_row = dict(server_metrics) if isinstance(server_metrics, dict) else {}
        metrics_row.update({
            "run_id": self.run_id,
            "suite": suite,
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "concurrency": concurrency,
            "path": str(server_metrics_path),
        })
        self.server_metrics_rows.append(metrics_row)

    def send_requests(self, client, suite, scenario, mode, repeat, concurrency, requests, stream):
        if concurrency <= 1:
            rows = []
            for i, req in enumerate(requests):
                rows.append(
                    self.send_one(client, suite, scenario, mode, repeat, i, 1, req, stream)
                )
            return rows
        barrier = threading.Barrier(concurrency)

        def worker(i):
            req = requests[i % len(requests)]
            barrier.wait()
            return self.send_one(
                client, suite, scenario, mode, repeat, i, concurrency, req, stream
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(worker, i) for i in range(concurrency)]
            rows = []
            for fut in concurrent.futures.as_completed(futs):
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    rows.append(self.failure_record(
                        suite, scenario, mode, repeat, concurrency, compact_error(exc)
                    ))
        rows.sort(key=lambda r: int(r.get("request_index", 0)))
        return rows

    def send_one(self, client, suite, scenario, mode, repeat, index, concurrency, req, stream):
        rec = client.post_chat(
            req["messages"],
            max_tokens=req.get("max_tokens", self.args.max_new_tokens),
            stream=stream,
            temperature=req.get("temperature", 0.0),
            top_p=req.get("top_p", 1.0),
            seed=req.get("seed"),
            raw_name=f"{scenario}_r{repeat}_c{concurrency}_{index}.raw",
        )
        rec.update({
            "run_id": self.run_id,
            "suite": suite,
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "request_index": index,
            "concurrency": concurrency,
            "stream": stream,
            "max_tokens": req.get("max_tokens", self.args.max_new_tokens),
            "temperature": req.get("temperature", 0.0),
            "prompt_kind": req.get("prompt_kind", ""),
            "topic": req.get("topic", ""),
            "shared_prefix_kind": req.get("shared_prefix_kind", ""),
        })
        return rec

    def failure_record(self, suite, scenario, mode, repeat, concurrency, error):
        return {
            "run_id": self.run_id,
            "suite": suite,
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "request_index": 0,
            "concurrency": concurrency,
            "stream": "",
            "max_tokens": "",
            "temperature": "",
            "prompt_kind": "",
            "topic": "",
            "shared_prefix_kind": "",
            "success": False,
            "error": error,
            "http_status": "",
            "response_id": "",
            "client_start_ts": "",
            "client_ttft_ms": None,
            "client_total_ms": None,
            "client_chunks": 0,
            "client_chars": 0,
        }

    def attach_jsonl_metrics(self, records, jsonl_items):
        for i, rec in enumerate(records):
            item = jsonl_items[i] if i < len(jsonl_items) else None
            if not item:
                rec["match_method"] = "none"
                continue
            rec["match_method"] = "order"
            rec.update({
                "jsonl_request_id": item.get("request_id", ""),
                "jsonl_session_id": item.get("session_id", ""),
                "jsonl_prompt_tokens": item.get("prompt_tokens", ""),
                "jsonl_generated_tokens": item.get("generated_tokens", ""),
                "jsonl_cached_prefix_tokens": item.get("cached_prefix_tokens", ""),
                "jsonl_cached_prefix_blocks": item.get("cached_prefix_blocks", ""),
                "jsonl_computed_prefill_tokens": item.get("computed_prefill_tokens", ""),
                "jsonl_prefill_ms": item.get("prefill_ms", ""),
                "jsonl_decode_ms": item.get("decode_ms", ""),
                "jsonl_first_token_ms": item.get("first_token_ms", ""),
                "jsonl_total_ms": item.get("total_ms", ""),
                "jsonl_tokens_per_second": item.get("tokens_per_second", ""),
                "jsonl_paged_attention_calls": item.get("paged_attention_calls", ""),
                "jsonl_paged_attention_fallbacks": item.get("paged_attention_fallbacks", ""),
                "jsonl_decode_batch_size_avg": item.get("decode_batch_size_avg", ""),
                "jsonl_decode_batch_size_max": item.get("decode_batch_size_max", ""),
                "jsonl_prefill_microbatch_size_avg": item.get("prefill_microbatch_size_avg", ""),
                "jsonl_prefill_microbatch_size_max": item.get("prefill_microbatch_size_max", ""),
                "jsonl_prefill_microbatch_executor": item.get("prefill_microbatch_executor", ""),
            })

    def summarize_scenario(
        self,
        suite,
        scenario,
        mode,
        repeat,
        concurrency,
        records,
        server_metrics,
        wall_ms,
        log_path,
        metrics_path,
        server_metrics_path,
    ):
        generated = [r.get("jsonl_generated_tokens") for r in records]
        generated = [int(v) for v in generated if isinstance(v, int) or str(v).isdigit()]
        ttfts = [r.get("client_ttft_ms") for r in records]
        totals = [r.get("client_total_ms") for r in records]
        prefill = [r.get("jsonl_prefill_ms") for r in records]
        decode = [r.get("jsonl_decode_ms") for r in records]
        first = [r.get("jsonl_first_token_ms") for r in records]
        tps = [r.get("jsonl_tokens_per_second") for r in records]
        success_count = sum(1 for r in records if r.get("success") is True)
        fail_count = len(records) - success_count
        server_metrics = server_metrics if isinstance(server_metrics, dict) else {}
        out_tokens = sum(generated)
        return {
            "run_id": self.run_id,
            "suite": suite,
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "concurrency": concurrency,
            "num_requests": len(records),
            "success_count": success_count,
            "fail_count": fail_count,
            "abort_count": int(server_metrics.get("requests_aborted", 0) or 0),
            "total_wall_ms": wall_ms,
            "aggregate_output_tokens": out_tokens,
            "aggregate_output_tok_s": (out_tokens * 1000.0 / wall_ms) if wall_ms > 0 else None,
            "p50_client_ttft_ms": percentile(ttfts, 50),
            "p95_client_ttft_ms": percentile(ttfts, 95),
            "p50_client_total_ms": percentile(totals, 50),
            "p95_client_total_ms": percentile(totals, 95),
            "avg_server_tps": safe_mean(tps) or server_metrics.get("avg_tokens_per_second", ""),
            "avg_prefill_ms": safe_mean(prefill),
            "avg_decode_ms": safe_mean(decode),
            "avg_first_token_ms": safe_mean(first) or server_metrics.get("avg_first_token_ms", ""),
            "prefix_hit_requests": server_metrics.get("prefix_hit_requests", 0),
            "prefix_cached_tokens_total": server_metrics.get("prefix_cached_tokens_total", 0),
            "prefix_cached_blocks_total": server_metrics.get("prefix_cached_blocks_total", 0),
            "paged_attention_calls": server_metrics.get("paged_attention_calls", 0),
            "paged_attention_fallbacks": server_metrics.get("paged_attention_fallbacks", 0),
            "decode_batch_size_max": server_metrics.get("decode_batch_size_max", 0),
            "avg_decode_batch_size": server_metrics.get("avg_decode_batch_size", 0),
            "prefill_batching_enabled": server_metrics.get("prefill_batching_enabled", False),
            "prefill_microbatch_size_max": server_metrics.get("prefill_microbatch_size_max", 0),
            "avg_prefill_microbatch_size": server_metrics.get("avg_prefill_microbatch_size", 0),
            "prefill_microbatch_tokens_total": server_metrics.get("prefill_microbatch_tokens_total", 0),
            "server_requests_total": server_metrics.get("requests_total", 0),
            "server_requests_finished": server_metrics.get("requests_finished", 0),
            "server_requests_failed": server_metrics.get("requests_failed", 0),
            "server_requests_aborted": server_metrics.get("requests_aborted", 0),
            "server_tokens_generated_total": server_metrics.get("tokens_generated_total", 0),
            "raw_log_path": str(log_path),
            "metrics_jsonl_path": str(metrics_path),
            "server_metrics_path": str(server_metrics_path),
        }

    def request_for_topic(self, topic, prompt_kind="medium", max_tokens=None, shared=False):
        if prompt_kind == "short":
            user = make_short_prompt(topic)
            messages = make_chat_messages(user_text=user)
        elif prompt_kind == "long":
            messages = make_chat_messages(user_text=make_long_prompt(topic, self.args.prompt_repeat))
        elif shared:
            messages = make_chat_messages(
                system_text=make_shared_system_prompt(self.args.prompt_repeat),
                user_text=make_medium_prompt(topic),
            )
        else:
            messages = make_chat_messages(user_text=make_medium_prompt(topic))
        return {
            "messages": messages,
            "max_tokens": max_tokens or self.args.max_new_tokens,
            "prompt_kind": prompt_kind,
            "topic": topic,
            "shared_prefix_kind": "shared_system" if shared else "",
            "temperature": 0.0,
            "top_p": 1.0,
        }

    def run_smoke(self):
        reqs = [
            self.request_for_topic("鲁迅", prompt_kind="short", max_tokens=32),
            self.request_for_topic("李大钊", prompt_kind="short", max_tokens=32),
        ]
        self.run_server_scenario("smoke", "smoke_nonstream", "baseline_service", 0, 1, [reqs[0]], False)
        self.run_server_scenario("smoke", "smoke_stream", "baseline_service", 0, 1, [reqs[1]], True)

    def run_prefix_cache(self):
        for rep in range(self.args.repeats):
            for mode, scenario in (
                ("paged_attention", "prefix_off"),
                ("prefix_cache", "prefix_on"),
            ):
                reqs = [
                    self.request_for_topic("鲁迅", "medium", 64, shared=True),
                    self.request_for_topic("鲁迅", "medium", 64, shared=True),
                    self.request_for_topic("李大钊", "medium", 64, shared=True),
                ]
                self.run_server_scenario(
                    "prefix-cache", scenario, mode, rep, 1, reqs, self.args.stream_prefix_cache
                )

    def run_continuous_batching(self):
        for rep in range(self.args.repeats):
            for conc in self.args.concurrency_values:
                reqs = [
                    self.request_for_topic(TOPICS[i % len(TOPICS)], "medium", 128)
                    for i in range(conc)
                ]
                self.run_server_scenario(
                    "continuous-batching", "v2_off", "paged_attention", rep, conc, reqs, True
                )
                self.run_server_scenario(
                    "continuous-batching", "v2_on", "continuous_batching", rep, conc, reqs, True
                )

    def run_prefill_batching(self):
        for rep in range(self.args.repeats):
            for conc in self.args.concurrency_values:
                reqs = [
                    self.request_for_topic(TOPICS[i % len(TOPICS)], "long", 64)
                    for i in range(conc)
                ]
                self.run_server_scenario(
                    "prefill-batching", "prefill_batch_off",
                    "continuous_batching", rep, conc, reqs, True
                )
                self.run_server_scenario(
                    "prefill-batching", "prefill_batch_on",
                    "prefill_batching", rep, conc, reqs, True
                )

    def run_paged_attention(self):
        for rep in range(self.args.repeats):
            for context_repeat in (5, 10, 20, 40):
                old = self.args.prompt_repeat
                self.args.prompt_repeat = context_repeat
                req = self.request_for_topic("鲁迅", "long", 64)
                self.args.prompt_repeat = old
                self.run_server_scenario(
                    "paged-attention", f"paged_attention_off_ctx{context_repeat}",
                    "paged_session", rep, 1, [req], True
                )
                self.run_server_scenario(
                    "paged-attention", f"paged_attention_on_ctx{context_repeat}",
                    "paged_attention", rep, 1, [req], True
                )

    def run_full(self):
        modes = [
            ("service_baseline", "baseline_service"),
            ("paged_session", "paged_session"),
            ("paged_attention", "paged_attention"),
            ("prefix_cache", "prefix_cache"),
            ("continuous_batching", "continuous_batching"),
            ("full_stack", "full_stack"),
        ]
        conc = self.args.concurrency_values[0] if self.args.concurrency_values else 4
        reqs = []
        for i in range(conc):
            topic = TOPICS[i % len(TOPICS)]
            if i % 4 == 0:
                reqs.append(self.request_for_topic(topic, "short", 32))
            elif i % 4 == 1:
                reqs.append(self.request_for_topic(topic, "medium", 64))
            elif i % 4 == 2:
                reqs.append(self.request_for_topic(topic, "medium", 64, shared=True))
            else:
                reqs.append(self.request_for_topic(topic, "long", 128))
        for rep in range(self.args.repeats):
            for scenario, mode in modes:
                self.run_server_scenario("full", scenario, mode, rep, conc, reqs, True)

    def run(self):
        suites = [self.args.suite]
        if self.args.suite == "all":
            suites = [
                "smoke", "prefix-cache", "continuous-batching",
                "prefill-batching", "paged-attention", "full",
            ]
        for suite in suites:
            if suite == "smoke":
                self.run_smoke()
            elif suite == "prefix-cache":
                self.run_prefix_cache()
            elif suite == "continuous-batching":
                self.run_continuous_batching()
            elif suite == "prefill-batching":
                self.run_prefill_batching()
            elif suite == "paged-attention":
                self.run_paged_attention()
            elif suite == "full":
                self.run_full()
            else:
                raise ValueError(f"unknown suite: {suite}")
        self.write_outputs()

    def write_outputs(self):
        write_csv(self.csv_dir / "requests.csv", self.records, REQUEST_FIELDS)
        write_csv(self.csv_dir / "scenario_summary.csv", self.summaries, SUMMARY_FIELDS)
        if self.server_metrics_rows:
            server_fields = sorted({k for row in self.server_metrics_rows for k in row.keys()})
            write_csv(self.csv_dir / "server_metrics.csv", self.server_metrics_rows, server_fields)
        write_json(self.json_dir / "request_records.json", self.records)
        write_json(self.json_dir / "scenario_summary.json", self.summaries)
        self.write_report()
        print(f"[BENCH] wrote output_dir={self.out_root}")

    def write_report(self):
        lines = []
        lines.append("# Serving Engine Benchmark Report")
        lines.append("")
        lines.append(f"- run_id: `{self.run_id}`")
        lines.append(f"- binary: `{self.args.binary}`")
        lines.append(f"- suite: `{self.args.suite}`")
        lines.append(f"- threads: `{self.args.threads}`")
        lines.append(f"- taskset: `{self.args.taskset or ''}`")
        lines.append("")
        lines.append("## Scenario Summary")
        lines.append("")
        cols = [
            "suite", "scenario", "mode", "repeat", "concurrency",
            "success_count", "fail_count", "p95_client_ttft_ms",
            "aggregate_output_tok_s", "decode_batch_size_max",
            "prefill_microbatch_size_max", "paged_attention_fallbacks",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in self.summaries:
            lines.append("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
        lines.append("")
        lines.append("## Findings")
        lines.extend(self.report_findings())
        lines.append("")
        lines.append("## Output Files")
        lines.append(f"- requests: `{self.csv_dir / 'requests.csv'}`")
        lines.append(f"- scenario summary: `{self.csv_dir / 'scenario_summary.csv'}`")
        lines.append(f"- server metrics: `{self.csv_dir / 'server_metrics.csv'}`")
        lines.append(f"- request json: `{self.json_dir / 'request_records.json'}`")
        lines.append(f"- summary json: `{self.json_dir / 'scenario_summary.json'}`")
        lines.append("")
        lines.append("## Notes")
        lines.append("- Python client timing includes HTTP, socket, JSON parsing and SSE parsing overhead.")
        lines.append("- stream=false has no real client-side TTFT; the script leaves TTFT empty for it.")
        lines.append("- Prefill batching currently validates scheduler-level conservative executor behavior, not operator-level batch prefill speedup.")
        lines.append("- Client records and JSONL request metrics are matched by completion order as a best effort.")
        (self.out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def report_findings(self):
        lines = []
        prefix_on = [s for s in self.summaries if s["scenario"] == "prefix_on"]
        if prefix_on:
            best = max(prefix_on, key=lambda x: int(x.get("prefix_cached_tokens_total") or 0))
            hit = best.get("prefix_cached_tokens_total", 0)
            if hit:
                lines.append(f"- Prefix cache: warm requests cached {hit} tokens in `{best['metrics_jsonl_path']}`.")
            else:
                lines.append(f"- Prefix cache: no cached tokens observed; inspect `{best['metrics_jsonl_path']}` and `{best['raw_log_path']}`.")
        cont = [s for s in self.summaries if s["scenario"] == "v2_on"]
        if cont:
            best = max(cont, key=lambda x: int(x.get("decode_batch_size_max") or 0))
            lines.append(f"- Continuous batching: max decode batch size observed = {best.get('decode_batch_size_max', 0)}.")
        prefill = [s for s in self.summaries if s["scenario"] == "prefill_batch_on"]
        if prefill:
            best = max(prefill, key=lambda x: int(x.get("prefill_microbatch_size_max") or 0))
            lines.append(
                "- Prefill batching: "
                f"max microbatch size = {best.get('prefill_microbatch_size_max', 0)}, "
                f"avg = {fmt(best.get('avg_prefill_microbatch_size'))}, "
                f"tokens = {best.get('prefill_microbatch_tokens_total', 0)}; "
                "executor=conservative."
            )
        paged = [s for s in self.summaries if "paged_attention" in s["scenario"] or s["mode"] == "paged_attention"]
        if paged:
            fallbacks = sum(int(s.get("paged_attention_fallbacks") or 0) for s in paged)
            calls = sum(int(s.get("paged_attention_calls") or 0) for s in paged)
            lines.append(f"- Paged attention: calls={calls}, fallbacks={fallbacks}.")
        if not lines:
            lines.append("- No feature-specific scenarios were run.")
        return lines


def parse_concurrency(text, default):
    if not text:
        return default
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(max(1, int(part)))
    return values or default


def build_arg_parser():
    p = argparse.ArgumentParser(description="Benchmark the LLM HTTP serving engine.")
    p.add_argument("--binary", default="./qwen_model_loader")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--base-port", type=int, default=18080)
    p.add_argument("--out-dir", default="bench_results")
    p.add_argument(
        "--suite",
        default="smoke",
        choices=[
            "smoke", "prefix-cache", "continuous-batching", "prefill-batching",
            "paged-attention", "full", "all",
        ],
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--taskset", default="")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", default="")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument(
        "--server-request-timeout",
        type=int,
        default=None,
        help="server-side request timeout seconds; defaults to --request-timeout",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prompt-repeat", type=int, default=20)
    p.add_argument("--stream-prefix-cache", action="store_true")
    p.add_argument("--keep-logs", action="store_true", default=True)
    p.add_argument("--no-kill", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.suite == "continuous-batching":
        default_conc = [1, 2, 4, 8]
    elif args.suite == "prefill-batching":
        default_conc = [2, 4, 8]
    elif args.suite == "full":
        default_conc = [4]
    else:
        default_conc = [1]
    args.concurrency_values = parse_concurrency(args.concurrency, default_conc)
    runner = BenchmarkRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[BENCH] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
