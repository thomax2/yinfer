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
    "jsonl_prefill_microbatch_executor", "jsonl_final_status",
    "jsonl_selective_decode_enabled", "jsonl_selective_decode_steps",
    "jsonl_selective_decode_size_avg", "jsonl_selective_decode_size_max",
    "jsonl_selective_decode_linear_batch_rows",
    "jsonl_selective_decode_attention_per_sequence_calls",
    "jsonl_selective_decode_lm_head_rows",
    "jsonl_selective_decode_fallbacks",
    "jsonl_selective_decode_model_ms",
    "jsonl_selective_decode_mode",
    "jsonl_error_message", "match_method",
    "valid_effective_run", "validity_warnings",
]

SUMMARY_FIELDS = [
    "run_id", "suite", "scenario", "mode", "repeat", "concurrency",
    "num_requests", "success_count", "fail_count", "abort_count",
    "total_wall_ms", "aggregate_output_tokens", "aggregate_output_tok_s",
    "generated_tokens_per_request_avg", "decode_ms_per_token_avg",
    "selective_model_ms_per_token_avg", "total_ms_per_output_token",
    "selective_batch_utilization",
    "p50_client_ttft_ms", "p95_client_ttft_ms", "p50_client_total_ms",
    "p95_client_total_ms", "avg_server_tps", "avg_prefill_ms",
    "avg_decode_ms", "avg_first_token_ms", "prefix_hit_requests",
    "prefix_cached_tokens_total", "prefix_cached_blocks_total",
    "paged_attention_calls", "paged_attention_fallbacks",
    "decode_batch_size_max", "avg_decode_batch_size",
    "prefill_batching_enabled", "prefill_microbatch_size_max",
    "avg_prefill_microbatch_size", "prefill_microbatch_tokens_total",
    "selective_decode_enabled", "selective_decode_size_max",
    "avg_selective_decode_size", "selective_decode_steps_total",
    "selective_decode_linear_batch_rows_total",
    "selective_decode_attention_per_sequence_calls_total",
    "selective_decode_lm_head_rows_total",
    "selective_decode_fallbacks_total",
    "selective_decode_model_ms_total",
    "valid_effective_run", "validity_warnings",
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


def to_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


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


def make_decode_heavy_prompt(topic, target_items=200, prompt_kind="numbered-list"):
    if prompt_kind == "short-lines":
        return (
            f"请连续写 {target_items} 行短句，每行以编号开头，主题是：{topic}。\n"
            "不要提前结束，不要总结，不要反问。"
        )
    return (
        f"请严格按照编号列表连续输出 1 到 {target_items} 条内容。\n"
        f"主题是：{topic}。\n"
        "每一条只写一句简短说明。\n"
        "不要总结。\n"
        "不要提前结束。\n"
        "不要输出结束语。\n"
        "格式如下：\n"
        "1. ...\n"
        "2. ...\n"
        "3. ..."
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
                        if "error" in chunk:
                            err = chunk.get("error") or {}
                            if isinstance(err, dict):
                                rec["error"] = err.get("message", "server error")
                            else:
                                rec["error"] = str(err)
                            rec["success"] = False
                            break
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
                    if not rec["success"] and not rec["error"]:
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
        self.expected_scenarios = 0
        self.completed_scenarios = 0

    def log(self, *parts):
        if self.args.verbose:
            print("[BENCH]", *parts)

    def progress(self, *parts):
        print("[BENCH]", *parts, flush=True)

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
            "selective_decode_off",
            "selective_decode_on",
            "selective_decode",
            "full_stack",
        ):
            env.update({"LLM_PAGED_KV": "1", "LLM_ENABLE_SESSION_CACHE": "1"})
        if mode in (
            "paged_attention",
            "prefix_cache",
            "continuous_batching",
            "prefill_batching",
            "selective_decode_off",
            "selective_decode_on",
            "selective_decode",
            "full_stack",
        ):
            env["LLM_ENABLE_PAGED_ATTENTION"] = "1"
        if mode in ("prefix_cache", "full_stack"):
            env.update({
                "LLM_ENABLE_PREFIX_CACHE": "1",
                "LLM_OPENAI_STATELESS": "1",
                "LLM_HTTP_PREFIX_CACHE_FRIENDLY": "1",
            })
        if mode in (
            "continuous_batching",
            "prefill_batching",
            "selective_decode_off",
            "selective_decode_on",
            "selective_decode",
            "full_stack",
        ):
            env.update({
                "LLM_ENABLE_CONTINUOUS_BATCHING": "1",
                "LLM_CONT_BATCH_MAX_ACTIVE_DECODE": "8",
                "LLM_CONT_BATCH_DECODE_FIRST": "1",
                "LLM_CONT_BATCH_PREFILL_WHEN_DECODE_EMPTY": "1",
                "LLM_CONT_BATCH_PREFILL_AFTER_DECODE": "0",
            })
        if mode == "selective_decode_off":
            env["LLM_ENABLE_SELECTIVE_BATCH_DECODE"] = "0"
        if mode in ("selective_decode_on", "selective_decode", "full_stack"):
            env.update({
                "LLM_ENABLE_SELECTIVE_BATCH_DECODE": "1",
                "LLM_SELECTIVE_DECODE_MAX_BATCH": str(max(1, self.args.selective_decode_max_batch)),
                "LLM_SELECTIVE_DECODE_MIN_BATCH": str(max(1, self.args.selective_decode_min_batch)),
                "LLM_SELECTIVE_DECODE_GREEDY_ONLY": "1",
                "LLM_SELECTIVE_DECODE_ALLOW_SAMPLING": "0",
                "LLM_SELECTIVE_DECODE_FALLBACK": "1",
                "LLM_SELECTIVE_DECODE_COMPARE": "0",
            })
        if mode in ("prefill_batching", "full_stack"):
            env.update({
                "LLM_ENABLE_PREFILL_BATCHING": "1",
                "LLM_PREFILL_MICROBATCH_MAX_REQUESTS": "4",
                "LLM_PREFILL_MICROBATCH_CHUNK_SIZE": "8",
                "LLM_PREFILL_MICROBATCH_EXECUTOR": "conservative",
                "LLM_PREFILL_MICROBATCH_AFTER_DECODE": "0",
            })
        if self.args.engine_debug:
            env.update({
                "LLM_CONT_BATCH_DEBUG": "1",
                "LLM_PREFILL_BATCH_DEBUG": "1",
                "LLM_DEBUG_PREFIX_CACHE": "1",
                "LLM_DEBUG_PAGED_ATTENTION": "1",
                "LLM_SELECTIVE_DECODE_DEBUG": "1",
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
        next_index = self.completed_scenarios + 1
        total = self.expected_scenarios or "?"
        self.progress(
            f"scenario {next_index}/{total} start",
            f"suite={suite}",
            f"scenario={scenario}",
            f"mode={mode}",
            f"repeat={repeat}",
            f"concurrency={concurrency}",
        )
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
        validation = self.validate_selective_decode_summary(summary, scenario_records)
        summary.update(validation)
        for rec in scenario_records:
            rec["valid_effective_run"] = validation["valid_effective_run"]
            rec["validity_warnings"] = validation["validity_warnings"]
        self.records.extend(scenario_records)
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
        self.completed_scenarios += 1
        self.progress(
            f"scenario {self.completed_scenarios}/{total} done",
            f"suite={suite}",
            f"scenario={scenario}",
            f"repeat={repeat}",
            f"concurrency={concurrency}",
            f"success={summary.get('success_count')}",
            f"fail={summary.get('fail_count')}",
        )
        try:
            self.write_outputs(final=False)
        except Exception as exc:  # noqa: BLE001
            self.progress(f"incremental output failed: {compact_error(exc)}")

    def send_requests(self, client, suite, scenario, mode, repeat, concurrency, requests, stream):
        if concurrency <= 1:
            rows = []
            for i, req in enumerate(requests):
                rows.append(
                    self.send_one(client, suite, scenario, mode, repeat, i, 1, req, stream)
                )
            return rows
        barrier = threading.Barrier(concurrency)
        scenario_timeout = max(1.0, float(self.args.timeout))

        def worker(i):
            req = requests[i % len(requests)]
            barrier.wait(timeout=30.0)
            return self.send_one(
                client, suite, scenario, mode, repeat, i, concurrency, req, stream
            )

        self.log(
            f"dispatch scenario={scenario} mode={mode} repeat={repeat} "
            f"concurrency={concurrency} timeout_s={scenario_timeout}"
        )
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
        fut_to_index = {pool.submit(worker, i): i for i in range(concurrency)}
        rows = []
        try:
            done, pending = concurrent.futures.wait(
                fut_to_index.keys(),
                timeout=scenario_timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for fut in done:
                idx = fut_to_index[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    rows.append(self.failure_record(
                        suite, scenario, mode, repeat, concurrency, compact_error(exc), idx
                    ))
            for fut in pending:
                idx = fut_to_index[fut]
                fut.cancel()
                rows.append(self.failure_record(
                    suite,
                    scenario,
                    mode,
                    repeat,
                    concurrency,
                    f"scenario request timeout after {scenario_timeout:.1f}s",
                    idx,
                ))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        seen = {int(r.get("request_index", 0)) for r in rows}
        for i in range(concurrency):
            if i not in seen:
                rows.append(self.failure_record(
                    suite,
                    scenario,
                    mode,
                    repeat,
                    concurrency,
                    "missing request result",
                    i,
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

    def failure_record(self, suite, scenario, mode, repeat, concurrency, error, request_index=0):
        return {
            "run_id": self.run_id,
            "suite": suite,
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "request_index": request_index,
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
            "valid_effective_run": "",
            "validity_warnings": "",
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
                "jsonl_selective_decode_enabled": item.get("selective_decode_enabled", ""),
                "jsonl_selective_decode_steps": item.get("selective_decode_steps", ""),
                "jsonl_selective_decode_size_avg": item.get("selective_decode_size_avg", ""),
                "jsonl_selective_decode_size_max": item.get("selective_decode_size_max", ""),
                "jsonl_selective_decode_linear_batch_rows":
                    item.get("selective_decode_linear_batch_rows", ""),
                "jsonl_selective_decode_attention_per_sequence_calls":
                    item.get("selective_decode_attention_per_sequence_calls", ""),
                "jsonl_selective_decode_lm_head_rows":
                    item.get("selective_decode_lm_head_rows", ""),
                "jsonl_selective_decode_fallbacks": item.get("selective_decode_fallbacks", ""),
                "jsonl_selective_decode_model_ms": item.get("selective_decode_model_ms", ""),
                "jsonl_selective_decode_mode": item.get("selective_decode_mode", ""),
                "jsonl_final_status": item.get("final_status", ""),
                "jsonl_error_message": item.get("error_message", ""),
            })
            final_status = item.get("final_status", "")
            generated = int(item.get("generated_tokens") or 0)
            if final_status and final_status != "FINISHED":
                rec["success"] = False
                rec["error"] = item.get("error_message") or final_status
            elif generated <= 0:
                rec["success"] = False
                if not rec.get("error"):
                    rec["error"] = "no generated tokens"

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
        generated_avg = (out_tokens / len(records)) if records else None
        decode_ms_per_token_values = []
        for r in records:
            decode_ms = to_float(r.get("jsonl_decode_ms"))
            gen = to_int(r.get("jsonl_generated_tokens"), 0)
            if decode_ms is not None and gen > 0:
                decode_ms_per_token_values.append(decode_ms / gen)
        selective_model_ms = to_float(
            server_metrics.get("selective_decode_model_ms_total"),
            0.0,
        )
        avg_selective_decode_size = to_float(
            server_metrics.get("avg_selective_decode_size"),
            0.0,
        )
        selective_util = (
            avg_selective_decode_size / max(1, self.args.selective_decode_max_batch)
            if avg_selective_decode_size is not None
            else None
        )
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
            "generated_tokens_per_request_avg": generated_avg,
            "decode_ms_per_token_avg": safe_mean(decode_ms_per_token_values),
            "selective_model_ms_per_token_avg":
                (selective_model_ms / out_tokens) if out_tokens > 0 else None,
            "total_ms_per_output_token": (wall_ms / out_tokens) if out_tokens > 0 else None,
            "selective_batch_utilization": selective_util,
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
            "selective_decode_enabled": server_metrics.get("selective_decode_enabled", False),
            "selective_decode_size_max": server_metrics.get("selective_decode_size_max", 0),
            "avg_selective_decode_size": server_metrics.get("avg_selective_decode_size", 0),
            "selective_decode_steps_total": server_metrics.get("selective_decode_steps_total", 0),
            "selective_decode_linear_batch_rows_total":
                server_metrics.get("selective_decode_linear_batch_rows_total", 0),
            "selective_decode_attention_per_sequence_calls_total":
                server_metrics.get("selective_decode_attention_per_sequence_calls_total", 0),
            "selective_decode_lm_head_rows_total":
                server_metrics.get("selective_decode_lm_head_rows_total", 0),
            "selective_decode_fallbacks_total":
                server_metrics.get("selective_decode_fallbacks_total", 0),
            "selective_decode_model_ms_total":
                server_metrics.get("selective_decode_model_ms_total", 0),
            "server_requests_total": server_metrics.get("requests_total", 0),
            "server_requests_finished": server_metrics.get("requests_finished", 0),
            "server_requests_failed": server_metrics.get("requests_failed", 0),
            "server_requests_aborted": server_metrics.get("requests_aborted", 0),
            "server_tokens_generated_total": server_metrics.get("tokens_generated_total", 0),
            "raw_log_path": str(log_path),
            "metrics_jsonl_path": str(metrics_path),
            "server_metrics_path": str(server_metrics_path),
        }

    def validate_selective_decode_summary(self, summary, records):
        if summary.get("suite") != "selective-decode" or not self.args.validate_effective:
            return {"valid_effective_run": "", "validity_warnings": ""}

        warnings = []
        valid = True
        concurrency = to_int(summary.get("concurrency"), 1)
        max_new_tokens = max(1, int(self.args.max_new_tokens))
        generated_total = to_int(summary.get("server_tokens_generated_total"), 0)
        expected_min_tokens = int(0.8 * concurrency * max_new_tokens)

        failed = to_int(summary.get("server_requests_failed"), 0)
        aborted = to_int(summary.get("server_requests_aborted"), 0)
        client_fail_count = to_int(summary.get("fail_count"), 0)
        if client_fail_count != 0:
            valid = False
            warnings.append(f"client_fail_count={client_fail_count}")
        if failed != 0:
            valid = False
            warnings.append(f"requests_failed={failed}")
        if aborted != 0:
            valid = False
            warnings.append(f"requests_aborted={aborted}")
        if generated_total < expected_min_tokens:
            valid = False
            warnings.append(
                f"generated_tokens_low={generated_total}<expected_min={expected_min_tokens}"
            )

        selective_enabled = to_bool(summary.get("selective_decode_enabled"))
        selective_size_max = to_float(summary.get("selective_decode_size_max"), 0.0)
        avg_selective_size = to_float(summary.get("avg_selective_decode_size"), 0.0)
        decode_batch_size_max = to_float(summary.get("decode_batch_size_max"), 0.0)
        paged_fallbacks = to_int(summary.get("paged_attention_fallbacks"), 0)
        selective_fallbacks = to_int(summary.get("selective_decode_fallbacks_total"), 0)
        selective_steps = to_int(summary.get("selective_decode_steps_total"), 0)

        if summary.get("scenario") == "selective_decode_on":
            if not selective_enabled:
                valid = False
                warnings.append("selective_decode_enabled=false")
            if concurrency >= 2 and selective_size_max < 2:
                valid = False
                warnings.append(f"selective_decode_size_max={selective_size_max}<2")
            if concurrency >= 2 and avg_selective_size <= 1.0:
                valid = False
                warnings.append(f"avg_selective_decode_size={avg_selective_size}<=1")
            if to_int(summary.get("selective_decode_linear_batch_rows_total"), 0) <= 0:
                valid = False
                warnings.append("selective_decode_linear_batch_rows_total=0")
            if to_int(summary.get("selective_decode_attention_per_sequence_calls_total"), 0) <= 0:
                valid = False
                warnings.append("selective_decode_attention_per_sequence_calls_total=0")
            if to_int(summary.get("selective_decode_lm_head_rows_total"), 0) <= 0:
                valid = False
                warnings.append("selective_decode_lm_head_rows_total=0")
            if paged_fallbacks != 0:
                valid = False
                warnings.append(f"paged_attention_fallbacks={paged_fallbacks}")
            if selective_steps > 0 and selective_fallbacks > max(2, int(0.25 * selective_steps)):
                warnings.append(
                    f"selective_decode_fallbacks_high={selective_fallbacks}/{selective_steps}"
                )
        elif summary.get("scenario") == "selective_decode_off":
            if selective_enabled:
                valid = False
                warnings.append("selective_decode_enabled=true_in_off_mode")
            if selective_size_max not in (0, 0.0):
                valid = False
                warnings.append(f"selective_decode_size_max={selective_size_max}_in_off_mode")
            if concurrency >= 2 and decode_batch_size_max < 2:
                valid = False
                warnings.append(f"decode_batch_size_max={decode_batch_size_max}<2")

        if records and all(r.get("match_method") == "none" for r in records):
            warnings.append("jsonl_metrics_not_matched")

        return {
            "valid_effective_run": bool(valid),
            "validity_warnings": "; ".join(warnings),
        }

    def request_for_topic(self, topic, prompt_kind="medium", max_tokens=None, shared=False):
        if prompt_kind == "short":
            user = make_short_prompt(topic)
            messages = make_chat_messages(user_text=user)
        elif prompt_kind == "decode-heavy":
            target_items = 200 if self.args.force_long_output else 80
            messages = make_chat_messages(
                user_text=make_decode_heavy_prompt(
                    topic,
                    target_items=target_items,
                    prompt_kind=self.args.decode_heavy_prompt,
                )
            )
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

    def run_selective_decode(self):
        self.expected_scenarios = (
            len(self.args.concurrency_values) *
            max(0, int(self.args.repeats)) *
            2
        )
        self.progress(
            "selective-decode plan",
            f"repeats={self.args.repeats}",
            f"concurrency={','.join(str(x) for x in self.args.concurrency_values)}",
            f"expected_scenarios={self.expected_scenarios}",
        )
        for rep in range(self.args.repeats):
            self.progress(f"selective-decode repeat {rep + 1}/{self.args.repeats} start")
            for conc in self.args.concurrency_values:
                reqs = [
                    self.request_for_topic(
                        TOPICS[i % len(TOPICS)],
                        "decode-heavy",
                        self.args.max_new_tokens,
                    )
                    for i in range(conc)
                ]
                self.run_server_scenario(
                    "selective-decode", "selective_decode_off",
                    "selective_decode_off", rep, conc, reqs, True
                )
                self.run_server_scenario(
                    "selective-decode", "selective_decode_on",
                    "selective_decode_on", rep, conc, reqs, True
                )
            self.progress(f"selective-decode repeat {rep + 1}/{self.args.repeats} done")

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
            ("selective_decode", "selective_decode"),
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
                "prefill-batching", "selective-decode", "paged-attention", "full",
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
            elif suite == "selective-decode":
                self.run_selective_decode()
            elif suite == "paged-attention":
                self.run_paged_attention()
            elif suite == "full":
                self.run_full()
            else:
                raise ValueError(f"unknown suite: {suite}")
        self.write_outputs(final=True)

    def write_outputs(self, final=True):
        write_csv(self.csv_dir / "requests.csv", self.records, REQUEST_FIELDS)
        write_csv(self.csv_dir / "scenario_summary.csv", self.summaries, SUMMARY_FIELDS)
        if self.server_metrics_rows:
            server_fields = sorted({k for row in self.server_metrics_rows for k in row.keys()})
            write_csv(self.csv_dir / "server_metrics.csv", self.server_metrics_rows, server_fields)
        write_json(self.json_dir / "request_records.json", self.records)
        write_json(self.json_dir / "scenario_summary.json", self.summaries)
        try:
            self.write_report()
        except Exception as exc:  # noqa: BLE001
            error_path = self.out_root / "report_error.txt"
            error_path.write_text(compact_error(exc) + "\n", encoding="utf-8")
            self.progress(f"report generation failed: {compact_error(exc)}")
        if final:
            print(f"[BENCH] wrote output_dir={self.out_root}")
            print("")
            print("Recommended selective-decode commands:")
            for line in self.recommended_commands():
                print(line)

    def write_report(self):
        lines = []
        lines.append("# Serving Engine Benchmark Report")
        lines.append("")
        lines.append(f"- run_id: `{self.run_id}`")
        lines.append(f"- binary: `{self.args.binary}`")
        lines.append(f"- suite: `{self.args.suite}`")
        lines.append(f"- threads: `{self.args.threads}`")
        lines.append(f"- taskset: `{self.args.taskset or ''}`")
        lines.append(f"- completed_scenarios: `{self.completed_scenarios}`")
        lines.append(f"- expected_scenarios: `{self.expected_scenarios or ''}`")
        lines.append("")
        lines.append("## Scenario Summary")
        lines.append("")
        cols = [
            "suite", "scenario", "mode", "repeat", "concurrency",
            "success_count", "fail_count", "p95_client_ttft_ms",
            "aggregate_output_tok_s", "decode_batch_size_max",
            "selective_decode_size_max", "prefill_microbatch_size_max",
            "paged_attention_fallbacks",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in self.summaries:
            lines.append("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
        lines.append("")
        lines.append("## Findings")
        lines.extend(self.report_findings())
        selective_lines = self.selective_decode_report_lines()
        if selective_lines:
            lines.append("")
            lines.extend(selective_lines)
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
        lines.append("")
        lines.append("## Recommended Commands")
        lines.extend(self.recommended_commands(markdown=True))
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
        selective = [s for s in self.summaries if s["scenario"] == "selective_decode_on"]
        if selective:
            best = max(selective, key=lambda x: int(x.get("selective_decode_size_max") or 0))
            lines.append(
                "- Selective decode: "
                f"max batch size = {best.get('selective_decode_size_max', 0)}, "
                f"avg = {fmt(best.get('avg_selective_decode_size'))}, "
                f"fallbacks = {best.get('selective_decode_fallbacks_total', 0)}."
            )
        paged = [s for s in self.summaries if "paged_attention" in s["scenario"] or s["mode"] == "paged_attention"]
        if paged:
            fallbacks = sum(int(s.get("paged_attention_fallbacks") or 0) for s in paged)
            calls = sum(int(s.get("paged_attention_calls") or 0) for s in paged)
            lines.append(f"- Paged attention: calls={calls}, fallbacks={fallbacks}.")
        if not lines:
            lines.append("- No feature-specific scenarios were run.")
        return lines

    def selective_decode_report_lines(self):
        rows = [s for s in self.summaries if s.get("suite") == "selective-decode"]
        if not rows:
            return []

        lines = []
        lines.append("## Selective Batch Decode Summary")
        lines.append("")
        lines.append("### Experiment Settings")
        lines.append(f"- threads: `{self.args.threads}`")
        lines.append(f"- taskset: `{self.args.taskset or ''}`")
        lines.append(f"- max_new_tokens: `{self.args.max_new_tokens}`")
        lines.append(f"- concurrency: `{','.join(str(x) for x in self.args.concurrency_values)}`")
        lines.append(f"- selective_decode_max_batch: `{self.args.selective_decode_max_batch}`")
        lines.append(f"- selective_decode_min_batch: `{self.args.selective_decode_min_batch}`")
        lines.append(f"- prompt type: `{self.args.decode_heavy_prompt}`")
        lines.append("")

        valid = [s for s in rows if s.get("valid_effective_run") is True]
        invalid = [s for s in rows if s.get("valid_effective_run") is False]
        lines.append("### Validity")
        lines.append(f"- valid scenarios: `{len(valid)}`")
        lines.append(f"- invalid scenarios: `{len(invalid)}`")
        for row in invalid:
            lines.append(
                "- invalid: "
                f"scenario=`{row.get('scenario')}` "
                f"repeat=`{row.get('repeat')}` "
                f"concurrency=`{row.get('concurrency')}` "
                f"warnings=`{row.get('validity_warnings')}`"
            )
        lines.append("")

        lines.append("### On/Off Comparison")
        cols = [
            "concurrency", "off_tok_s", "on_tok_s", "speedup_pct",
            "off_p95_total_ms", "on_p95_total_ms",
            "selective_decode_size_max", "avg_selective_decode_size",
            "paged_attention_fallbacks",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        by_key = {}
        for row in rows:
            key = (row.get("repeat"), row.get("concurrency"))
            by_key.setdefault(key, {})[row.get("scenario")] = row
        for (_rep, conc), pair in sorted(by_key.items(), key=lambda x: (to_int(x[0][0]), to_int(x[0][1]))):
            off = pair.get("selective_decode_off")
            on = pair.get("selective_decode_on")
            if not off or not on:
                continue
            off_tps = to_float(off.get("aggregate_output_tok_s"))
            on_tps = to_float(on.get("aggregate_output_tok_s"))
            speedup = None
            if off_tps and off_tps > 0 and on_tps is not None:
                speedup = (on_tps / off_tps - 1.0) * 100.0
            vals = [
                conc,
                fmt(off_tps),
                fmt(on_tps),
                fmt(speedup),
                fmt(off.get("p95_client_total_ms")),
                fmt(on.get("p95_client_total_ms")),
                fmt(on.get("selective_decode_size_max")),
                fmt(on.get("avg_selective_decode_size")),
                fmt(on.get("paged_attention_fallbacks")),
            ]
            lines.append("| " + " | ".join(str(v) for v in vals) + " |")
        lines.append("")

        lines.append("### Interpretation")
        best_speedup = None
        best_pair = None
        for key, pair in by_key.items():
            off = pair.get("selective_decode_off")
            on = pair.get("selective_decode_on")
            if not off or not on:
                continue
            off_tps = to_float(off.get("aggregate_output_tok_s"))
            on_tps = to_float(on.get("aggregate_output_tok_s"))
            if off_tps and off_tps > 0 and on_tps is not None:
                speedup = (on_tps / off_tps - 1.0) * 100.0
                if best_speedup is None or speedup > best_speedup:
                    best_speedup = speedup
                    best_pair = (key, speedup)
        lines.append("- concurrency=1 usually has little or no selective batching benefit.")
        lines.append("- concurrency>=2 with avg_selective_decode_size>1 is the effective selective batching case.")
        if best_pair:
            (_rep, conc), speedup = best_pair
            if speedup >= 0:
                lines.append(f"- Best observed selective decode speedup: {fmt(speedup)}% at concurrency={conc}.")
            else:
                lines.append(
                    "- Selective decode was slower in this run. Likely causes include cache pressure, "
                    "too few generated tokens, too many fallbacks, per-sequence attention, thread contention, "
                    "or limited LM Head batch benefit."
                )
        lines.append(
            "- Selective Batch Decode is not batch attention. It batches token-independent operators "
            "such as QKV/O/FFN/LM Head, while attention remains per-sequence page-aware to avoid "
            "padding and mask waste from different KV lengths."
        )
        return lines

    def recommended_commands(self, markdown=False):
        exe = "python3 ../tools/bench_serving_engine.py"
        commands = [
            (
                "A. Functional validation",
                f"{exe} --binary ./qwen_model_loader --suite selective-decode "
                "--threads 4 --taskset 4-7 --concurrency 2,4 --repeats 1 "
                "--prompt-repeat 1 --max-new-tokens 32 --timeout 600 --request-timeout 600 "
                "--server-request-timeout 600 --selective-decode-max-batch 4 "
                "--selective-decode-min-batch 2 --validate-effective --verbose"
            ),
            (
                "B. Decode-heavy main experiment",
                f"{exe} --binary ./qwen_model_loader --suite selective-decode "
                "--threads 4 --taskset 4-7 --concurrency 1,2,4,8 --repeats 3 "
                "--prompt-repeat 1 --max-new-tokens 128 --timeout 1200 --request-timeout 1200 "
                "--server-request-timeout 1200 --selective-decode-max-batch 8 "
                "--selective-decode-min-batch 2 --validate-effective"
            ),
            (
                "C. Max batch sweep",
                "for b in 2 4 8; do "
                f"{exe} --binary ./qwen_model_loader --suite selective-decode "
                "--threads 4 --taskset 4-7 --concurrency 8 --repeats 3 "
                "--prompt-repeat 1 --max-new-tokens 128 --timeout 1200 --request-timeout 1200 "
                "--server-request-timeout 1200 --selective-decode-max-batch $b "
                "--selective-decode-min-batch 2 --validate-effective; done"
            ),
            (
                "D. Thread sweep",
                "for t in 1 2 3 4; do "
                f"{exe} --binary ./qwen_model_loader --suite selective-decode "
                "--threads $t --taskset 4-7 --concurrency 4 --repeats 3 "
                "--prompt-repeat 1 --max-new-tokens 128 --timeout 1200 --request-timeout 1200 "
                "--server-request-timeout 1200 --selective-decode-max-batch 4 "
                "--selective-decode-min-batch 2 --validate-effective; done"
            ),
        ]
        if not markdown:
            lines = []
            for title, cmd in commands:
                lines.append(f"{title}:")
                lines.append(cmd)
            return lines
        lines = []
        for title, cmd in commands:
            lines.append(f"### {title}")
            lines.append("```bash")
            lines.append(cmd)
            lines.append("```")
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
            "selective-decode", "paged-attention", "full", "all",
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
    p.add_argument("--selective-decode-max-batch", type=int, default=8)
    p.add_argument("--selective-decode-min-batch", type=int, default=2)
    p.add_argument("--decode-heavy-prompt", default="numbered-list")
    p.add_argument("--force-long-output", action="store_true", default=True)
    p.add_argument("--no-force-long-output", dest="force_long_output", action="store_false")
    p.add_argument("--validate-effective", action="store_true", default=True)
    p.add_argument("--no-validate-effective", dest="validate_effective", action="store_false")
    p.add_argument("--engine-debug", action="store_true", default=False)
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
    elif args.suite == "selective-decode":
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
