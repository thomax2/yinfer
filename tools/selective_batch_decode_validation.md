# True Selective Batch Decode Validation

All performance numbers must be collected on the RK3588. The local Windows
build can only perform non-ARM syntax checks.

## Build

```bash
cd /home/orangepi/test_di/yinfer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target \
  qwen_model_loader bench_gptq_batch_kernel test_gptq_batch_decode
```

## Kernel Correctness

```bash
cd /home/orangepi/test_di/yinfer
ctest --test-dir build -R gptq_batch_decode_test --output-on-failure

taskset -c 4-7 ./build/bench_gptq_batch_kernel \
  --weights-dir ./weights \
  --rows 2,3,4,5,6,7,8 \
  --threads 4 --warmup 1 --repeats 1 \
  > /tmp/gptq_batch_correctness.csv

awk -F, 'NR==1 || $10 > 0.02 || $11 != 1 || $15 != 0 || $16 != 0' \
  /tmp/gptq_batch_correctness.csv
```

Expected: the `awk` command prints only the header. Linear `max_abs` must be
at most 0.02, LM Head `argmax_match` must be 1, and both fallback and full
logits counters must be zero.

## Kernel Performance

```bash
cd /home/orangepi/test_di/yinfer
env -u LLM_GPTQ_BATCH_COMPARE \
  taskset -c 4-7 ./build/bench_gptq_batch_kernel \
  --weights-dir ./weights \
  --rows 2,4,8 \
  --threads 4 --warmup 2 --repeats 5 \
  | tee /tmp/gptq_batch_performance.csv
```

The `speedup` column compares true batch time with the sum of B optimized
single-row calls. Do not accept the serving optimization if the main matrix
shapes and LM Head fail to show a repeatable kernel-level gain.

## Single-Request Regression

```bash
cd /home/orangepi/test_di/yinfer/build
taskset -c 4-7 env \
  LLM_NUM_THREADS=4 \
  LLM_MAX_NEW_TOKENS=64 \
  ./qwen_model_loader
```

This path must not report any GPTQ batch-kernel calls.

## Selective Decode Off Regression

```bash
cd /home/orangepi/test_di/yinfer/build
taskset -c 4-7 env \
  LLM_PAGED_KV=1 \
  LLM_ENABLE_SESSION_CACHE=1 \
  LLM_ENABLE_PAGED_ATTENTION=1 \
  LLM_ENABLE_SERVICE=1 \
  LLM_ENABLE_SCHEDULER=1 \
  LLM_ENABLE_HTTP_SERVER=1 \
  LLM_ENABLE_CONTINUOUS_BATCHING=1 \
  LLM_ENABLE_SELECTIVE_BATCH_DECODE=0 \
  LLM_NUM_THREADS=4 \
  LLM_HTTP_HOST=127.0.0.1 \
  LLM_HTTP_PORT=8080 \
  ./qwen_model_loader
```

## Two Concurrent Requests

Terminal 1:

```bash
cd /home/orangepi/test_di/yinfer/build
taskset -c 4-7 env \
  LLM_PAGED_KV=1 \
  LLM_ENABLE_SESSION_CACHE=1 \
  LLM_ENABLE_PAGED_ATTENTION=1 \
  LLM_ENABLE_SERVICE=1 \
  LLM_ENABLE_SCHEDULER=1 \
  LLM_ENABLE_HTTP_SERVER=1 \
  LLM_ENABLE_CONTINUOUS_BATCHING=1 \
  LLM_ENABLE_SELECTIVE_BATCH_DECODE=1 \
  LLM_SELECTIVE_DECODE_MIN_BATCH=2 \
  LLM_SELECTIVE_DECODE_MAX_BATCH=8 \
  LLM_NUM_THREADS=4 \
  LLM_HTTP_HOST=127.0.0.1 \
  LLM_HTTP_PORT=8081 \
  LLM_SERVER_REQUEST_TIMEOUT_MS=600000 \
  LLM_DEBUG_METRICS=1 \
  LLM_METRICS_JSONL=/tmp/selective_true_batch.jsonl \
  ./qwen_model_loader
```

Terminal 2:

```bash
for sid in 1001 1002; do
  curl -N -sS http://127.0.0.1:8081/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen2.5-1.5b-rk3588\",\"session_id\":$sid,\"messages\":[{\"role\":\"user\",\"content\":\"请列出十条端侧推理优化建议\"}],\"stream\":true,\"temperature\":0,\"max_tokens\":64}" \
    > /tmp/selective_${sid}.sse &
done
wait

curl -sS http://127.0.0.1:8081/metrics | tee /tmp/selective_metrics.json
python3 -m json.tool /tmp/selective_metrics.json
```

Required metrics: `gptq_batch_kernel_calls > 0`,
`gptq_batch_row_gemv_fallbacks == 0`,
`gptq_batch_full_logits_elements_written == 0`,
`selective_decode_hotpath_allocations == 0`, and
`paged_attention_fallbacks == 0`.

## Serving Benchmark

```bash
cd /home/orangepi/test_di/yinfer/build
python3 ../tools/bench_serving_engine.py \
  --binary ./qwen_model_loader \
  --suite selective-decode \
  --threads 4 \
  --taskset 4-7 \
  --concurrency 1,2,4,8 \
  --repeats 3 \
  --max-new-tokens 256 \
  --timeout 1800 \
  --request-timeout 1800 \
  --server-request-timeout 1800 \
  --selective-decode-min-batch 2 \
  --selective-decode-max-batch 8 \
  --validate-effective
```

## Register-Spill Inspection

```bash
cd /home/orangepi/test_di/yinfer
objdump -d -C \
  build/CMakeFiles/llm_engine.dir/src/backends/cpu/arm_neon/linear_decode_neon.cpp.o \
  > /tmp/linear_decode_neon.asm

grep -n "gptq_batch_compute_subpanel" /tmp/linear_decode_neon.asm
```

Inspect the B=4/NR=16 and B=8/NR=8 template instances. The K inner loop must
not repeatedly spill accumulator vectors to stack memory.
