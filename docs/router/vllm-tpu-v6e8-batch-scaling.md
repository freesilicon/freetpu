# Cloud TPU `v6e-8` High-Batch (`c=64–128`) & W8A8/W8A16 Serving Playbook

Ready-to-run `vllm/vllm-tpu:latest` launch presets and FreeTPU Router configuration for scaling `Qwen3-32B`, `Llama-3.3-70B`, and `gpt-oss-120b` on Cloud TPU `v6e-8` (Trillium, `256 GiB` HBM, `12.8–13.1 TB/s` HBM BW, `7,344 TFLOP/s` BF16 / `14,688 TOPS` INT8).

---

## 1. Why High-Batch (`c=64–128`) & INT8 Matter on `v6e-8`

Live Phase 1 `v6e-8` serving measurements showed:
- **Extreme decode bandwidth bound:** The Trillium BF16 roofline ridge point is `7,344 / 12.8 ≈ 574` tokens/step (`560` at `13.1 TB/s`), so `c=1..16` operates at **0.1%–1.2% MFU**.
- **`c=8 -> c=16` proxy cliff (now fixed):** The initial `ThreadingHTTPServer` router hit a model-independent `+7.2 ms/step` stall from `listen(5)` SYN backlog saturation, Nagle/Delayed-ACK without `TCP_NODELAY`, per-chunk thread/GIL/JSON contention, and upstream HTTP/1.1 keep-alive waits (`active_inflight` ghost counts and `5.6–6.0 s` EWMA `latency_ms`).
- **Router fix in this branch:** `_SSEReactor` (`selectors.DefaultSelector` event-loop pump), `request_queue_size = 1024`, `TCP_NODELAY` on client/upstream sockets, `"Connection": "close"` upstream headers, single-pass SSE parsing, and `_BackendSlotLease` RAII guarantees keep router overhead `<0.5 ms/chunk` at `c=128` with `active_inflight == 0` after every request.
- **Next hardware unlock (`c=64..128` + W8A8/W8A16):** Scaling batch concurrency from `c=16` to `c=64..128` with async multi-step scheduling and 256-aligned static buckets projects **4×–7×** aggregate `tok/s`, and switching BF16 weights to native TPU INT8 (`W8A8` / `W8A16`) halves per-step HBM weight streaming (`65.5 -> ~33 GB` on 32B, `141 -> ~71 GB` on 70B).

---

## 2. 2D MXU-Aligned Static Compilation Bucket Matrix

Trillium (`v6e`) uses `256×256` systolic MXUs; reduction/output dimensions align to `256` (and intermediate dimensions to `128`). Configure chunked prefill and static buckets to cover `c=64..128` without triggering mid-benchmark XLA recompilation:

| Execution Phase | `batch_size` Buckets | `num_tokens` Buckets | Notes |
| --- | --- | --- | --- |
| **Decode (Low Concurrency)** | `1, 2, 4, 8` | `1` | Interactive / single-stream TTFT & TPOT baseline |
| **Decode (High Concurrency)** | `16, 32, 64, 128` | `1` | High-throughput batch scaling (`4×–7×` aggregate tok/s) |
| **Chunked Prefill** | `1, 2, 4` | `256, 512, 1024, 2048` | Caps per-step prefill tokens at `2048` (`256`-aligned) |
| **Mixed (Extend + Decode)** | `8, 16, 32` | `128, 256` | Multi-turn KV extension overlapping active decodes |

### Persistent XLA Compilation Cache

Prevent 15+ minute Spot cold-start recompilation across VM preemptions by persisting XLA artifacts to local NVMe/host disk or GCS:

```bash
export XLA_COMPILATION_CACHE_DIR=/var/cache/freetpu-xla-cache
mkdir -p "$XLA_COMPILATION_CACHE_DIR"
# Optional GCS-backed persistent cache for Spot fleets:
# export LIBTPU_INIT_ARGS="--xla_tpu_enable_persistent_cache=true"
```

---

## 3. Non-Privileged CDI Container Base Flags

Per the verified Cloud TPU `v6e-8` CDI contract (no `--privileged` required; `--net=host` is required for 8-chip `libtpu` coordinator initialization):

```bash
CONTAINER_FLAGS=(
  --rm
  --device=freetpu.dev/tpu=all
  --cap-add=IPC_LOCK
  --ulimit memlock=-1:-1
  --net=host
  --shm-size=64G
  -v /var/cache/freetpu-xla-cache:/var/cache/freetpu-xla-cache
  -e XLA_COMPILATION_CACHE_DIR=/var/cache/freetpu-xla-cache
)
```

---

## 4. `vllm/vllm-tpu:latest` Launch Presets (`v6e-8`, `TP=8`)

> **Flag check note:** Before launching on a fresh `vllm/vllm-tpu:latest` image, verify scheduler flags (`--async-scheduling` / `--num-scheduler-steps`) against `python3 -m vllm.entrypoints.openai.api_server --help` inside the container.

### Preset A: `Qwen/Qwen3-32B` BF16 High-Batch (`c=64..128`)

Targets `c=64..128` decode scaling (`~4,500–6,500 tok/s` projected vs `953 tok/s` baseline; `~$0.28–$0.40/1M` Spot):

```bash
docker run -d --name vllm-qwen3-32b-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

### Preset B: `unsloth/Llama-3.3-70B-Instruct` BF16 High-Batch (`c=64..128`)

Targets `c=64..128` decode scaling (`~2,500–3,800 tok/s` projected vs `633 tok/s` baseline; `~$0.47–$0.72/1M` Spot):

```bash
docker run -d --name vllm-llama70b-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model unsloth/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

### Preset C: Native TPU INT8 (`W8A8` / `W8A16`) Serving Presets

Trillium `v6e` delivers `1,836 TOPS/chip` (`14,688 TOPS` across 8 chips) on native `int8` MXUs while halving weight HBM reads vs BF16 (`33 GB` on 32B, `71 GB` on 70B).

1. **Dense `W8A8` / `W8A16` INT8 (`compressed-tensors` / `qwix` checkpoint):**

```bash
# Replace MODEL_INT8 with a TPU-supported INT8 checkpoint
# (compressed-tensors W8A8/W8A16 or qwix INT8 export)
docker run -d --name vllm-int8-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model "${MODEL_INT8}" \
    --quantization compressed-tensors \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

2. **`openai/gpt-oss-120b` Load-Time MXFP4 -> W8A16/W8A8 INT8 Bypass (`v6e-8`):**

`v6e` lacks native `f4E2M1FN` (`MXFP4`) MXU/Mosaic instructions. Converting MXFP4 -> W8A16/W8A8 INT8 (`~117 GB` total = `~14.6 GB/chip`, leaving `~108 GiB` free KV HBM) with `MOE_TC_PERMUTE=1` and Expert Parallelism avoids the `gmm_v2` / `fused_ep_moe` `f4E2M1FN` compile error while keeping `~5.1 GB` active weights per token:

```bash
docker run -d --name vllm-gptoss120b-int8 "${CONTAINER_FLAGS[@]}" \
  -e MOE_TC_PERMUTE=1 \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model "${GPT_OSS_120B_INT8_OR_CONVERTED}" \
    --quantization compressed-tensors \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

---

## 5. FreeTPU Router Launch & Post-Sweep Verification

Start the router with the high-batch config preset:

```bash
python3 router/freetpu_router.py \
  --config router/examples/cloud-tpu-v6e8-high-batch.json
```

Warm up static buckets (`c=1, 8, 16, 32, 64, 128`) before timing, and verify zero leaked in-flight slots immediately after every concurrency sweep:

```bash
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
# Assert: "active_inflight": 0 and per-backend "inflight": 0

curl -s http://127.0.0.1:8080/metrics | grep freetpu_router_active_inflight
# Assert: freetpu_router_active_inflight{backend_id="cloud-tpu-v6e-8-spot"} 0
```
