# Cloud TPU `v6e-8` High-Batch (`c=64–128`) & Dynamic W8A8 INT8 Serving Playbook

Ready-to-run `vllm/vllm-tpu:latest` launch presets and FreeTPU Router configuration for scaling `Qwen3-32B`, `Llama-3.3-70B`, and `gpt-oss-120b` on Cloud TPU `v6e-8` (Trillium, `256 GiB` HBM, `12.8–13.1 TB/s` HBM BW, `7,344 TFLOP/s` BF16 / `14,688 TOPS` INT8).

---

## 1. Why High-Batch (`c=64–128`) & INT8 Matter on `v6e-8`

Live Phase 1 `v6e-8` serving measurements showed:
- **Extreme decode bandwidth bound:** The Trillium BF16 roofline ridge point is `7,344 / 12.8 ≈ 574` tokens/step (`560` at `13.1 TB/s`), so `c=1..16` operates at **0.1%–1.2% MFU**.
- **`c=8 -> c=16` proxy cliff (now fixed):** The initial `ThreadingHTTPServer` router hit a model-independent `+7.2 ms/step` stall from `listen(5)` SYN backlog saturation, Nagle/Delayed-ACK without `TCP_NODELAY`, per-chunk thread/GIL/JSON contention, and upstream HTTP/1.1 keep-alive waits (`active_inflight` ghost counts and `5.6–6.0 s` EWMA `latency_ms`).
- **Router fix in this branch:** `_SSEReactor` (`selectors.DefaultSelector` event-loop pump), `request_queue_size = 1024`, `TCP_NODELAY` on client/upstream sockets, `"Connection": "close"` upstream headers, single-pass SSE parsing, and `_BackendSlotLease` RAII guarantees keep router overhead `<0.5 ms/chunk` at `c=128` with `active_inflight == 0` after every request.
- **Next hardware unlock (`c=64..128` + dynamic W8A8 INT8):** Scaling batch concurrency from `c=16` to `c=64..128` with async multi-step scheduling and 256-aligned static buckets projects **4×–7×** aggregate `tok/s`, and switching BF16 weights to native TPU dynamic `W8A8` INT8 halves per-step HBM weight streaming (`65.5 -> ~33 GB` on 32B, `141 -> ~71 GB` on 70B).

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

Prevent 15+ minute Spot cold-start recompilation across VM preemptions by persisting XLA artifacts to preemption-safe storage (Compute Engine Local SSD is ephemeral and discarded on Spot preemption — use a **retained Persistent Disk** reattached across replacements or a **GCS bucket**):

```bash
# Option A: Retained Persistent Disk mounted at /var/cache/freetpu-xla-cache
# (attach with --auto-delete=no and remount on replacement Spot VMs):
export JAX_COMPILATION_CACHE_DIR=/var/cache/freetpu-xla-cache
export VLLM_XLA_CACHE_PATH=/var/cache/freetpu-xla-cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Option B: Shared GCS bucket cache for ephemeral Spot fleets (recommended):
# export JAX_COMPILATION_CACHE_DIR="gs://${XLA_CACHE_BUCKET}/freetpu-xla-cache"
# export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"
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
  -e JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/var/cache/freetpu-xla-cache}"
  -e VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/var/cache/freetpu-xla-cache}"
  -e LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}"
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

### Preset C: Native TPU Dynamic `W8A8` INT8 Serving Presets

Trillium `v6e` delivers `1,836 TOPS/chip` (`14,688 TOPS` across 8 chips) on native `int8` MXUs while halving weight HBM reads vs BF16 (`33 GB` on 32B, `71 GB` on 70B). The TPU `compressed-tensors` INT8 linear loader requires **dynamic 8-bit activations (`W8A8`)** — weight-only `W8A16` checkpoints are not supported by that loader.

1. **Dense Dynamic `W8A8` INT8 (`compressed-tensors` checkpoint):**

```bash
# Set MODEL_W8A8 to a Hugging Face compressed-tensors dynamic W8A8 INT8 repo ID
# (for a local host directory, add -v /path/to/checkpoint:/models/dense-w8a8:ro
# and pass --model /models/dense-w8a8).
docker run -d --name vllm-int8-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model "${MODEL_W8A8}" \
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

2. **Preconverted `openai/gpt-oss-120b` Dynamic `W8A8` INT8 (`compressed-tensors`) Preset (`v6e-8`):**

`v6e` lacks native `f4E2M1FN` (`MXFP4`) MXU/Mosaic instructions, and the published `openai/gpt-oss-120b` checkpoint declares `quantization_config.quant_method = "mxfp4"` in `config.json`. Because `--quantization compressed-tensors` requires a checkpoint whose `config.json` already declares `"quant_method": "compressed-tensors"` with dynamic `W8A8` INT8 quantization, vLLM rejects raw `openai/gpt-oss-120b` under that flag. Perform an offline conversion first (dequantizing MXFP4 weights to BF16 and exporting a `compressed-tensors` dynamic `W8A8` INT8 checkpoint to a host directory, e.g. with `llm-compressor`), bind-mount that directory read-only into the container, and pass the container path `/models/gpt-oss-120b-int8` to `--model` (`~117 GB` total = `~14.6 GB/chip`, leaving `~108 GiB` free KV HBM, `~5.1 GB` active weights per token):

```bash
# Offline prerequisite (run once before serving; raw openai/gpt-oss-120b is mxfp4):
# Dequantize openai/gpt-oss-120b -> BF16 and export a compressed-tensors dynamic W8A8 INT8
# checkpoint to "${GPT_OSS_120B_INT8_HOST_DIR}" (with
# config.json quantization_config.quant_method == "compressed-tensors").
export GPT_OSS_120B_INT8_HOST_DIR=/srv/models/gpt-oss-120b-w8a8-int8

docker run -d --name vllm-gptoss120b-int8 "${CONTAINER_FLAGS[@]}" \
  -v "${GPT_OSS_120B_INT8_HOST_DIR}:/models/gpt-oss-120b-int8:ro" \
  -e MOE_TC_PERMUTE=1 \
  vllm/vllm-tpu:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8000 \
    --model /models/gpt-oss-120b-int8 \
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
