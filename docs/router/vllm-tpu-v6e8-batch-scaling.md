# Cloud TPU `v6e-8` High-Batch (`c=64–128`) BF16 Serving Playbook

Ready-to-run `vllm/vllm-tpu:v0.33.0` BF16 launch presets and FreeTPU Router configuration for scaling dense `Qwen3-32B` and `Llama-3.3-70B` models on Cloud TPU `v6e-8` (Trillium, `256 GiB` HBM, `12.8–13.1 TB/s` HBM BW, `7,344 TFLOP/s` BF16).

---

## 1. Why High-Batch (`c=64–128`) Matters on `v6e-8`

Live Phase 1 `v6e-8` serving measurements showed:
- **Extreme decode bandwidth bound:** The Trillium BF16 roofline ridge point is `7,344 / 12.8 ≈ 574` tokens/step (`560` at `13.1 TB/s`), so `c=1..16` operates at **0.1%–1.2% MFU**.
- **`c=8 -> c=16` proxy cliff (now fixed):** The initial `ThreadingHTTPServer` router hit a model-independent `+7.2 ms/step` stall from `listen(5)` SYN backlog saturation, Nagle/Delayed-ACK without `TCP_NODELAY`, per-chunk thread/GIL/JSON contention, and upstream HTTP/1.1 keep-alive waits (`active_inflight` ghost counts and `5.6–6.0 s` EWMA `latency_ms`).
- **Router fix in this branch:** `_SSEReactor` (`selectors.DefaultSelector` event-loop pump), `request_queue_size = 1024`, `TCP_NODELAY` on client/upstream sockets, `"Connection": "close"` upstream headers, single-pass SSE parsing, and `_BackendSlotLease` RAII guarantees keep router overhead `<0.5 ms/chunk` at `c=128` with `active_inflight == 0` after every request.
- **Next hardware unlock (`c=64..128` BF16):** Scaling batch concurrency from `c=16` to `c=64..128` with async multi-step scheduling and power-of-two `num_tokens` compilation buckets up to `2048` projects **4×–7×** aggregate `tok/s` while keeping model weights in native BF16 (`65.5 GB` on 32B, `141 GB` on 70B).

---

## 2. TPU Shape Compilation & Persistent XLA Cache

With the preset flags (`--max-num-seqs 128`, `--max-num-batched-tokens 2048`, `--enable-chunked-prefill`) and default bucket settings, pinned `vllm/vllm-tpu:v0.33.0` (`tpu-inference`) compiles:
- **Power-of-two token-count (`num_tokens`) padding buckets** starting at `16` (`16, 32, 64, 128, 256, 512, 1024, 2048`) up to `--max-num-batched-tokens 2048` (so decode steps at `c=64` and `c=128` use distinct shapes below `256`).
- **A single attention request bucket** sized to `--max-num-seqs 128` by default (unless attention request bucketing is explicitly enabled in the container configuration).

### Persistent Model Weights & XLA Compilation Cache

The default Cloud TPU VM boot disk (`100 GiB`) cannot hold `Llama-3.3-70B-Instruct` BF16 weights (`~141 GB`) plus XLA compilation artifacts, and Compute Engine Local SSD is ephemeral and discarded on Spot preemption. Cloud TPU `v6e` supports attached **Hyperdisk** rather than Persistent Disk — attach a retained **Hyperdisk Balanced** (`hyperdisk-balanced`, `>= 250 GiB` for 70B BF16 weights + caches, `auto-delete=no`, read-write mode) mounted at `/mnt/disks/tpu-cache` for Hugging Face weights (and optionally XLA cache), or use a **GCS bucket** for shared XLA caching across Spot replacements:

```bash
# Option A: Retained Hyperdisk Balanced (`--type=hyperdisk-balanced`, >=250 GiB)
# attached in read-write mode (with `auto-delete=no` so weights and compiled XLA
# artifacts both survive Spot preemption and can be reattached/remounted on
# replacement v6e-8 VMs):
export HF_HUB_CACHE=/mnt/disks/tpu-cache/hf
export XLA_CACHE_DIR=/mnt/disks/tpu-cache/xla
export JAX_COMPILATION_CACHE_DIR=/xla-cache
export VLLM_XLA_CACHE_PATH=/xla-cache
mkdir -p "$HF_HUB_CACHE" "$XLA_CACHE_DIR"

# Option B: Shared GCS bucket for XLA cache across ephemeral Spot fleets (recommended;
# still keep HF_HUB_CACHE on a >=250 GiB Hyperdisk Balanced volume for 70B BF16 weights):
# export JAX_COMPILATION_CACHE_DIR="gs://${XLA_CACHE_BUCKET}/freetpu-xla-cache"
# export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"
# export LIBTPU_INIT_ARGS="--xla_tpu_enable_persistent_cache=true"
```

> **Shared cache security note:** JAX trusts compiled artifacts loaded from `JAX_COMPILATION_CACHE_DIR` / `VLLM_XLA_CACHE_PATH` (a maliciously crafted cache entry can achieve arbitrary code execution under the serving worker identity). Restrict write IAM permissions on any shared GCS bucket or prefix strictly to trusted serving service accounts, and isolate workloads in different trust domains into separate cache buckets or prefixes.

---

## 3. Non-Privileged CDI Container Base Flags

Before launching either preset with `--device=freetpu.dev/tpu=all`, enable CDI in Docker (for Docker versions where CDI is not enabled by default, add `"features": {"cdi": true}` to `/etc/docker/daemon.json` and restart Docker) and register the boot-scoped FreeTPU CDI specification on the TPU VM (see [ops/tpu-cdi/README.md](../../ops/tpu-cdi/README.md) for continuous `sync` reconciliation and VFIO rebind safety):

```bash
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /var/run/cdi/freetpu.json --validate
```

Per the verified Cloud TPU `v6e-8` CDI contract (no `--privileged` required; `--net=host` is required for 8-chip `libtpu` coordinator initialization). Both BF16 presets below share `CONTAINER_FLAGS`, mounting both `HF_HUB_CACHE` and the XLA compilation cache off the retained Hyperdisk so neither weights nor compiled artifacts live on the 100 GiB boot disk:

```bash
CONTAINER_FLAGS=(
  --rm
  --device=freetpu.dev/tpu=all
  --cap-add=IPC_LOCK
  --ulimit memlock=-1:-1
  --net=host
  --shm-size=64G
  -v "${HF_HUB_CACHE:-/mnt/disks/tpu-cache/hf}:/hf-cache"
  -e HF_HUB_CACHE=/hf-cache
  -v "${XLA_CACHE_DIR:-/mnt/disks/tpu-cache/xla}:/xla-cache"
  -e JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/xla-cache}"
  -e VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/xla-cache}"
  -e LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}"
)
```

---

## 4. `vllm/vllm-tpu:v0.33.0` Launch Presets (`v6e-8`, `TP=8`)

> **Flag check note:** These presets pin `vllm/vllm-tpu:v0.33.0` and pass `--additional-config '{"enable_continue_decode": true}'` alongside `--async-scheduling` to opt into TPU multi-step decoding. When upgrading to a newer image tag, verify scheduler and `--additional-config` flags against `python3 -m vllm.entrypoints.openai.api_server --help` inside the container.

### Preset A: `Qwen/Qwen3-32B` BF16 High-Batch (`c=64..128`)

Targets `c=64..128` decode scaling (`~4,500–6,500 tok/s` projected vs `953 tok/s` baseline; `~$0.28–$0.40/1M` Spot):

```bash
docker run -d --name vllm-qwen3-32b-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:v0.33.0 \
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
    --async-scheduling \
    --additional-config '{"enable_continue_decode": true}'
```

### Preset B: `unsloth/Llama-3.3-70B-Instruct` BF16 High-Batch (`c=64..128`)

Targets `c=64..128` decode scaling (`~2,500–3,800 tok/s` projected vs `633 tok/s` baseline; `~$0.47–$0.72/1M` Spot):

```bash
docker run -d --name vllm-llama70b-highbatch "${CONTAINER_FLAGS[@]}" \
  vllm/vllm-tpu:v0.33.0 \
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
    --async-scheduling \
    --additional-config '{"enable_continue_decode": true}'
```

---

## 5. TPU Exporter, FreeTPU Router Launch & Post-Sweep Verification

`router/examples/cloud-tpu-v6e8-high-batch.json` enables `cloud-tpu-v6e-8-spot` with `tpu_exporter_url: "http://127.0.0.1:9400/metrics"`, so start the FreeTPU TPU exporter and verify it reports healthy chips **before** launching the router:

```bash
python3 ops/tpu-exporter/tpu_exporter.py \
  --host 127.0.0.1 \
  --port 9400 &

# Verify the exporter is up and all 8 chips report healthy (1) before starting the router:
curl -sf http://127.0.0.1:9400/metrics | grep -E 'freetpu_(exporter_up|tpu_devices_discovered|tpu_health_status)'
```

Then start the router with the high-batch config preset:

```bash
python3 router/freetpu_router.py \
  --config router/examples/cloud-tpu-v6e8-high-batch.json
```

Run a short warmup sweep (`c=1..128`) to populate the XLA compilation cache before timing, and verify zero leaked in-flight slots immediately after every concurrency sweep:

```bash
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
# Assert: "active_inflight": 0 and per-backend "inflight": 0

curl -s http://127.0.0.1:8080/metrics | grep freetpu_router_active_inflight
# Assert: freetpu_router_active_inflight{backend_id="cloud-tpu-v6e-8-spot"} 0
```
