# Playbook 02: Cloud TPU `v6e-1` / `v5e-1` Headless Colab 60-Second Benchmark

Run `nvbench`-style compute and `nvbandwidth`-style HBM2e streaming benchmarks headlessly on Google Colab Cloud TPU (`v6e-1` Trillium 32 GB HBM with automatic `v5e-1` 16 GB HBM fallback) and verify session teardown via `colab sessions` / `colab stop -s` to prevent idle Compute Unit burn.

---

## Prerequisites

- `python3` + `numpy` locally.
- [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) (`colab`) installed and authenticated (`--auth=adc` or `--auth=oauth2`) with Colab Pro+ / Compute Units.

---

## 60-Second Copy-Paste Commands

### 1. Pre-Flight Compute Unit Check + 1-Command Headless Run (`v6e-1` -> `v5e-1`)

```bash
# Isolate Colab session metadata in a private 0700 directory (never /tmp)
mkdir -p -m 700 ~/.cache/freetpu-colab

# Check available Compute Units before provisioning
colab --auth=adc usage

# Run compute + bandwidth remotely on Colab TPU, plus local comms, with isolated state
python3 bench/freetpu_bench.py \
  --suite all \
  --backend colab_tpu \
  --colab-auth adc \
  --colab-tpu auto \
  --colab-config ~/.cache/freetpu-colab/sessions.json \
  --json runs/colab-tpu.json \
  --markdown runs/colab-tpu.md

cat runs/colab-tpu.md
```

### 2. Force a Specific Chip (`v5e1` or `v6e1`) or Inspect Planned Commands

```bash
# Dry-run: inspect exact commands without allocating a TPU VM (prints plan directly to stdout; add --json - for JSON)
python3 bench/freetpu_bench.py --suite compute,bandwidth --backend colab_tpu --dry-run

# Target v5e-1 directly (skips v6e-1 capacity probe)
python3 bench/freetpu_bench.py \
  --suite compute,bandwidth \
  --backend colab_tpu \
  --colab-tpu v5e1 \
  --colab-config ~/.cache/freetpu-colab/v5e1.json \
  --markdown runs/colab-v5e1.md
```

### 3. Direct `colab run` Invocation (Zero Wrapper)

```bash
mkdir -p -m 700 ~/.cache/freetpu-colab
colab --auth=adc --config ~/.cache/freetpu-colab/direct.json run \
  -s freetpu-bench-direct --tpu V5E1 --keep --timeout 300 \
  bench/freetpu_bench.py --suite compute,bandwidth --backend jax_tpu --emit-markers
# Record the backend runtime endpoint, stop the session, and verify the endpoint disappears (including under `[?]`)
colab --auth=adc --config ~/.cache/freetpu-colab/direct.json sessions
colab --auth=adc --config ~/.cache/freetpu-colab/direct.json stop -s freetpu-bench-direct
colab --auth=adc --config ~/.cache/freetpu-colab/direct.json sessions
```

Passing `--keep` preserves the local session-to-endpoint mapping until `colab stop -s` succeeds, so a failed unassign cannot silently drop local state and leave an orphaned `[?] <endpoint>` runtime burning Compute Units. Always confirm in the final `colab sessions` output that the runtime endpoint is gone.

---

## What Good Looks Like

| Metric | Cloud TPU `v5e-1` (Verified Live Silicon) | Cloud TPU `v6e-1` (Published Spec) | Notes |
| --- | ---: | ---: | --- |
| **BF16 GEMM (`8192^3`)** | **`179.36 TFLOP/s`** peak (`91.0%` of `197 TF/s`) | `918 TFLOP/s` peak | MXU saturates at `>=4096^3` |
| **INT8 GEMM (`8192^3`)** | **`350.69 TOPS`** peak (`89.2%` of `394 TOPS`) | `1,836 TOPS` peak | `2x` BF16 throughput |
| **HBM2e Triad BW (`1 GiB`)** | **`655.55 GB/s`** (`80.0%` of `819 GB/s`) | `1,638 GB/s` peak | 2 reads + 1 write BF16 triad |
| **Small-Op Sync Floor (`1x64x32`)** | **`117.6–135.0 us`** (`32.0 us` pipelined async) | `~100 us` PJRT est. | Motivates fused/persistent kernels |
| **Public On-Demand Rate** | **`$1.20 / chip-hr`** | **`$2.70 / chip-hr`** | GCP `us-east5` / `us-central1` |

---

## Operational Guardrails

- **Always isolate `--colab-config` in a private `0700` dir**: Passing `--colab-config ~/.cache/freetpu-colab/...` keeps parallel benchmark runs separate from `~/.config/colab-cli/sessions.json` and avoids shared `/tmp` directories.
- **Automatic `v6e1 -> v5e1` Fallback & Endpoint-Verified Teardown**: If Colab refuses allocation before `[colab] Session READY`, `freetpu_bench.py` falls back to `v5e1` automatically (never retrying after `[colab] Session READY` to avoid duplicate CU spend). It invokes `colab run --keep`, records the runtime endpoint from `colab sessions`, runs `colab stop -s <session>`, and verifies that the backend endpoint disappeared (failing verification if the endpoint persists even as an orphaned `[?]` entry).
- **Scope Boundary**: Colab is strictly for single-chip R&D and reproducible benchmarking—never deploy public inference APIs or multi-tenant proxies on Colab runtimes.
