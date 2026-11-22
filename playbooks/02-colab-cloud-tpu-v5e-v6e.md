# Playbook 02: Cloud TPU `v6e-1` / `v5e-1` Headless Colab 60-Second Benchmark

Run `nvbench`-style compute and `nvbandwidth`-style HBM2e streaming benchmarks headlessly on Google Colab Cloud TPU (`v6e-1` Trillium 32 GB HBM with automatic `v5e-1` 16 GB HBM fallback) and auto-stop cleanly with zero idle Compute Unit burn.

---

## Prerequisites

- `python3` + `numpy` locally.
- [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) (`colab`) installed and authenticated (`--auth=adc` or `--auth=oauth2`) with Colab Pro+ / Compute Units.

---

## 60-Second Copy-Paste Commands

### 1. Pre-Flight Compute Unit Check + 1-Command Headless Run (`v6e-1` -> `v5e-1`)

```bash
# Check available Compute Units before provisioning
colab --auth=adc usage

# Run compute + bandwidth remotely on Colab TPU, plus local comms, with isolated state
python3 bench/freetpu_bench.py \
  --suite all \
  --backend colab_tpu \
  --colab-auth adc \
  --colab-tpu auto \
  --colab-config /tmp/freetpu-colab-sessions.json \
  --json runs/colab-tpu.json \
  --markdown runs/colab-tpu.md

cat runs/colab-tpu.md
```

### 2. Force a Specific Chip (`v5e1` or `v6e1`) or Inspect Planned Commands

```bash
# Dry-run: inspect exact commands without allocating a TPU VM
python3 bench/freetpu_bench.py --suite compute,bandwidth --backend colab_tpu --dry-run --json -

# Target v5e-1 directly (skips v6e-1 capacity probe)
python3 bench/freetpu_bench.py \
  --suite compute,bandwidth \
  --backend colab_tpu \
  --colab-tpu v5e1 \
  --colab-config /tmp/freetpu-colab-v5e1.json \
  --markdown runs/colab-v5e1.md
```

### 3. Direct `colab run` Invocation (Zero Wrapper)

```bash
colab --auth=adc --config /tmp/freetpu-colab-direct.json run \
  --tpu V5E1 --timeout 300 \
  bench/freetpu_bench.py --suite compute,bandwidth --backend jax_tpu --emit-markers
```

`colab run` provisions an ephemeral TPU VM, executes `bench/freetpu_bench.py`, streams stdout back, and immediately releases the VM on exit.

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

- **Always isolate `--colab-config`**: Passing `--colab-config /tmp/...` keeps parallel benchmark runs from overwriting your default `~/.config/colab-cli/sessions.json`.
- **Automatic `v6e1 -> v5e1` Fallback**: If Colab returns `HTTP 412 / 400 / 503` or `REASON_NO_BACKENDS` during VM allocation, `freetpu_bench.py` falls back to `v5e1` automatically (never retrying after `[colab] Session READY` to avoid duplicate CU spend).
- **Scope Boundary**: Colab is strictly for single-chip R&D and reproducible benchmarking—never deploy public inference APIs or multi-tenant proxies on Colab runtimes.
