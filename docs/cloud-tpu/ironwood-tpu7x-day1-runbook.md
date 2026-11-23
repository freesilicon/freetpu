# Cloud TPU v7x (Ironwood) Day-1 Evaluation Runbook

**Live Cloud TPU provisioning target:**
- `tpu7x-standard-4t`: `4` physical chip packages (`8` logical chiplets × `96 GiB` HBM3e = `768 GiB`, intra-package D2D pairs `(0,1), (2,3), (4,5), (6,7)`). Google Cloud documents `tpu7x-standard-4t` as the VM machine type across Ironwood topologies.

**Offline Host Mock profiles (local dev/testing only, not live GCP VM shapes):**
- `tpu7x-1t` (`CLOUDTPU_MOCK_PROFILE=tpu7x-1t`, `OPENTPU_MOCK_TOPOLOGY=tpu7x-1t`): `2`-chiplet single-package mock (`192 GiB`, D2D pair `(0, 1)`)
- `tpu7x-4t` (`CLOUDTPU_MOCK_PROFILE=tpu7x-4t`, `OPENTPU_MOCK_TOPOLOGY=tpu7x-4t`): `8`-chiplet quad-package mock (`768 GiB`)

**Goal:** Zero wasted DWS Calendar Mode clock minutes during the 24-hour evaluation window by running four reproducible, non-destructive/benchmark scripts under [`tools/ironwood-day1/`](../../tools/ironwood-day1/) (`00` -> `01` -> `02` -> `03`).

---

## Pre-Flight (Before the VM Window Starts)

1. Stage quantized GGUF checkpoints in a same-region GCS bucket (`GCS_MODEL_BUCKET=gs://YOUR_BUCKET/models`, aligned to `TPU_ZONE`) or local `/opt/models` directory (symlinks are followed via `find -L`) and reserve a persistent XLA cache prefix (`XLA_COMPILATION_CACHE_DIR=gs://YOUR_BUCKET/xla-cache/tpu7x`):
   - **1t bracket tiers (`Q4_K_M`):**
     - `8B` (sanity & baseline latency)
     - `32B` (`Qwen3-32B-Q4_K_M`)
     - `70B` (`Llama-3.3-70B-Instruct-Q4_K_M`)
     - `120B` (`gpt-oss-120b` `Q4_K_M`)
     - `235B` (`Qwen3-235B-Q4_K_M`)
   - **Hero `tpu7x-standard-4t` tier (`768 GiB`, `8` logical chiplets):**
     - `671B` (`DeepSeek-V3` / `DeepSeek-R1` `FP8` or `Q4_K_M`/`Q4_0`)
2. Validate the entire 4-step Day-1 sequence offline in Host Mock before provisioning any VM:

```bash
./tools/ironwood-day1/00_pci_vfio_survey.sh --mock
./tools/ironwood-day1/01_ggml_cloudtpu_ops_bench.sh --mock
./tools/ironwood-day1/02_libopentpu_vfio_probe.sh --mock
./tools/ironwood-day1/03_ops_and_router_smoke.sh --mock
```

---

## Step 00 (Minute 0–15): PCIe / Resizable BAR / VFIO Survey & Optional `libtpu 0.0.53.1` Staging

On VMs without JAX installed (or running Python 3.10 where `pip install libtpu` pulls a legacy 2025 wheel), pass `--stage-libtpu` to extract `libtpu/libtpu.so` directly from the PyPI `libtpu 0.0.53.1` `cp312` `x86_64` wheel (verified by SHA-256) with zero runtime Python/JAX dependency:

```bash
./tools/ironwood-day1/00_pci_vfio_survey.sh \
  --stage-libtpu \
  --expected-packages 4 \
  --expected-devices 8 \
  --out-dir runs/ironwood-day1
```

Record from `runs/ironwood-day1/00_pci_vfio_survey.txt`:
- PCI Vendor/Device ID (`0x1ae0:0x0076` guest VFIO ID / `0x0075` or observed Silicon DID) and physical PCI function count (`4` on `tpu7x-standard-4t`).
- `BAR0`, `BAR2`, `BAR4` window sizes and PCIe **Resizable BAR** (`0x0015`) capability fields (`current size:` and `supported:` lines).
- `MSI-X` vector count, `/dev/iommu` (`iommufd`) availability, and `/dev/vfio/*` group viability.
- Confirm physical-to-logical mapping in PJRT (native C++ `GetPjrtApi` probe with automatic `jax.local_devices(backend="tpu")` fallback preserving any native failure reason in the log):
  - `tpu7x-standard-4t` -> `8` logical `PJRT_Device*` in serpentine 2×2 Hamiltonian package walk with consecutive D2D pairs `(0,1), (2,3), (4,5), (6,7)` (each package pair sharing `coords` with `core_on_chip = 0, 1`).
  - Confirm `STATUS: PJRT_TOPOLOGY_CAPTURED` (the script flags `WARN: TOPOLOGY_COUNT_MISMATCH` / `STATUS: PJRT_TOPOLOGY_MISMATCH` whenever live counts disagree with the expected `4` packages / `8` logical devices).

---

## Step 01 (Minute 15–120): `ggml-cloudtpu` Verification, `f8E4M3FN`, FreeTile Mosaic `Q4_K`/`Q6_K`, & Model Matrix

Run the turnkey ops & benchmark script with same-region GCS model staging (`--gcs-model-bucket`, skipped cleanly if unset) and persistent GCS `XLA_COMPILATION_CACHE_DIR`:

```bash
./tools/ironwood-day1/01_ggml_cloudtpu_ops_bench.sh \
  --gcs-model-bucket gs://YOUR_BUCKET/models \
  --tpu-zone YOUR_TPU_ZONE \
  --models-dir /opt/models \
  --spmd auto \
  --xla-cache gs://YOUR_BUCKET/xla-cache/tpu7x \
  --out-dir runs/ironwood-day1
```

What this captures:
- Optional same-region GCS sync into `--models-dir` (`find -L` follows symlinks for both `/opt/models` and staged `.gguf` shards) and persistent `XLA_COMPILATION_CACHE_DIR` (`gs://...` or local fallback).
- `test_cloudtpu_host` standalone checks + `MAPS_LIBPYTHON=0`.
- Live `test-backend-ops -b CLOUDTPU` under pinned `BF16` pure StableHLO baseline (`CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_FREETILE_MOSAIC=0`), plus the dedicated live `f8E4M3FN` accuracy gate (`test_cloudtpu_host --fp8-parity-only`, `NMSE < 1e-2` for `Q8_0` and factored `Q4_K_I4` + exact zero cancellation, avoiding upstream `test-backend-ops`'s hardcoded `5e-4` `Q8_0` BF16 limit).
- Dedicated **FreeTile Mosaic `Q4_K`/`Q6_K` pass** (`CLOUDTPU_FREETILE_MOSAIC=1`) comparing Mosaic `tpu_custom_call` kernels against the pure StableHLO baseline (both in `test-backend-ops` / `--mock` and per-tier `llama-bench`; pass `--no-mosaic` to skip).
- Side-by-side `llama-bench` throughput (`pp512` and `tg64`) across the `1t` bracket tiers (`8B`, `Qwen3-32B`, `Llama-3.3-70B`, `gpt-oss-120b`, `Qwen3-235B` `Q4_K_M`) and the hero **`tpu7x-standard-4t` (`768 GiB`) `DeepSeek-V3` / `DeepSeek-R1` 671B (`FP8`/`Q4`)** tier in pure StableHLO BF16, FreeTile Mosaic `Q4_K`/`Q6_K`, and `f8E4M3FN` modes.

### Runtime Knobs on Ironwood (`tpu7x`)

| Env Var | Behavior on `tpu7x` |
| --- | --- |
| `LIBTPU_PATH` | Explicit path to standalone `libtpu-0.0.53.1.so` staged from the `cp312` wheel (`--stage-libtpu`) |
| `CLOUDTPU_SPMD_CHIPS=auto` | Shards across `8` chiplets on live `tpu7x-standard-4t` (or `2` chiplets under the offline `tpu7x-1t` Host Mock profile) in Hamiltonian D2D-paired order (`chip = id / 2`, `chiplet = id % 2`) |
| `CLOUDTPU_MATMUL_FP8` | `0` = BF16 fallback; `auto` = `f8E4M3FN` dynamically scaled MXU `dot_general` on `tpu7x`; `1` = force `f8E4M3FN` |
| `CLOUDTPU_FREETILE_MOSAIC=1` | Enables FreeTile Mosaic `Q4_K`/`Q6_K` `tpu_custom_call` lowering with automatic StableHLO retry fallback |
| `CLOUDTPU_EMIT_SDY_MESH=1` | Emits D2D-aware Shardy `sdy.mesh` module declaration (`["ici"=4, "d2d"=2]`) |
| `GCS_MODEL_BUCKET` / `TPU_ZONE` | Optional same-region GCS model staging bucket and zone metadata (`01_ggml_cloudtpu_ops_bench.sh` skips GCS sync cleanly if unset) |
| `XLA_COMPILATION_CACHE_DIR` | Persistent XLA executable cache (`gs://...` or local disk fallback) to eliminate repeated compile overhead across restarts |

---

## Step 02 (Minute 120–150): Clean-Room `libopentpu` HAL & VFIO Probe

```bash
./tools/ironwood-day1/02_libopentpu_vfio_probe.sh \
  --live-vfio \
  --out-dir runs/ironwood-day1
```

Wait for:
- `ot_chip_geometry_for_device_id()` & Resizable BAR clamp verification passing (`ALL_HOST_TESTS_PASSED`).
- Live `/dev/vfio/*` probe confirming **`MAPS_LIBTPU=0` and `MAPS_LIBPYTHON=0`** (`CLEANROOM_MAPS_STATUS=PASS`) and either `LIVE_VFIO_PROBE_STATUS=PASS` on validated `0x0063` or explicit pre-reset refusal (`Refusing to reset/program unsupported VFIO PCI device` -> `LIVE_VFIO_PROBE_STATUS=EXPECTED_UNVALIDATED_DEVICE_REFUSAL`) on unvalidated PCI DIDs until `0x0076` / `0x0075` (or observed DID) doorbell registers are documented cleanly.

---

## Step 03 (Minute 150–180): Ops Control Plane (`CDI v1.1`, `tpu-exporter`) & `router/` Concurrency Smoke

With `llama-server` (or `vllm-tpu`) serving an OpenAI-compatible endpoint (default `http://127.0.0.1:8000/v1`), run:

```bash
./tools/ironwood-day1/03_ops_and_router_smoke.sh \
  --backend-url http://127.0.0.1:8000/v1 \
  --concurrencies 16,32,64,128 \
  --out-dir runs/ironwood-day1
```

What this verifies:
1. **CNCF CDI v1.1 (`ops/tpu-cdi/freetpu_cdi.py`)**: generates a `cdiVersion: "1.1.0"` JSON spec and validates it against live host VFIO/iommufd topology (`generate --validate` + `validate --check-host`).
2. **Prometheus Telemetry (`ops/tpu-exporter/tpu_exporter.py --once`)**: captures direct `libtpu` gRPC (`127.0.0.1:8431`), SeqLock SHM, and pure sysfs PCIe AER counters in a single scrape.
3. **FreeTPU Router (`router/freetpu_router.py`)**: validates generated config (`--check-config`) and runs a bounded streaming SSE concurrency sweep across `c=16, 32, 64, 128`, verifying `100%` SSE `[DONE]` completion and `active_inflight == 0` (`/health` and `/metrics` `freetpu_router_active_inflight`) after every tier.

---

## Clean-Room Proof Artifacts

All captured logs stay in `runs/ironwood-day1/` (untracked by git; sanitize before committing any `docs/` update) and must contain zero vendor binaries, zero local paths, and zero private credentials.
