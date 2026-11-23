# Cloud TPU v7x (Ironwood) Day-1 Evaluation Runbook

**Live Cloud TPU provisioning target:**
- `tpu7x-standard-4t`: `4` physical chip packages (`8` logical chiplets × `96 GiB` HBM3e = `768 GiB`, intra-package D2D pairs `(0,1), (2,3), (4,5), (6,7)`). Google Cloud documents `tpu7x-standard-4t` as the VM machine type across Ironwood topologies.

**Offline Host Mock profiles (local dev/testing only, not live GCP VM shapes):**
- `tpu7x-1t` (`CLOUDTPU_MOCK_PROFILE=tpu7x-1t`, `OPENTPU_MOCK_TOPOLOGY=tpu7x-1t`): `2`-chiplet single-package mock (`192 GiB`, D2D pair `(0, 1)`)
- `tpu7x-4t` (`CLOUDTPU_MOCK_PROFILE=tpu7x-4t`, `OPENTPU_MOCK_TOPOLOGY=tpu7x-4t`): `8`-chiplet quad-package mock (`768 GiB`)

**Goal:** Zero wasted DWS Calendar Mode clock minutes during the 24-hour evaluation window by running three reproducible, non-destructive/benchmark scripts under [`tools/ironwood-day1/`](../../tools/ironwood-day1/).

---

## Pre-Flight (Before the VM Window Starts)

1. Warm GCS XLA compilation cache bucket & verify GGUF staging directory (`/opt/models` or GCS mirror) containing quantized `Q4_K_M` checkpoints for:
   - `8B` (sanity & baseline latency)
   - `32B` (`Qwen3-32B-Q4_K_M`)
   - `70B` (`Llama-3.3-70B-Instruct-Q4_K_M`)
   - `120B`
   - `235B` (`Qwen3-235B-Q4_K_M`)
2. Validate the Day-1 scripts offline in Host Mock before provisioning any VM:

```bash
./tools/ironwood-day1/00_pci_vfio_survey.sh --mock
./tools/ironwood-day1/01_ggml_cloudtpu_ops_bench.sh --mock
./tools/ironwood-day1/02_libopentpu_vfio_probe.sh --mock
```

---

## Minute 0–15: PCIe / Resizable BAR / VFIO Survey

```bash
./tools/ironwood-day1/00_pci_vfio_survey.sh --out-dir runs/ironwood-day1
```

Record from `runs/ironwood-day1/00_pci_vfio_survey.txt`:
- PCI Vendor/Device ID (`0x1ae0:0x0075` or observed Silicon DID) and physical PCI function count (`4` on `tpu7x-standard-4t`).
- `BAR0`, `BAR2`, `BAR4` window sizes and PCIe **Resizable BAR** (`0x0015`) capability fields (`current size:` and `supported:` lines).
- `MSI-X` vector count, `/dev/iommu` (`iommufd`) availability, and `/dev/vfio/*` group viability.
- Confirm physical-to-logical mapping in PJRT:
  - `tpu7x-standard-4t` -> `8` logical `PJRT_Device*` in serpentine 2×2 Hamiltonian package walk with consecutive D2D pairs `(0,1), (2,3), (4,5), (6,7)` (each package pair sharing `coords` with `core_on_chip = 0, 1`).

---

## Minute 15–120: `ggml-cloudtpu` Verification, `f8E4M3FN`, & Model Matrix

Run the turnkey ops & benchmark script with persistent GCS `XLA_COMPILATION_CACHE_DIR`:

```bash
./tools/ironwood-day1/01_ggml_cloudtpu_ops_bench.sh \
  --models-dir /opt/models \
  --spmd auto \
  --xla-cache gs://YOUR_BUCKET/freetpu-xla-cache/tpu7x \
  --out-dir runs/ironwood-day1
```

What this captures:
- `test_cloudtpu_host` standalone checks + `MAPS_LIBPYTHON=0`.
- Live `test-backend-ops -b CLOUDTPU` under default `BF16` MXU path (`CLOUDTPU_MATMUL_FP8=0`) and Ironwood `f8E4M3FN` MXU path (`CLOUDTPU_MATMUL_FP8=auto`).
- Side-by-side `llama-bench` throughput (`pp512` and `tg64`) across `8B`, `32B`, `70B`, `120B`, and `235B` (`Q4_K_M`) in both `BF16` and `f8E4M3FN` modes.

### Runtime Knobs on Ironwood (`tpu7x`)

| Env Var | Behavior on `tpu7x` |
| --- | --- |
| `CLOUDTPU_SPMD_CHIPS=auto` | Shards across `8` chiplets on live `tpu7x-standard-4t` (or `2` chiplets under the offline `tpu7x-1t` Host Mock profile) in Hamiltonian D2D-paired order (`chip = id / 2`, `chiplet = id % 2`) |
| `CLOUDTPU_MATMUL_FP8` | `0` = BF16 fallback; `auto` = `f8E4M3FN` dynamically scaled MXU `dot_general` on `tpu7x`; `1` = force `f8E4M3FN` |
| `CLOUDTPU_EMIT_SDY_MESH=1` | Emits D2D-aware Shardy `sdy.mesh` module declaration (`["ici"=4, "d2d"=2]`) |
| `XLA_COMPILATION_CACHE_DIR` | Persistent XLA executable cache (`gs://...` or local disk) to eliminate repeated compile overhead across restarts |

---

## Minute 120–150: Clean-Room `libopentpu` HAL & VFIO Probe

```bash
./tools/ironwood-day1/02_libopentpu_vfio_probe.sh \
  --live-vfio \
  --out-dir runs/ironwood-day1
```

Wait for:
- `ot_chip_geometry_for_device_id()` & Resizable BAR clamp verification passing (`ALL_HOST_TESTS_PASSED`).
- Live `/dev/vfio/*` probe confirming zero `libtpu.so` / `libpython` links (`MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`) and safe refusal on unvalidated non-`0x0063` ring DMA paths until `0x0075` (or observed DID) doorbell registers are documented cleanly.

---

## Clean-Room Proof Artifacts

All captured logs stay in `runs/ironwood-day1/` (untracked by git; sanitize before committing any `docs/` update) and must contain zero vendor binaries, zero local paths, and zero private credentials.
