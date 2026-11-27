# Cloud TPU `v6e-8` (Trillium) Live Proof Packet

- **Roadmap tracking:** [#3](https://github.com/saariuslystoned/freetpu/issues/3) ([posted summary comment](https://github.com/saariuslystoned/freetpu/issues/3#issuecomment-7080992844))
- **Phase 1 Status:** **Verified on real hardware** (Cloud TPU `v6e-8` Trillium, `2×4`, Ubuntu 22.04 x86_64 kernel `6.8.0-1015-gcp`, Docker `27.3.1`, `jax 0.6.2` host / `jax 0.11.2` in `vllm/vllm-tpu:latest` `v0.33.0rc1.dev430+g7b77ed581`)
- **Phase 2 Status:** `WAITING_FOR_QUOTA` — waiting on GCP quota approval for **Ironwood (`tpu7x-standard-4t`, 4 chips / `768 GiB` HBM3e)** (`TPUS-PER-TPU-FAMILY` `tpu_family=tpu7x` + `PREEMPTIBLE-TPU7X`). Once approved, Phase 2 will capture Ironwood's `1ae0:xxxx` PCI ID, run `70B` / `235B FP8` / `671B FP8` serving, and follow up with a targeted code patch PR for `ops/tpu-cdi` (`--net=host` + `tpu7x` PCI ID), `ops/tpu-exporter`, and `bench/freetpu_bench.py`.

---

## 1. Hardware & VFIO Topology
- **PCI accelerators:** `8 × Google, Inc. Device [1ae0:006f]` (`TPU v6 lite` / Trillium), bound to `vfio-pci` across `0000:00:{0b..0e}.0` and `0000:80:{00..03}.0`, exposed via `/dev/vfio/{0..7}` + `/dev/vfio/vfio`.
- **Memory:** `256 GiB` HBM total (`34,359,738,368` B raw / `31.25 GiB` usable per chip in JAX) and `1.4 TiB` host RAM.

---

## 2. CNCF CDI v0.6.0 (`ops/tpu-cdi`) Non-Privileged Container Proof
- Generated and validated CNCF CDI `v0.6.0` spec (`freetpu.dev/tpu`, 9 device entries: chips `0..7` + `all`): [`artifacts/freetpu_cdi.json`](artifacts/freetpu_cdi.json).
- Verified non-privileged container execution (**no `--privileged`**) in [`artifacts/02_tpu_cdi_container_jax.txt`](artifacts/02_tpu_cdi_container_jax.txt):
  ```bash
  docker run --rm \
    --device=freetpu.dev/tpu=all \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --net=host \
    --shm-size=16G \
    vllm/vllm-tpu:latest ...
  ```
  - Container JAX enumerated `8 × TPU v6 lite` and executed a `512×512` `bfloat16` matmul (`cdi_matmul_sum=512.0`).
  - **Field finding for follow-up code PR:** multi-chip `libtpu` coordinator initialization (`[::]:8353`) fails with `DEADLINE_EXCEEDED` on Docker's default bridge network; `--net=host` is required alongside CDI + `CAP_IPC_LOCK` + unlimited `memlock`.

---

## 3. Live `ops/tpu-exporter` Prometheus Telemetry
- Mapped all 8 chips (`device_id="0x006f"`, `generation="v6e"`) PCI BDF ↔ `/dev/vfio/*`, confirmed `freetpu_tpu_devices_discovered 8`, `freetpu_tpu_pci_enabled=1`, `freetpu_tpu_pid_telemetry_available=1`, and active `vllm` container PIDs (`freetpu_tpu_active_pids=1` across all 8 chips) in [`artifacts/03_tpu_exporter_serving.prom`](artifacts/03_tpu_exporter_serving.prom).
- Confirmed `freetpu_tpu_info_available 1` alongside `freetpu_tpu_pid_telemetry_available 1` when `tpu-info` is installed system-wide (`sudo pip3 install tpu-info`): [`artifacts/03_tpu_exporter_tpuinfo_ok.txt`](artifacts/03_tpu_exporter_tpuinfo_ok.txt).

---

## 4. Compute & HBM Microbenchmarks (`bench/freetpu_bench.py` + 8-Chip Sweep)

- Raw outputs: [`artifacts/04_bench_v6e8.json`](artifacts/04_bench_v6e8.json), [`artifacts/04_bench_v6e8.md`](artifacts/04_bench_v6e8.md), [`artifacts/04_v6e8_8chip_fp8_hbm.json`](artifacts/04_v6e8_8chip_fp8_hbm.json).

| Scope | Workload | Dtype | Median | Peak | Min / Floor |
| --- | --- | --- | ---: | ---: | ---: |
| 1-chip | `gemm_bf16` (`1×64×32`) | `bf16` | — | — | `139.56 us` min (`145.88 us` med) |
| 1-chip | `gemm_bf16` (`8192³`) | `bf16` | `647.07 TFLOP/s` | **`651.03 TFLOP/s`** | `1688.87 us` |
| 1-chip | `gemm_int8` (`8192³`) | `int8` | `1,135.14 TOPS` | **`1,146.58 TOPS`** | `958.95 us` |
| 1-chip | `fused_gemm_silu` (`4096³`) | `bf16+silu` | `435.68 TFLOP/s` | **`439.79 TFLOP/s`** | `312.51 us` |
| 1-chip | `gemm_w8a16` (`4096³`) | `w8a16_bf16` | `366.04 TFLOP/s` | **`372.44 TFLOP/s`** | `369.02 us` |
| 1-chip | `triad_bf16` (`1024 MiB` / `3072 MiB`) | `bf16` | `1,272.39 GB/s` | **`1,335.25 GB/s`** | `2531.64 us` med |
| 8-chip | `8 × 4096³` matmul | `bf16` | `2,163.76 TFLOP/s` | **`2,413.06 TFLOP/s`** | `455.65 us` |
| 8-chip | `8 × 4096³` native FP8 matmul | `float8_e4m3fn->bf16` | `1,857.61 TFLOP/s` | **`1,946.90 TFLOP/s`** | `564.75 us` |
| 8-chip | `8 × 4096³` matmul | `int8` | `2,731.84 TOPS` | **`2,934.46 TOPS`** | `374.69 us` |
| 8-chip | `8 ×` HBM triad (`12.88 GB` traffic) | `bf16` | `4,591.76 GB/s` | **`4,667.68 GB/s`** | `2760.45 us` |

---

## 5. `@freetpu.tile_kernel` Pallas/Mosaic Lowering & `cpu_ref` Parity

All 4 kernels (`64×512×512`, `block=64×512×512`) compiled via Pallas/Mosaic with `tpu_custom_call=True` and passed exact `cpu_ref` parity:

| Kernel | Dtype | `tpu_custom_call` | Mean HW Lat | Min HW Lat | Max Abs Err | NMSE vs `cpu_ref` | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `fused_matmul_silu` | `bf16` | `True` | `101.41 us` | `99.15 us` | `7.63e-06` | `2.41e-14` | [`05_tile_gemm_silu_bf16.json`](artifacts/05_tile_gemm_silu_bf16.json) |
| `fused_matmul_silu` | `w8a16` | `True` | `136.83 us` | `130.29 us` | `1.03e-02` | `1.05e-05` | [`05_tile_gemm_silu_w8a16.json`](artifacts/05_tile_gemm_silu_w8a16.json) |
| `swiglu` | `bf16` | `True` | `147.83 us` | `143.19 us` | `1.95e-03` | `1.56e-05` | [`05_tile_swiglu_bf16.json`](artifacts/05_tile_swiglu_bf16.json) |
| `swiglu` | `w8a16` | `True` | `153.47 us` | `141.30 us` | `1.62e-03` | `3.33e-05` | [`05_tile_swiglu_w8a16.json`](artifacts/05_tile_swiglu_w8a16.json) |

---

## 6. End-to-End `router/freetpu_router.py` + Non-Privileged CDI `vllm/vllm-tpu:latest` Serving

Served OpenAI SSE traffic (`max_tokens=128`, `max_model_len=4096`, `TP=8`) through `router/freetpu_router.py` (`28/28` HTTP `200`, `3,488` completion tokens per model; Prometheus counter snapshot in [`artifacts/06_router_metrics.prom`](artifacts/06_router_metrics.prom)):

| Model | Quant / Mode | 1-Stream Med TTFT | 1-Stream Decode | `c=8` Agg tok/s | `c=16` Agg tok/s | Spot `$ / 1M` (`$6.48/hr`) | Flex `$ / 1M` (`$10.80/hr`) | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-32B` | BF16 (`flax_nnx`, `TP=8`) | `17.99 ms` | `124.91 tok/s` | `883.29` | **`953.17`** | **`$1.89`** | **`$3.15`** | [`06_serving_bracket1_32b.json`](artifacts/06_serving_bracket1_32b.json) |
| `unsloth/Llama-3.3-70B-Instruct` | BF16 (`flax_nnx`, `TP=8`) | `31.09 ms` | `60.51 tok/s` | `463.60` | **`632.73`** | **`$2.84`** | **`$4.74`** | [`06_serving_bracket2_70b.json`](artifacts/06_serving_bracket2_70b.json) |

### Bracket 3 Root-Cause Findings (`artifacts/06_bracket3_root_causes.txt`)
Verbatim excerpts preserved in [`artifacts/06_bracket3_root_causes.txt`](artifacts/06_bracket3_root_causes.txt):
- **`openai/gpt-oss-120b` (`60.77 GiB` MXFP4):**
  - `TP=8` without EP failed sharding MXFP4 scale axis size `6` across `8` chips (`IndivisibleError`).
  - `--enable-expert-parallel` resolved sharding cleanly (`66.26 / 249.94 GiB` = `8.28 GiB/chip`, `1,120,442` KV tokens free), then hit two upstream compiler limits on Trillium `v6e`:
    1. Default `GMM EP` hit JAX SparseCore lowering `NotImplementedError` (`sc_lowering.py:645`).
    2. All three SparseCore-bypass paths (`MOE_TC_PERMUTE=1`, `ONEHOT_MOE_PERMUTE_THRESHOLD=32768`, `USE_MOE_EP_KERNEL=1`) cleanly bypassed SparseCore and failed in `v6e` Mosaic lowering because Trillium lacks native `f4E2M1FN` (`MXFP4`) support (`gmm_v2`: `Unsupported type 'vector<8x128x8xf4E2M1FN>'` in `tpu.unpack_subelements`; `fused_ep_moe`: `Bad lhs/rhs type: 'vector<16x256xbf16>' 'vector<256x256xf4E2M1FN>'` in `tpu.matmul`).
- **`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (`220.19 GiB` raw FP8):** loaded all 24 shards cleanly in `53.03s`, then hit `v6e-8`'s `250 GiB` usable HBM ceiling during `jit__process_quantized_moe_weights_impl` (`864.12 MiB` requested vs `592.05 MiB` free per chip).
- **`Qwen/Qwen3-235B-A22B-GPTQ-Int4`:** rejected upfront by `vllm-tpu` (`auto_gptq quantization is currently not supported in tpu`).

---

## Repro Summary
1. Generate CDI spec (`sudo ops/tpu-cdi/freetpu_cdi.py generate --validate --output /etc/cdi/freetpu.json`) and enable `features.cdi=true` in `/etc/docker/daemon.json`.
2. Install `tpu-info` system-wide (`sudo pip3 install tpu-info`) and start `sudo python3 ops/tpu-exporter/tpu_exporter.py --host 127.0.0.1 --port 9400`.
3. Run `bench/freetpu_bench.py` and `examples/tile_dsl/{fused_gemm_silu,swiglu}.py` directly on the TPU host.
4. Launch `vllm/vllm-tpu:latest` with `--device=freetpu.dev/tpu=all --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --net=host --shm-size=64G` and route traffic through `router/freetpu_router.py`.
