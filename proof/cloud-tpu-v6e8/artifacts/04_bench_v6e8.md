# `freetpu-bench` Cross-Silicon Scorecard

- **Timestamp (UTC)**: `2026-11-22T05:27:56Z`
- **Backend**: `jax_tpu` | **Host**: `Linux-x86_64` | **Suites**: `compute, bandwidth, comms`

## 1. Cross-Hardware Scorecard (`measured` vs `published-reference`)

| Target | Row Type | Memory | BF16 (TF/s) | INT8 (TOPS) | Mem BW (GB/s) | Dispatch Floor | $/hr | TF/s per $ | Est. $/1M tok |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| LIVE RUN: TPU v6 lite (jax_tpu) | `measured` | 31.25 GiB HBM | 651.03 | 1146.58 | 1335.24 | 145.88 us | — | — | — |
| Pixel Tensor G5 (buenos_a0) | `published-reference` | 15.6 GiB LPDDR (unified) | — | 0.99 | 27.96 | 60.8-150 us hw (395 us wall) | $0.05 | 19.9 (I8) | $5.67 |
| Cloud TPU v5e-1 | `published-reference` | 16 GB HBM2e | 179.36 | 350.69 | 655.55 | 117.6-135.0 us sync (32 us pipelined) | $1.20 | 149.5 | $5.81 |
| Cloud TPU v6e-1 (Trillium) | `published-reference` | 32 GB HBM2e | 918.00 | 1836.00 | 1638.00 | ~100 us PJRT est. | $2.70 | 340.0 | $5.23 |
| NVIDIA H100 SXM | `published-reference` | 80 GB HBM3 | 989.00 | 1979.00 | 3350.00 | ~5-15 us CUDA/Graph | $3.49 | 283.4 | $3.31 |
| NVIDIA RTX 5090 | `published-reference` | 32 GB GDDR7 | 209.50 | 419.00 | 1792.00 | ~5-15 us CUDA/Graph | $0.50 | 419.0 | $0.89 |

> **Normalized `$/1M tok` Assumptions**: `8.0 GB` active weight read per token (e.g. Llama-3-8B W8A16), `70%` memory-bandwidth decode efficiency, `batch_size=1` (`tok/s = BW * eff * B / model_gb`). Published references distinguish peak spec vs verified silicon runs.

## 2. Live `compute` Suite (`nvbench`-style)

| Kernel | Shape `(M,K,N)` | Dtype | Median Lat (us) | Min Lat (us) | Median Perf | Peak Perf |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `gemm_bf16` | `1x64x32` | `bf16` | 145.88 | 139.56 | 0.0 TFLOP/s | 0.0 TFLOP/s |
| `gemm_bf16` | `16x128x128` | `bf16` | 145.55 | 137.18 | 0.004 TFLOP/s | 0.004 TFLOP/s |
| `gemm_bf16` | `1x576x1536` | `bf16` | 148.28 | 139.28 | 0.012 TFLOP/s | 0.013 TFLOP/s |
| `gemm_bf16` | `1024x1024x1024` | `bf16` | 149.47 | 143.19 | 14.367 TFLOP/s | 14.997 TFLOP/s |
| `gemm_bf16` | `2048x2048x2048` | `bf16` | 169.27 | 161.74 | 101.494 TFLOP/s | 106.219 TFLOP/s |
| `gemm_bf16` | `4096x4096x4096` | `bf16` | 314.15 | 305.86 | 437.495 TFLOP/s | 449.352 TFLOP/s |
| `gemm_bf16` | `8192x8192x8192` | `bf16` | 1699.21 | 1688.87 | 647.072 TFLOP/s | 651.034 TFLOP/s |
| `gemm_int8` | `4096x4096x4096` | `int8` | 242.79 | 236.63 | 566.082 TOPS | 580.818 TOPS |
| `gemm_int8` | `8192x8192x8192` | `int8` | 968.61 | 958.95 | 1135.144 TOPS | 1146.579 TOPS |
| `fused_gemm_silu` | `4096x4096x4096` | `bf16+silu` | 315.46 | 312.51 | 435.678 TFLOP/s | 439.791 TFLOP/s |
| `gemm_w8a16` | `4096x4096x4096` | `w8a16_bf16` | 375.48 | 369.02 | 366.036 TFLOP/s | 372.444 TFLOP/s |

## 3. Live `bandwidth` Suite (`nvbandwidth`-style)

| Probe | Payload / Traffic | Median Lat (us) | Median BW (GB/s) | Peak BW (GB/s) |
| --- | --- | ---: | ---: | ---: |
| `triad_bf16` | 64.0 MiB tensor (192.0 MiB 3x) | 287.64 | 699.926 | 721.135 |
| `triad_bf16` | 256.0 MiB tensor (768.0 MiB 3x) | 709.16 | 1135.578 | 1145.887 |
| `triad_bf16` | 1024.0 MiB tensor (3072.0 MiB 3x) | 2531.64 | 1272.385 | 1335.245 |
| `h2d_device_put` | 4096 B | 141.03 | 0.029 | — |
| `d2h_to_host` | 4096 B | 95.32 | 0.043 | — |
| `h2d_device_put` | 65536 B | 144.40 | 0.454 | — |
| `d2h_to_host` | 65536 B | 96.78 | 0.677 | — |
| `h2d_device_put` | 4194304 B | 400.05 | 10.484 | — |
| `d2h_to_host` | 4194304 B | 409.96 | 10.231 | — |

## 4. Live `comms` Suite (`nccl-tests`-style, `tcp_loopback`)

| Payload | 1-Way Hop Median (us) | RTT Median (us) | RTT p95 (us) | Median Bidir (GB/s) | Peak Bidir (GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16 KB` | 20.59 | 41.19 | 51.15 | 0.796 | 1.015 |
| `64 KB` | 37.18 | 74.36 | 95.45 | 1.763 | 1.788 |
| `256 KB` | 103.64 | 207.27 | 235.98 | 2.530 | 2.744 |
| `1 MB` | 306.94 | 613.89 | 663.10 | 3.416 | 3.542 |
| `4 MB` | 1114.62 | 2229.25 | 2424.71 | 3.763 | 3.900 |

## 5. Published Pricing & Hardware Citations

- **Pixel Tensor G5 (buenos_a0)**: $1,299 Pixel 10 Pro amortized 3yr 24/7 ($0.049/hr) + $0.001/hr power (~5W); $0 cloud bill — *FreeTPU Phase 6 Report & buenos_a0 silicon probes (W8A16 BF16 1x64x32 = 411-426 us hw)*.
- **Cloud TPU v5e-1**: Google Cloud TPU v5e public on-demand (us-central1/us-east5/us-west1: $1.20/chip-hr) — *Google Cloud TPU v5e spec + FreeTPU Track B live v5e-1 silicon run*.
- **Cloud TPU v6e-1 (Trillium)**: Google Cloud TPU v6e public on-demand (us-east1/us-east5: $2.70/chip-hr) — *Google Cloud TPU v6e (Trillium) public architecture & pricing docs*.
- **NVIDIA H100 SXM**: Specialized GPU cloud on-demand ($3.49/GPU-hr reference; hyperscaler 8-GPU VMs ~$7-11/GPU-hr) — *NVIDIA H100 SXM datasheet (dense Tensor Core, non-sparse) & public cloud rates*.
- **NVIDIA RTX 5090**: Specialized community GPU cloud ($0.50/GPU-hr midpoint of $0.32-$0.80/hr; $1,999 FE MSRP ~$0.20/hr self-hosted) — *NVIDIA Blackwell RTX 5090 spec (dense non-sparse Tensor Core) & public community cloud pricing*.
