# Headless 7-Slot Colab Fleet Dispatcher & GPU Fast-Kernel Toolkit

Clean-room operator guide for FreeTPU's multi-slot ephemeral Google Colab fleet dispatcher ([`tools/colab_fleet.py`](../../tools/colab_fleet.py)) and Hopper/Ampere/Ada GPU training & RL acceleration toolkit ([`tools/h100_fast_kernels.py`](../../tools/h100_fast_kernels.py), implementing `freesilicon/.github#1`).

> **Clean-Room & Public-Data Boundary:** Strictly for personal/R&D OSS workloads using public repositories and synthetic or open datasets. Never upload private data, secrets, tokens, or credentials to ephemeral Colab VMs. Work queues and object-storage endpoints are supplied strictly through external configuration files and environment variables.

---

## 1. GPU Training & RL Speedup Toolkit (`tools/h100_fast_kernels.py`)

Implements all 5 training & RL acceleration primitives with automatic hardware detection and bit-exact fallback on CPU/MPS/lower-VRAM GPUs:

1. **`ensure_causal_conv1d()`** — Prefers prebuilt binary wheels (`--only-binary=:all:`) with optional wheelhouse caching and graceful fallback to PyTorch `F.conv1d` for `sconv_kernel_size=4` hybrid layers.
2. **`configure_liger_and_checkpointing()`** — Patches only weighted `RMSNorm` modules (protecting unweighted value norms, SiTU softcaps `4.0/25.0`, no-RoPE relative bias, and `final_logit_softcapping=30.0`), verifies `LigerFusedLinearCrossEntropyLoss(..., softcap=30.0)` support, and automatically disables `gradient_checkpointing` on `>= 70 GiB` GPUs (`H100`/`A100-80GB`).
3. **`pack_sft_sequences()`** — First-Fit Decreasing dynamic sequence packer (`max_len=2048`) emitting per-segment `position_ids`, 4D additive block-diagonal causal masks, `-100` prompt/separator labels, and `sconv_guard=3` masked boundary tokens for mathematically `0.0` cross-example attention and `sconv(k=4)` leakage.
4. **`grouped_moe_dispatch()` / `patch_inkling2_experts_with_grouped_mm()`** — Fused 64-expert MoE + `ExpertLoRA` dispatch via `torch._grouped_mm` (`sm_80+` BF16/FP16) with bit-exact eager fallback.
5. **Fast GRPO Rollouts, Opt-In Hopper FP8 & Startup Cache:**
   - `FastRolloutEngine` — Colocated `vLLM` sleep-mode (`wake_up()` → `generate()` → `sleep(level=1)`) or `torch.compile(mode="reduce-overhead")` CUDA-graph static-cache decode with eager fallback.
   - `configure_hopper_fp8()` — Opt-in `sm_90+` FP8 training via `torchao.float8` (strictly skipping `lm_head`, MoE routers, norms, embeddings, and `< 256` dims) with safe BF16 fallback on `sm_80`/`sm_89`.
   - `StartupArtifactCache` — Content-addressed SHA-256 cache for public prebuilt wheels and deterministic pruned bases.

```bash
# Offline CI self-test (0 CUs, works with or without PyTorch)
python3 tools/h100_fast_kernels.py --self-test
python3 tools/test_h100_fast_kernels.py -v
```

### Live Silicon Verification (`NVIDIA L4`, `sm_89`, `PyTorch 2.14.1+cu130`, `2.33 CU/hr`)
- **Fused MoE `GroupedGEMM` (`torch._grouped_mm`, `E=64, top_k=8, H=2304, I=1152, ExpertLoRA r=16`):** **`4.35×`** faster at `512` tokens (`119.57 -> 27.52 ms`) and **`9.54×`** faster at `2048` packed tokens (`470.90 -> 49.35 ms`) with `0.0` forward max abs error and `1.0` forward/backward cosine similarity.
- **Dynamic SFT Sequence Packing (`128` short examples → `9` bins of `2048`, `sconv_guard=3`):** **`1.83×`** wall speedup (`69,021 -> 126,439 tok/s`, `97.86%` density, `0.0` cross-example attention & `sconv(k=4)` leak).
- **`liger-kernel` + VRAM-Aware GC:** Weighted `LigerRMSNorm` **`1.76×`** faster (`1.413 -> 0.802 ms`, `12/12` unweighted value norms skipped); `LigerFusedLinearCrossEntropyLoss(softcap=30.0)` on `201,024`-vocab saves **`3.03 GiB` VRAM** (`5,116.6 -> 2,082.0 MiB`, `1e-6` loss parity, auto-recommended on `< 35 GiB` GPUs while keeping fast eager softcap CE on `>= 40 GiB` A100/H100); `gradient_checkpointing=False` gives **`1.44×`** speedup (`105.37 -> 72.96 ms`, `0.0` loss diff) on 36 looped layers (`auto-off >= 70 GiB`).
- **Fast Rollouts, Hopper FP8 Guard & Cache:** `torch.compile(reduce-overhead)` verified exact greedy parity; `configure_hopper_fp8` cleanly falls back to BF16 on `sm_89 < sm_90`; `StartupArtifactCache` verified `24.05 ms` SHA-256 hit on an `8 MiB` shard.

---

## 2. 7-Slot Colab Fleet Dispatcher (`tools/colab_fleet.py`)

Extends [`tools/colab_tpu_bench.py`](../../tools/colab_tpu_bench.py) isolation and zero-idle-burn patterns to concurrent multi-lane GPU + TPU + High-RAM CPU sweeps:

### Concurrency & Slot Isolation
- **Total Cap (`--max-total-slots 7`):** Queries live `colab sessions` and counts all active assignments (including pre-existing external sessions). Never stops, execs on, or touches any session not prefixed `fleet-`.
- **Accelerator Caps & Ordered Fallback (`COLAB_CLI_ASSIGN_RETRIES=0`):**
  - GPU (`--max-gpu-slots 3`): `H100 -> A100_HM -> A100 -> L4 -> T4_HM -> T4` with per-SKU cooldown backoff on `503`/`429` and immediate stop on `412`.
  - TPU (`--max-tpu-slots 2`): `V6E1 -> V5E1` with verified-zero-leak reconciliation on `ReadTimeout`.
  - CPU (`--max-cpu-slots 4`): `CPU_HM` (`8` vCPU / `51 GiB` RAM) and `CPU` workers for verifiers and DiLoCo/merge coordination.

### Dual Pluggable Sync Backends
- **`--sync local` (Default):** Zero cloud credentials on Colab. Stages inputs via `colab upload` and retrieves artifacts via `colab download` to a private `0700` local directory.
- **`--sync s3`:** Generic S3-compatible object store (Cloudflare R2, GCS HMAC, MinIO, AWS S3) using pure-stdlib AWS Signature V4 configured via environment variables (`FREETPU_S3_ENDPOINT`, `FREETPU_S3_BUCKET`, `FREETPU_S3_ACCESS_KEY_ID`, `FREETPU_S3_SECRET_ACCESS_KEY`, optional `FREETPU_S3_REGION=auto`, `FREETPU_S3_PREFIX`). Credentials are never logged or written to `STATE.md`, `events.jsonl`, or `PROOF.md`.

### Two-Stage CU Governor
- **Sprint Mode (`balance_cu > --floor-cu`, default `500` CUs):** Runs up to 7 slots concurrently across queued waves until balance reaches `--floor-cu`.
- **Steady-State Mode (`balance_cu <= --floor-cu`):** Throttles concurrency and caps daily spend at `--steady-daily-cap-cu` (default `65` CUs/day ≈ `2,000` CUs/month rollover), idling cleanly at `0` CUs when no new work is admitted.
- **Hard Emergency Stop (`--hard-stop-cu 50`):** Refuses new allocations below the safety floor.

```bash
# Offline self-test & unit tests (0 CUs, 0 network)
python3 tools/colab_fleet.py --self-test
python3 tools/test_colab_fleet.py -v

# Dry-run an external public queue without provisioning VMs
python3 tools/colab_fleet.py --dry-run --queue /path/to/public_queue.json --sync local

# Execute sprint-to-500-CU fleet run with durable STATE.md / events.jsonl / heartbeat / PROOF.md
python3 tools/colab_fleet.py \
  --queue /path/to/public_queue.json \
  --run-dir /path/to/private-run-dir \
  --sync local \
  --floor-cu 500 \
  --steady-daily-cap-cu 65
```
