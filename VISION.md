# Vision

FreeTPU is an open, clean-room runtime and developer ecosystem for TPU silicon,
built for the masses so compute cannot stay locked behind proprietary vendor
SDKs and hardware oligopolies. Consumer Pixel Tensor hardware (`G5`/`G6`) is
the accessible beachhead that proves bare-metal control and multi-device
pooling; the destination is one native, open-source C++17 & bare-metal runtime
and portable tile programming model spanning edge TPUs, multi-chip datacenter
TPUs (`v5e`, `v6e`, `tpu7x`), and non-NVIDIA AI ASICs.

## In scope

- Clean-room native C++17 userspace runtimes, VLIW ISA specifications, and
  `llama.cpp` backends across edge and cloud TPUs:
  - **Edge (`Pixel Tensor G5/G6`):** `libopenbuenos` + `ggml-edgetpu` directly
    over `/dev/edgetpu-soc` with zero proprietary vendor `.so` dependencies.
  - **Cloud native runtime (`v5e` / `v6e` / `tpu7x`):** `ggml-cloudtpu` over
    the open C `PJRT_Api` (zero Python, JAX, or `vllm-tpu` at runtime).
  - **Cloud bare-metal moonshot:** `libopentpu.so` driving Cloud TPU PCIe
    silicon directly over `/dev/vfio/*` and exporting `GetPjrtApi()` with zero
    proprietary vendor `.so` dependencies.
- Wired multi-device clustering and pooled-RAM inference proving consumer TPUs
  can run models larger than any single device.
- One portable tile DSL lowering cleanly from a single kernel definition to
  both edge TPUs (`buenos_a0`, future Tensor generations) and Cloud TPUs
  (`v5e` / `v6e` / `tpu7x` Pallas/Mosaic and clean-room VLIW targets), plus
  drop-in quantized LLM blocks.
- Reproducible cross-silicon benchmarking (`freetpu-bench`), 60-second
  playbooks, headless R&D tooling, and portable Agent Skills so humans and
  coding agents can target TPUs in minutes. Small and mid-size dense models
  (`1.5B`–`70B`) serve strictly as fast, reproducible silicon benchmarks.
- Open TPU operations plumbing (`/dev/vfio/*` CNCF CDI container specs,
  Kubernetes DRA/device plugins, Prometheus `tpu-exporter`, and optional
  extension-hook router primitives) that make multi-chip TPUs feel as easy to
  operate as GPUs.

## Out of scope

- Proprietary vendor libraries, extracted firmware blobs (such as
  `dive_interp_rv64.bin`), compiled vendor models, or private research dumps.
- Commercial billing, Stripe payments, user accounts, or hosted SaaS
  control-plane code (owned separately by external hosting such as FreeTPU
  Cloud).
- Hosting commercial production APIs or reselling compute on ephemeral R&D
  sandboxes like Google Colab or non-commercial research grants like Google's
  TPU Research Cloud (TRC) — TRC is strictly for open, non-commercial
  Apache-2.0 R&D and public benchmarks, while production serving belongs on
  paid Cloud TPU VMs or dedicated partner silicon.
- Unverified claims or synthetic estimates presented as silicon proof.

## Guardrails

- **Clean-room and Apache-2.0 first.** Public source, specifications, and
  offline self-tests only; zero proprietary vendor dependencies in `DT_NEEDED`
  or Git history.
- **Native runtime over Python wrappers.** Treat `vllm-tpu` / JAX runs strictly
  as reference baselines; core delivery is native C++17 and bare-metal silicon
  control.
- **Silicon proof before claims.** Every ISA rule, hardware erratum, latency,
  and throughput number links to reproducible commands and real measurements.
- **Tile-native and transport-agnostic.** Build around tile abstractions and
  pluggable interconnects rather than single-device hacks.
- **Clean OSS / commercial boundary.** `saariuslystoned/freetpu` remains the
  open engine for everyone; commercial platforms wrap its public hooks without
  gating or forking core primitives.
