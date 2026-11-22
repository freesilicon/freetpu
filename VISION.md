# Vision

FreeTPU is an open, clean-room runtime and developer ecosystem for TPU silicon,
built for the masses so compute cannot stay locked behind proprietary vendor
SDKs and hardware oligopolies. Consumer Pixel Tensor hardware is the accessible
beachhead that proves bare-metal control and multi-device pooling; the
destination is one portable tile programming model spanning edge TPUs,
multi-chip datacenter TPUs, and non-NVIDIA AI ASICs.

## In scope

- Clean-room userspace runtimes, VLIW ISA specifications, and `llama.cpp`
  backends with zero proprietary vendor `.so` dependencies (`libopenbuenos`,
  `ggml-edgetpu`).
- Wired multi-device clustering and pooled-RAM inference proving consumer TPUs
  can run models larger than any single device.
- One portable tile DSL lowering cleanly from a single kernel definition to
  both edge TPUs (`buenos_a0`, future Tensor generations) and Cloud TPUs
  (`v5e` / `v6e` Pallas/Mosaic), plus drop-in quantized LLM blocks.
- Reproducible cross-silicon benchmarking (`freetpu-bench`), 60-second
  playbooks, headless R&D tooling, and portable Agent Skills so humans and
  coding agents can target TPUs in minutes.
- Open TPU operations plumbing (`/dev/vfio/*` CDI container specs, Prometheus
  `tpu-exporter`) and an open OpenAI-compatible TPU router (`v0`) with clean
  extension hooks.

## Out of scope

- Proprietary vendor libraries, extracted firmware blobs (such as
  `dive_interp_rv64.bin`), compiled vendor models, or private research dumps.
- Commercial billing, Stripe payments, user accounts, or hosted SaaS
  control-plane code (owned separately by external hosting such as FreeTPU
  Cloud).
- Hosting public production APIs or reselling compute on ephemeral R&D
  sandboxes like Google Colab (production multi-chip serving belongs on Cloud
  TPU VMs or dedicated partner silicon).
- Unverified claims or synthetic estimates presented as silicon proof.

## Guardrails

- **Clean-room and Apache-2.0 first.** Public source, specifications, and
  offline self-tests only; zero proprietary vendor dependencies in `DT_NEEDED`
  or Git history.
- **Silicon proof before claims.** Every ISA rule, hardware erratum, latency,
  and throughput number links to reproducible commands and real measurements.
- **Tile-native and transport-agnostic.** Build around tile abstractions and
  pluggable interconnects rather than single-device hacks.
- **Clean OSS / commercial boundary.** `saariuslystoned/freetpu` remains the
  open engine for everyone; commercial platforms wrap its public hooks without
  gating or forking core primitives.
