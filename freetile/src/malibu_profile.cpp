// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 Pixel 11 (malibu / Tensor G6 santafe) Hardware Descriptor
// & Phase-Aware 3-Engine Hybrid Scheduling Profile.

#include "freetile/malibu_profile.hpp"

#include <algorithm>

namespace freetile {

namespace {

bool set_err(std::string* err, const char* msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

const MalibuHardwareProfile kMalibuProfile{};

bool has_other_heavy_engine_active(uint32_t active_mask, MalibuComputeEngine chosen) {
    constexpr uint32_t kHeavyMask =
        malibu_engine_bit(MalibuComputeEngine::EDGETPU_HEAVY) |
        malibu_engine_bit(MalibuComputeEngine::CPU_SME2) |
        malibu_engine_bit(MalibuComputeEngine::GPU_VULKAN_BDA);
    const uint32_t other_heavy = active_mask & kHeavyMask & ~malibu_engine_bit(chosen);
    return other_heavy != 0u;
}

}  // namespace

const char* malibu_engine_name(MalibuComputeEngine engine) {
    switch (engine) {
        case MalibuComputeEngine::EDGETPU_HEAVY:  return "edgetpu_heavy";
        case MalibuComputeEngine::EDGETPU_NANO:   return "edgetpu_nano";
        case MalibuComputeEngine::CPU_SME2:       return "cpu_sme2";
        case MalibuComputeEngine::CPU_NEON:       return "cpu_neon";
        case MalibuComputeEngine::GPU_VULKAN_BDA: return "gpu_vulkan_bda";
    }
    return "unknown";
}

const char* malibu_phase_name(MalibuWorkloadPhase phase) {
    switch (phase) {
        case MalibuWorkloadPhase::PREFILL_DENSE:         return "prefill_dense";
        case MalibuWorkloadPhase::DECODE_M1:             return "decode_m1";
        case MalibuWorkloadPhase::ATTENTION_SHORT_CTX:   return "attention_short_ctx";
        case MalibuWorkloadPhase::ATTENTION_LONG_CTX:    return "attention_long_ctx";
        case MalibuWorkloadPhase::VISION_EMBEDDING:      return "vision_embedding";
        case MalibuWorkloadPhase::ELEMENTWISE_NORM_ROPE: return "elementwise_norm_rope";
    }
    return "unknown";
}

const char* malibu_quant_name(MalibuQuantKind quant) {
    switch (quant) {
        case MalibuQuantKind::F16:    return "f16";
        case MalibuQuantKind::BF16:   return "bf16";
        case MalibuQuantKind::Q8_0:   return "q8_0";
        case MalibuQuantKind::Q4_0:   return "q4_0";
        case MalibuQuantKind::Q4_K:   return "q4_K";
        case MalibuQuantKind::Q6_K:   return "q6_K";
        case MalibuQuantKind::IQ4_NL: return "iq4_nl";
        case MalibuQuantKind::IQ4_XS: return "iq4_xs";
    }
    return "unknown";
}

const MalibuHardwareProfile& malibu_hardware_profile() {
    return kMalibuProfile;
}

TargetProfile malibu_a0_target_profile() {
    TargetProfile p;
    // Retain ChipTarget::BUENOS_A0 enum slot until ChipTarget::MALIBU_A0 is merged into tile_ir.hpp.
    p.target = ChipTarget::BUENOS_A0;
    p.name = kMalibuProfile.target_name;
    p.mxu_m = 64;
    p.mxu_n = 64;
    p.sublane_align = 8;
    p.lane_align = 64;
    p.vmem_bytes = kMalibuProfile.tpu.core_narrow_sram_bytes;
    p.smem_bytes = 64ULL * 1024ULL;
    p.supports_mosaic = false;
    p.supports_s4_native = kMalibuProfile.tpu.supports_s4_native;  // false (Day-1 safe)
    p.supports_bf16 = kMalibuProfile.tpu.supports_bf16;
    p.mosaic_version = 0;
    return p;
}

bool malibu_engine_saturates_lpddr5x(MalibuComputeEngine engine) {
    switch (engine) {
        case MalibuComputeEngine::EDGETPU_HEAVY:
        case MalibuComputeEngine::CPU_SME2:
        case MalibuComputeEngine::GPU_VULKAN_BDA:
            return true;
        case MalibuComputeEngine::EDGETPU_NANO:
        case MalibuComputeEngine::CPU_NEON:
            return false;
    }
    return true;
}

bool malibu_allows_concurrent_engines(MalibuComputeEngine a, MalibuComputeEngine b) {
    if (a == b) {
        return true;
    }
    // Never allow two heavy LPDDR5X-saturating engines concurrently (prevents 30s thermal cliff).
    if (malibu_engine_saturates_lpddr5x(a) && malibu_engine_saturates_lpddr5x(b)) {
        return false;
    }
    return true;
}

uint32_t malibu_clamp_sme2_threads(uint32_t requested_threads,
                                   const MalibuHardwareProfile& prof) {
    const uint32_t cap = (prof.cpu.default_sme_thread_cap > 0u)
                             ? prof.cpu.default_sme_thread_cap
                             : 1u;
    if (requested_threads == 0u) {
        return cap;
    }
    return std::min(requested_threads, cap);
}

bool malibu_quant_has_native_sme2_kernel(MalibuQuantKind quant) {
    return quant == MalibuQuantKind::Q4_0 || quant == MalibuQuantKind::Q8_0;
}

bool route_malibu_workload(const MalibuRouteRequest& req,
                           MalibuRouteDecision* out_decision,
                           const MalibuHardwareProfile& prof,
                           std::string* err) {
    if (out_decision == nullptr) {
        return set_err(err, "out_decision must not be null");
    }
    if (req.batch_m == 0u) {
        return set_err(err, "batch_m must be > 0");
    }

    MalibuRouteDecision d;
    d.sme2_native_zero_repack = malibu_quant_has_native_sme2_kernel(req.quant);
    d.requires_gpu_workgroup_serialization =
        (req.n_ctx > prof.gpu.serialize_workgroups_n_ctx_threshold);

    switch (req.phase) {
        case MalibuWorkloadPhase::VISION_EMBEDDING: {
            d.engine = prof.tpu.has_nano_core ? MalibuComputeEngine::EDGETPU_NANO
                                              : MalibuComputeEngine::EDGETPU_HEAVY;
            d.fallback_engine = MalibuComputeEngine::EDGETPU_HEAVY;
            d.recommended_threads = 1u;
            d.cpu_affinity_mask = 0x03u;  // efficiency cores CPUs 0..1 for lightweight host control
            d.requires_syscall_barrier = false;
            d.reason = "Route vision embeddings to Darwinn Nano-TPU (~0.77 ms, host CPU/GPU idle)";
            break;
        }

        case MalibuWorkloadPhase::PREFILL_DENSE: {
            if (req.mesa_pco_compute_ready && req.n_ctx >= 4096u && req.batch_m >= 256u) {
                d.engine = MalibuComputeEngine::GPU_VULKAN_BDA;
                d.fallback_engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.recommended_threads = 1u;
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_gpu_bda =
                    prof.gpu.supports_bda &&
                    (req.max_tensor_bytes == 0u ||
                     req.max_tensor_bytes > prof.gpu.max_storage_buffer_range_bytes);
                d.requires_syscall_barrier = prof.cpu.syscall_clears_pstate_sm;
                d.reason =
                    "Route long-context batched prefill to PowerVR CXTP-48-1536 Vulkan BDA (wave128)";
            } else {
                d.engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.fallback_engine = MalibuComputeEngine::CPU_SME2;
                d.recommended_threads = 1u;
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_syscall_barrier = prof.cpu.syscall_clears_pstate_sm;
                d.reason =
                    "Route compute-bound pp512 prefill to Darwinn Heavy EdgeTPU (<0.5 W rail; "
                    "Day-1 u8 load-time unpack, supports_s4_native=false)";
            }
            break;
        }

        case MalibuWorkloadPhase::DECODE_M1: {
            if (req.quant == MalibuQuantKind::IQ4_NL || req.quant == MalibuQuantKind::IQ4_XS) {
                d.engine = MalibuComputeEngine::CPU_NEON;
                d.fallback_engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.recommended_threads = malibu_clamp_sme2_threads(req.requested_threads, prof);
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_syscall_barrier = false;
                d.reason =
                    "Route M=1 IQ4 decode to ARMv9 NEON/i8mm on CPUs 2..6 "
                    "(non-linear I-quants lack native KleidiAI SME2 microkernels)";
            } else if (prof.cpu.has_sme2) {
                d.engine = MalibuComputeEngine::CPU_SME2;
                d.fallback_engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.recommended_threads = malibu_clamp_sme2_threads(req.requested_threads, prof);
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_syscall_barrier = prof.cpu.syscall_clears_pstate_sm;
                if (d.sme2_native_zero_repack) {
                    d.reason =
                        "Route M=1 Q4_0/Q8_0 decode to ARMv9 SME2 CPU (CPUs 2..6, threads <= "
                        "sme_thread_cap, zero-repack KleidiAI microkernel, 68.2 GB/s LPDDR5X)";
                } else {
                    d.reason =
                        "Route M=1 decode to ARMv9 SME2 CPU (CPUs 2..6, threads <= sme_thread_cap; "
                        "note 10-15% runtime repack penalty for K-quants)";
                }
            } else {
                d.engine = MalibuComputeEngine::CPU_NEON;
                d.fallback_engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.recommended_threads = malibu_clamp_sme2_threads(req.requested_threads, prof);
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.reason = "Fallback M=1 decode to ARMv9 NEON (SME2 disabled in profile)";
            }
            break;
        }

        case MalibuWorkloadPhase::ATTENTION_SHORT_CTX:
        case MalibuWorkloadPhase::ELEMENTWISE_NORM_ROPE: {
            d.engine = prof.cpu.has_sme2 ? MalibuComputeEngine::CPU_SME2
                                         : MalibuComputeEngine::CPU_NEON;
            d.fallback_engine = MalibuComputeEngine::CPU_NEON;
            d.recommended_threads = malibu_clamp_sme2_threads(req.requested_threads, prof);
            d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
            d.requires_syscall_barrier = prof.cpu.has_sme2 && prof.cpu.syscall_clears_pstate_sm;
            d.reason =
                "Keep short-context attention / RMSNorm / RoPE on CPU (10-30 us beats ~300 us NPU VII)";
            break;
        }

        case MalibuWorkloadPhase::ATTENTION_LONG_CTX: {
            if (req.seq_len >= 512u) {
                d.engine = MalibuComputeEngine::EDGETPU_HEAVY;
                d.fallback_engine = MalibuComputeEngine::CPU_SME2;
                d.recommended_threads = 1u;
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_syscall_barrier = prof.cpu.syscall_clears_pstate_sm;
                d.reason =
                    "Route long-context (seq_len >= 512) GQA FlashAttention to Darwinn Heavy EdgeTPU";
            } else {
                d.engine = MalibuComputeEngine::CPU_SME2;
                d.fallback_engine = MalibuComputeEngine::CPU_NEON;
                d.recommended_threads = malibu_clamp_sme2_threads(req.requested_threads, prof);
                d.cpu_affinity_mask = prof.cpu.sme_perf_cpu_mask;
                d.requires_syscall_barrier = prof.cpu.syscall_clears_pstate_sm;
                d.reason = "Route sub-512 attention to SME2 CPU to avoid NPU dispatch tax";
            }
            break;
        }
    }

    if (malibu_engine_saturates_lpddr5x(d.engine) &&
        has_other_heavy_engine_active(req.active_engine_mask, d.engine)) {
        d.requires_bus_serialization_wait = true;
    }

    // If transitioning from active SME2 to a syscall-backed engine (EdgeTPU ioctl / Vulkan ioctl),
    // Linux 6.12+ clears PSTATE.SM and wipes ZA registers at syscall entry.
    if (prof.cpu.syscall_clears_pstate_sm &&
        ((req.active_engine_mask & malibu_engine_bit(MalibuComputeEngine::CPU_SME2)) != 0u) &&
        d.engine != MalibuComputeEngine::CPU_SME2) {
        d.requires_syscall_barrier = true;
    }

    *out_decision = d;
    return true;
}

}  // namespace freetile
