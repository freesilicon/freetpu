// SPDX-License-Identifier: Apache-2.0
// Clean-room zero-dependency C++17 Pixel 11 (malibu / Tensor G6 santafe) Hardware Descriptor
// & Phase-Aware 3-Engine Hybrid Scheduling Profile.
//
// Encodes empirical Tensor G6 constants (7-core ARMv9 SME/SME2 1x 0xd8c + 6x 0xd8b,
// PowerVR CXTP-48-1536 MC1 wave128 + 128 MiB maxStorageBufferRange + BDA, Darwinn
// gchips-mbu Heavy+Nano Dual-TPU, 68.2 GB/s unified LPDDR5X, Day-1 safe
// supports_s4_native = false, Linux 6.12+ PSTATE.SM syscall-boundary flag) and enforces
// phase-aware routing with zero concurrent LPDDR5X bus saturation.
//
// Builds cleanly under -std=c++17 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti.

#pragma once

#include "freetile/tile_ir.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace freetile {

constexpr uint16_t kMalibuCpuPartPrime = 0x0D8Cu;  // Cortex-C1-Ultra (1x @ 4.11 GHz)
constexpr uint16_t kMalibuCpuPartPro = 0x0D8Bu;    // Cortex-C1-Pro (4x @ 3.38 GHz + 2x @ 2.65 GHz)
constexpr uint32_t kMalibuCpuCount = 7u;
constexpr uint32_t kMalibuDefaultSmeThreadCap = 5u;  // CPUs 2..6 (policy2 + policy6)
constexpr uint32_t kMalibuSmePerfCpuMask = 0x7Cu;    // bits 2,3,4,5,6 -> 4x C1-Pro mid + 1x C1-Ultra

constexpr uint32_t kMalibuGpuVendorId = 0x1010u;
constexpr uint32_t kMalibuGpuDeviceId = 0x70058002u;
constexpr uint32_t kMalibuGpuWaveSize = 128u;
constexpr size_t kMalibuGpuMaxStorageBufferBytes = 128ULL * 1024ULL * 1024ULL;
constexpr uint32_t kMalibuGpuSerializeNCtxThreshold = 7168u;

constexpr size_t kMalibuUnifiedRamBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr float kMalibuLpddr5xBwGbps = 68.2f;
constexpr uint32_t kMalibuLpddr5xFreqMhz = 4200u;

enum class MalibuComputeEngine : uint8_t {
    EDGETPU_HEAVY = 0,
    EDGETPU_NANO = 1,
    CPU_SME2 = 2,
    CPU_NEON = 3,
    GPU_VULKAN_BDA = 4,
};

enum class MalibuWorkloadPhase : uint8_t {
    PREFILL_DENSE = 0,         // pp512 / batched prompt processing
    DECODE_M1 = 1,             // autoregressive M=1 token generation
    ATTENTION_SHORT_CTX = 2,   // seq_len < 512
    ATTENTION_LONG_CTX = 3,    // seq_len >= 512
    VISION_EMBEDDING = 4,      // continuous / background vision embeddings
    ELEMENTWISE_NORM_ROPE = 5, // RMSNorm, RoPE, residual add
};

enum class MalibuQuantKind : uint8_t {
    F16 = 0,
    BF16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
    Q4_K = 4,
    Q6_K = 5,
    IQ4_NL = 6,
    IQ4_XS = 7,
};

const char* malibu_engine_name(MalibuComputeEngine engine);
const char* malibu_phase_name(MalibuWorkloadPhase phase);
const char* malibu_quant_name(MalibuQuantKind quant);

constexpr uint32_t malibu_engine_bit(MalibuComputeEngine engine) {
    return 1u << static_cast<uint32_t>(engine);
}

struct MalibuCpuClusterProfile {
    uint32_t total_cores = kMalibuCpuCount;
    uint32_t prime_cores = 1u;
    uint32_t mid_cores = 4u;
    uint32_t eff_cores = 2u;
    uint16_t prime_part_id = kMalibuCpuPartPrime;
    uint16_t pro_part_id = kMalibuCpuPartPro;
    uint32_t prime_max_mhz = 4110u;
    uint32_t mid_max_mhz = 3380u;
    uint32_t eff_max_mhz = 2650u;
    bool has_sme = true;
    bool has_sme2 = true;
    bool has_smei8i32 = true;
    bool has_smef16f32 = true;
    bool has_smeb16f32 = true;
    bool has_smef32f32 = true;
    uint32_t sme_perf_cpu_mask = kMalibuSmePerfCpuMask;
    uint32_t default_sme_thread_cap = kMalibuDefaultSmeThreadCap;
    uint32_t linux_sme_min_major = 6u;
    uint32_t linux_sme_min_minor = 12u;
    bool syscall_clears_pstate_sm = true;
};

struct MalibuGpuProfile {
    const char* name = "PowerVR CXTP-48-1536 MC1";
    uint32_t vendor_id = kMalibuGpuVendorId;
    uint32_t device_id = kMalibuGpuDeviceId;
    uint32_t clock_mhz = 1300u;
    uint32_t compute_units = 6u;
    uint32_t subgroup_size = kMalibuGpuWaveSize;
    size_t max_storage_buffer_range_bytes = kMalibuGpuMaxStorageBufferBytes;
    bool supports_bda = true;
    bool supports_cooperative_matrix = true;
    uint32_t vulkan_major = 1u;
    uint32_t vulkan_minor = 4u;
    uint32_t vulkan_patch = 317u;
    uint32_t serialize_workgroups_n_ctx_threshold = kMalibuGpuSerializeNCtxThreshold;
};

struct MalibuTpuProfile {
    const char* codename = "santafe";
    const char* fw_family = "gchips-mbu";
    bool dual_heavy_nano = true;
    bool has_heavy_core = true;
    bool has_nano_core = true;
    size_t core_narrow_sram_bytes = 786432u;  // Day-1 conservative floor until unlocked G6 probe
    uint32_t async_queue_credits = 16u;
    float estimated_tops = 27.0f;
    bool supports_s4_native = false;   // Day-1 safe assumption (load-time u8 unpack)
    bool supports_hw_huffman = false;  // Day-1 safe assumption until unlocked G6 probe
    bool supports_w8a8 = true;
    bool supports_w8a16 = true;
    bool supports_fp16 = true;
    bool supports_bf16 = true;
    bool silicon_abi_verified = false;  // Explicitly false until unlocked Pixel 11 root probe
};

struct MalibuHardwareProfile {
    const char* soc_codename = "malibu";
    const char* target_name = "malibu_a0";
    const char* modem_name = "MediaTek M90";
    uint32_t process_node_nm = 2u;
    size_t unified_ram_bytes = kMalibuUnifiedRamBytes;
    float lpddr5x_bw_gbps = kMalibuLpddr5xBwGbps;
    uint32_t lpddr5x_freq_mhz = kMalibuLpddr5xFreqMhz;
    uint32_t max_concurrent_bw_saturating_engines = 1u;
    MalibuCpuClusterProfile cpu;
    MalibuGpuProfile gpu;
    MalibuTpuProfile tpu;
};

// Returns the clean-room empirical Pixel 11 (malibu / Tensor G6 santafe) hardware descriptor.
const MalibuHardwareProfile& malibu_hardware_profile();

// Returns a FreeTile TargetProfile skeleton for Malibu A0 (day-1 safe supports_s4_native=false,
// supports_mosaic=false, 768 KiB conservative narrow SRAM floor) without altering ChipTarget.
TargetProfile malibu_a0_target_profile();

// Returns true if the given engine saturates the shared 68.2 GB/s LPDDR5X bus under LLM streams.
bool malibu_engine_saturates_lpddr5x(MalibuComputeEngine engine);

// Returns true iff engines a and b may stream concurrently without violating the zero
// concurrent LPDDR5X saturation rule (preventing 30s thermal throttling cliffs).
bool malibu_allows_concurrent_engines(MalibuComputeEngine a, MalibuComputeEngine b);

// Clamps requested SME2 worker threads to [1, profile.cpu.default_sme_thread_cap] so KleidiAI
// never silently collapses SME2 matmuls back to NEON.
uint32_t malibu_clamp_sme2_threads(uint32_t requested_threads,
                                   const MalibuHardwareProfile& prof = malibu_hardware_profile());

// Returns true if the quantization format has native zero-repack KleidiAI SME2 microkernels.
bool malibu_quant_has_native_sme2_kernel(MalibuQuantKind quant);

struct MalibuRouteRequest {
    MalibuWorkloadPhase phase = MalibuWorkloadPhase::DECODE_M1;
    MalibuQuantKind quant = MalibuQuantKind::Q4_0;
    uint32_t batch_m = 1u;
    uint32_t seq_len = 512u;
    uint32_t n_ctx = 4096u;
    size_t max_tensor_bytes = 0u;
    uint32_t requested_threads = 7u;
    uint32_t active_engine_mask = 0u;      // bitmask via malibu_engine_bit(...)
    bool mesa_pco_compute_ready = false;   // true only once Mesa pco Volcanic compute is live
};

struct MalibuRouteDecision {
    MalibuComputeEngine engine = MalibuComputeEngine::CPU_SME2;
    MalibuComputeEngine fallback_engine = MalibuComputeEngine::CPU_NEON;
    uint32_t recommended_threads = 5u;
    uint32_t cpu_affinity_mask = kMalibuSmePerfCpuMask;
    bool sme2_native_zero_repack = false;
    bool requires_syscall_barrier = false;
    bool requires_gpu_bda = false;
    bool requires_gpu_workgroup_serialization = false;
    bool requires_bus_serialization_wait = false;
    std::string reason;
};

// Phase-aware 3-engine hybrid scheduler for Pixel 11 (malibu / Tensor G6).
bool route_malibu_workload(const MalibuRouteRequest& req,
                           MalibuRouteDecision* out_decision,
                           const MalibuHardwareProfile& prof = malibu_hardware_profile(),
                           std::string* err = nullptr);

}  // namespace freetile
