/*
 * libopentpu - Clean-Room Bare-Metal Cloud TPU Userspace Driver (/dev/vfio/<group>)
 *
 * Public C11 / C++17 ABI header.
 * Zero Google libtpu.so runtime dependencies.
 *
 * Transport (Linux x86_64/aarch64):
 *   - VFIO container/group/device: /dev/vfio/vfio + /dev/vfio/<iommu_group>
 *   - PCIe BAR0 (4 GiB MMIO/MSI-X/CSRs) & BAR4 (16 MiB config/descriptor aperture)
 *   - IOMMU Type1v2 DMA mapping (VFIO_IOMMU_MAP_DMA / UNMAP_DMA)
 *   - Offline host testing: set ot_config_t::mock_mode = 1
 */
#ifndef OPENTPU_H_
#define OPENTPU_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OT_API_VERSION_MAJOR 0u
#define OT_API_VERSION_MINOR 1u
#define OT_API_VERSION_PATCH 0u

typedef enum ot_status {
  OT_OK              = 0,
  OT_ERR_INVALID_ARG = -1,
  OT_ERR_NO_MEMORY   = -2,
  OT_ERR_DEVICE      = -3,
  OT_ERR_IOCTL       = -4,
  OT_ERR_TIMEOUT     = -5,
  OT_ERR_UNSUPPORTED = -6,
  OT_ERR_NOT_FOUND   = -7,
  OT_ERR_IO          = -8,
  OT_ERR_BUSY        = -9
} ot_status_t;

typedef enum ot_chip_gen {
  OT_CHIP_UNKNOWN = 0,
  OT_CHIP_V5E     = 1, /* PCI 0x1ae0:0x0063 */
  OT_CHIP_V6E     = 2, /* PCI 0x1ae0:0x006f (documented; transport not yet validated) */
  OT_CHIP_TPU7X   = 3
} ot_chip_gen_t;

typedef struct ot_config {
  uint32_t    struct_size;       /* sizeof(ot_config_t) or 0 */
  int         mock_mode;         /* 1 = offline mock device */
  const char* bdf;               /* Optional PCI BDF e.g. "0000:00:05.0", NULL = auto-discover */
  uint64_t    hbm_budget_bytes;  /* 0 = auto-detect chip HBM capacity */
  uint32_t    timeout_ms;        /* 0 = default 5000 ms */
} ot_config_t;

typedef struct ot_device_info {
  char          bdf[32];
  uint16_t      vendor_id;
  uint16_t      device_id;
  uint16_t      subsys_vendor_id;
  uint16_t      subsys_device_id;
  uint32_t      iommu_group;
  uint32_t      chip_gen;        /* enum ot_chip_gen */
  uint64_t      bar0_size;
  uint64_t      bar4_size;
  uint32_t      msix_vectors;
  uint64_t      hbm_bytes_total;
  uint64_t      vmem_bytes;
  uint64_t      smem_bytes;
  uint32_t      num_mxus;
  uint32_t      is_mock;
  uint32_t      is_prebooted;
} ot_device_info_t;

typedef struct ot_context ot_context_t;
typedef struct ot_buffer  ot_buffer_t;

ot_config_t ot_config_default(void);

ot_status_t ot_context_create(const ot_config_t* config, ot_context_t** out_ctx);
void        ot_context_destroy(ot_context_t* ctx);

const char* ot_status_string(ot_status_t status);
/* Pass NULL to retrieve the last ot_context_create() error on the calling thread. */
const char* ot_context_last_error(const ot_context_t* ctx);
int         ot_context_is_mock(const ot_context_t* ctx);

ot_status_t ot_device_get_info(const ot_context_t* ctx, ot_device_info_t* out_info);

/* IOMMU-mapped host staging & HBM device buffer management.
 * Callers should free buffers before destroying their owning context; any
 * outstanding buffers are detached during ot_context_destroy() so a later
 * ot_buffer_free() remains safe, though transfers on detached handles fail. */
ot_status_t ot_buffer_alloc(ot_context_t* ctx, size_t size_bytes, ot_buffer_t** out_buf);
void        ot_buffer_free(ot_buffer_t* buf);
size_t      ot_buffer_size(const ot_buffer_t* buf);
uint64_t    ot_buffer_hbm_addr(const ot_buffer_t* buf);

ot_status_t ot_buffer_h2d(ot_context_t* ctx, ot_buffer_t* dst,
                          const void* host_src, size_t size_bytes);
ot_status_t ot_buffer_d2h(ot_context_t* ctx, const ot_buffer_t* src,
                          void* host_dst, size_t size_bytes);

/* =========================================================================
 * FreeTile-Aligned Tile Descriptor & Quantized GEMM ABI
 * ========================================================================= */

#define OT_QK8_0 32u

typedef enum ot_dtype {
  OT_DTYPE_BF16 = 0,
  OT_DTYPE_F32  = 1,
  OT_DTYPE_INT8 = 2,
  OT_DTYPE_Q8_0 = 3,
  OT_DTYPE_Q4_0 = 4,
  OT_DTYPE_Q4_K = 5  /* Reserved; ot_matmul_quant returns OT_ERR_UNSUPPORTED */
} ot_dtype_t;

typedef enum ot_quant_mode {
  OT_QUANT_NONE        = 0,
  OT_QUANT_PER_TENSOR  = 1,
  OT_QUANT_PER_CHANNEL = 2,
  OT_QUANT_BLOCK_32    = 3,
  OT_QUANT_BLOCK_256   = 4  /* Reserved; ot_matmul_quant returns OT_ERR_UNSUPPORTED */
} ot_quant_mode_t;

/* Interleaved block-32 records used by ot_matmul_quant() when rhs_dtype is
 * OT_DTYPE_Q8_0 or OT_DTYPE_Q4_0, quant_mode is OT_QUANT_BLOCK_32, and
 * rhs_scales is NULL. Each record encodes 32 consecutive weights along K with
 * an inline BF16 scale `d`. Blocks are laid out as [K/32, N] when
 * rhs_transposed == 0, or [N, K/32] when rhs_transposed == 1. For Q4_0,
 * element ki in [0, 32) occupies the low nibble (even ki) or high nibble
 * (odd ki) of qs[ki >> 1] using offset-8 encoding (signed value = nibble - 8). */
#if defined(__GNUC__) || defined(__clang__)
typedef struct __attribute__((packed)) ot_block_q8_0 {
  uint16_t d;              /* BF16 block scale */
  int8_t   qs[OT_QK8_0];
} ot_block_q8_0_t;
typedef struct __attribute__((packed)) ot_block_q4_0 {
  uint16_t d;              /* BF16 block scale */
  uint8_t  qs[OT_QK8_0 / 2u];
} ot_block_q4_0_t;
#else
#pragma pack(push, 1)
typedef struct ot_block_q8_0 {
  uint16_t d;              /* BF16 block scale */
  int8_t   qs[OT_QK8_0];
} ot_block_q8_0_t;
typedef struct ot_block_q4_0 {
  uint16_t d;              /* BF16 block scale */
  uint8_t  qs[OT_QK8_0 / 2u];
} ot_block_q4_0_t;
#pragma pack(pop)
#endif

typedef struct ot_tile_shape {
  uint32_t block_m;            /* VMEM M tile height (default 128 on v5e) */
  uint32_t block_k;            /* VMEM K tile width (default 128 on v5e) */
  uint32_t block_n;            /* VMEM N tile width (default 128 on v5e) */
  uint8_t  vmem_double_buffer; /* Hint: multi-K BF16 uses 2-stage ping-pong; quant uses 1-stage */
  uint8_t  k_loop_accum_f32;   /* Hint (0 -> default): v5e resident GEMM always accumulates FP32 */
  uint8_t  rhs_transposed;     /* 0 = [K, N], 1 = [N, K] */
  uint8_t  reserved;
} ot_tile_shape_t;

typedef struct ot_freetile_gemm_desc {
  uint32_t        struct_size;      /* sizeof(ot_freetile_gemm_desc_t) or 0 */
  uint32_t        m_dim;
  uint32_t        k_dim;
  uint32_t        n_dim;
  ot_tile_shape_t tile;
  uint32_t        lhs_dtype;        /* enum ot_dtype */
  uint32_t        rhs_dtype;        /* enum ot_dtype */
  uint32_t        out_dtype;        /* enum ot_dtype */
  uint32_t        quant_mode;       /* enum ot_quant_mode */
  uint32_t        quant_group_size; /* 32 (or 0) for BLOCK_32, 0 otherwise */
  float           quant_scale;      /* Per-tensor scale when rhs_scales is NULL (rounded to BF16 on v5e) */
  uint32_t        scale_dtype;      /* Element dtype of rhs_scales: OT_DTYPE_BF16 or OT_DTYPE_F32 (rounded to BF16 on v5e MXU scale broadcast) */
  uint32_t        flags;
} ot_freetile_gemm_desc_t;

typedef struct ot_exec_stats {
  uint64_t hw_us;
  uint64_t wall_us;
  uint64_t hbm_bytes_moved;
  uint32_t tiles_m;
  uint32_t tiles_k;
  uint32_t tiles_n;
  uint32_t imem_bundles;
  uint32_t vmem_bytes_used;
  uint32_t reserved;
} ot_exec_stats_t;

ot_freetile_gemm_desc_t ot_freetile_gemm_desc_default(uint32_t m_dim,
                                                      uint32_t k_dim,
                                                      uint32_t n_dim);

/* Clean-room BF16 GEMM dispatch (M x K) * (K x N) -> (M x N) */
ot_status_t ot_matmul_bf16(ot_context_t* ctx,
                           const ot_buffer_t* lhs_mk,
                           const ot_buffer_t* rhs_kn,
                           ot_buffer_t* out_mn,
                           uint32_t m_dim,
                           uint32_t k_dim,
                           uint32_t n_dim);

ot_status_t ot_matmul_bf16_ex(ot_context_t* ctx,
                              const ot_buffer_t* lhs_mk,
                              const ot_buffer_t* rhs_kn,
                              ot_buffer_t* out_mn,
                              const ot_freetile_gemm_desc_t* desc,
                              ot_exec_stats_t* out_stats);

/* Clean-room quantized INT8/Q8_0/Q4_0 unpack + dequant + MXU GEMM.
 * - Split layout (rhs_scales != NULL): rhs_quant stores contiguous int8 or
 *   offset-8 packed 4-bit weights in [K, N] (rhs_transposed == 0) or [N, K]
 *   (rhs_transposed == 1); rhs_scales stores 1 (PER_TENSOR), N (PER_CHANNEL),
 *   or [K/32, N] (BLOCK_32) scales of element type desc->scale_dtype
 *   (OT_DTYPE_BF16 or OT_DTYPE_F32). Because the v5e IMEM kernel broadcasts
 *   block scales through the BF16 MXU before VPU dequantization, OT_DTYPE_F32
 *   host scales (and scalar quant_scale) are rounded to BF16 prior to weight
 *   scaling identically in both live hardware and offline mock execution.
 * - Inline block layout (rhs_scales == NULL with Q8_0/Q4_0 + BLOCK_32):
 *   rhs_quant stores interleaved ot_block_q8_0_t or ot_block_q4_0_t records.
 * - OT_DTYPE_Q4_K and OT_QUANT_BLOCK_256 are reserved and return OT_ERR_UNSUPPORTED. */
ot_status_t ot_matmul_quant(ot_context_t* ctx,
                            const ot_buffer_t* lhs_mk,
                            const ot_buffer_t* rhs_quant,
                            const ot_buffer_t* rhs_scales,
                            ot_buffer_t* out_mn,
                            const ot_freetile_gemm_desc_t* desc,
                            ot_exec_stats_t* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* OPENTPU_H_ */
