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
const char* ot_context_last_error(const ot_context_t* ctx);
int         ot_context_is_mock(const ot_context_t* ctx);

ot_status_t ot_device_get_info(const ot_context_t* ctx, ot_device_info_t* out_info);

/* IOMMU-mapped host staging & HBM device buffer management */
ot_status_t ot_buffer_alloc(ot_context_t* ctx, size_t size_bytes, ot_buffer_t** out_buf);
void        ot_buffer_free(ot_buffer_t* buf);
size_t      ot_buffer_size(const ot_buffer_t* buf);
uint64_t    ot_buffer_hbm_addr(const ot_buffer_t* buf);

ot_status_t ot_buffer_h2d(ot_context_t* ctx, ot_buffer_t* dst,
                          const void* host_src, size_t size_bytes);
ot_status_t ot_buffer_d2h(ot_context_t* ctx, const ot_buffer_t* src,
                          void* host_dst, size_t size_bytes);

/* Clean-room BF16 GEMM dispatch (M x K) * (K x N) -> (M x N) */
ot_status_t ot_matmul_bf16(ot_context_t* ctx,
                           const ot_buffer_t* lhs_mk,
                           const ot_buffer_t* rhs_kn,
                           ot_buffer_t* out_mn,
                           uint32_t m_dim,
                           uint32_t k_dim,
                           uint32_t n_dim);

#ifdef __cplusplus
}
#endif

#endif /* OPENTPU_H_ */
