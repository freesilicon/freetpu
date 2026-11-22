// SPDX-License-Identifier: Apache-2.0
#ifndef OPENTPU_INTERNAL_H_
#define OPENTPU_INTERNAL_H_

#include "opentpu.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ot {

static constexpr size_t kTile128x128Bf16Bytes = 128u * 128u * sizeof(uint16_t);

enum class BufferLayout : uint8_t {
  kRaw = 0,
  kTiled2x128Bf16 = 1,
};

struct PciDiscoveryEntry {
  std::string bdf;
  uint16_t vendor_id = 0;
  uint16_t device_id = 0;
  uint32_t pci_class = 0;
  std::string driver;
  uint32_t iommu_group = 0xffffffffu;
  uint64_t bar0_size = 0;
  uint64_t bar4_size = 0;
};

// Parse a 0x1234 hex sysfs attribute safely (no exceptions).
bool parse_hex_u64(const std::string& text, uint64_t* out);

// Simple monotonic IOVA bump/free-list allocator over valid IOMMU ranges.
struct IovaBlock {
  uint64_t iova = 0;
  uint64_t size = 0;
  bool in_use = false;
};

class IovaAllocator {
 public:
  void init(uint64_t base, uint64_t limit);
  bool alloc(uint64_t size, uint64_t align, uint64_t* out_iova);
  void free(uint64_t iova);
  uint64_t used_bytes() const;

 private:
  uint64_t base_ = 0;
  uint64_t limit_ = 0;
  uint64_t next_ = 0;
  std::vector<IovaBlock> blocks_;
};

// Clean-room BAR4 device descriptor protobuf wire reader (discovered on v5e-1 at BAR4+0x08/0x10).
struct Bar4Descriptor {
  uint32_t proto_len = 0;
  uint32_t header_word = 0;
  uint32_t core_count_observed = 0;
  std::string instance_token;
  bool valid = false;
};

bool parse_bar4_descriptor(const uint8_t* bar4_page, size_t page_len, Bar4Descriptor* out);

// VFIO + PCI device transport handle.
struct VfioTransport {
  int container_fd = -1;
  int group_fd = -1;
  int device_fd = -1;

  uint64_t bar0_offset = 0;
  uint64_t bar0_size = 0;
  uint64_t bar4_offset = 0;
  uint64_t bar4_size = 0;
  uint64_t cfg_offset = 0;
  uint64_t cfg_size = 0;

  volatile uint8_t* bar0_mmio = nullptr;
  size_t bar0_mapped_len = 0;
  volatile uint8_t* bar4_mmio = nullptr;
  size_t bar4_mapped_len = 0;

  bool is_mock = false;
  std::vector<uint8_t> mock_bar0;
  std::vector<uint8_t> mock_bar4;
  std::vector<uint8_t> mock_hbm;

  uint16_t saved_pci_cmd = 0;
  bool pci_cmd_saved = false;

  // Bare-metal BAR0 + 0x10000 ring queues and pinned host staging DMA memory.
  uint8_t* sq_ring_mem = nullptr;
  uint8_t* cq_ring_mem = nullptr;
  volatile uint32_t* status_page_mem = nullptr;
  uint8_t* host_stage_mem = nullptr;
  size_t   host_stage_size = 0;
  uint64_t host_stage_iova = 0;
  uint32_t sq_seq = 0;
  bool ring_ready = false;

  // IMEM boot/poller & kernel runtime state (initialized lazily on first live matmul).
  bool imem_booted = false;
  uint64_t scratch_lhs_hbm = 0;
  uint64_t scratch_rhs_hbm = 0;
  uint32_t exec_seq = 0;

  IovaAllocator iova_alloc;
  IovaAllocator hbm_alloc;

  bool open_device(const ot_config_t& cfg, ot_device_info_t* info, std::string* err);
  void close_device();

  bool map_dma_at(void* vaddr, uint64_t iova, uint64_t size, std::string* err);
  bool map_dma(void* vaddr, uint64_t size, uint64_t* out_iova, std::string* err);
  bool unmap_dma(uint64_t iova, uint64_t size);

  bool init_ring_queues(std::string* err);
  void teardown_ring_queues();
  bool submit_sq_wait(const uint32_t desc[32], uint32_t timeout_ms,
                      uint32_t cq_out[8], std::string* err);
  bool dma_transfer_sync(uint64_t host_iova, uint64_t dev_addr, size_t len_bytes,
                         bool is_h2d, uint32_t dst_flags_hi, uint32_t w28,
                         uint32_t w29, uint32_t timeout_ms, std::string* err);
  bool dma_h2d_sync(uint64_t host_iova, uint64_t dst_hbm, size_t len_bytes,
                    uint32_t w28, uint32_t w29, uint32_t timeout_ms, std::string* err);
  bool dma_d2h_sync(uint64_t host_iova, uint64_t src_hbm, size_t len_bytes,
                    uint32_t w28, uint32_t w29, uint32_t timeout_ms, std::string* err);
  bool dma_host_to_imem(uint64_t host_iova, uint64_t dst_imem, size_t len_bytes,
                        uint32_t timeout_ms, std::string* err);
  bool submit_core_ctrl(uint32_t ctrl_word, uint32_t timeout_ms,
                        uint32_t cq_out[8], std::string* err);
  bool submit_exec_sync(uint32_t expected_exec_seq, uint32_t timeout_ms,
                        uint32_t cq_out[8], std::string* err);
  bool ensure_imem_booted(uint32_t timeout_ms, std::string* err);
  bool ensure_matmul_scratch(std::string* err);
  void teardown_imem_state();

  bool read_pci_config(uint32_t offset, void* dst, size_t len);
  bool write_pci_config(uint32_t offset, const void* src, size_t len);
  bool enable_bus_master(std::string* err);

  uint32_t read32_bar4(uint64_t off) const;
  void write32_bar4(uint64_t off, uint32_t val);
  uint32_t read32_bar0(uint64_t off) const;
  void write32_bar0(uint64_t off, uint32_t val);
};

// Internal C++ helpers shared between ot_device.cpp and ot_pjrt.cpp.
ot_status_t buffer_h2d_tiled_128x128_bf16(ot_context_t* ctx, ot_buffer_t* dst,
                                          const void* host_src_row_major);
uint64_t context_hbm_used_bytes(const ot_context_t* ctx);

}  // namespace ot

struct ot_buffer {
  ot_context_t* ctx = nullptr;
  size_t size_bytes = 0;
  uint64_t hbm_addr = 0;
  ot::BufferLayout layout = ot::BufferLayout::kRaw;
  // Mock host mirror when running offline
  std::vector<uint8_t> mock_data;
};

struct ot_context {
  std::mutex mu;
  ot_config_t cfg{};
  ot_device_info_t info{};
  ot::VfioTransport transport{};
  ot::Bar4Descriptor bar4_desc{};
  std::string last_error;
};

#endif  // OPENTPU_INTERNAL_H_
