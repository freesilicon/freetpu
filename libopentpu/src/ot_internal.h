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
  uint64_t capacity_bytes() const { return limit_ > base_ ? (limit_ - base_) : 0; }

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
  uint8_t* ring_arena_mem = nullptr;
  size_t   ring_arena_bytes = 0;
  uint8_t* sq_ring_mem = nullptr;
  uint8_t* cq_ring_mem = nullptr;
  volatile uint32_t* status_page_mem = nullptr;
  uint8_t* host_stage_mem = nullptr;
  size_t   host_stage_size = 0;
  uint64_t host_stage_iova = 0;
  uint64_t iommu_min_page_size = 4096ULL;
  uint32_t sq_seq = 0;
  bool ring_ready = false;
  bool dma_faulted = false;

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

  bool read_pci_config(uint32_t offset, void* dst, size_t len);
  bool write_pci_config(uint32_t offset, const void* src, size_t len);
  bool enable_bus_master(std::string* err);

  uint32_t read32_bar4(uint64_t off) const;
  void write32_bar4(uint64_t off, uint32_t val);
  uint32_t read32_bar0(uint64_t off) const;
  void write32_bar0(uint64_t off, uint32_t val);
};

ot_status_t buffer_h2d_offset(ot_context_t* ctx, ot_buffer_t* dst, size_t dst_offset,
                              const void* host_src, size_t size_bytes);

}  // namespace ot

struct ot_buffer {
  ot_context_t* ctx = nullptr;
  size_t size_bytes = 0;
  uint64_t hbm_addr = 0;
  // Mock host mirror when running offline
  std::vector<uint8_t> mock_data;
};

struct ot_context {
  std::mutex mu;
  ot_config_t cfg{};
  ot_device_info_t info{};
  ot::VfioTransport transport{};
  ot::Bar4Descriptor bar4_desc{};
  std::vector<ot_buffer_t*> live_buffers;
  std::string last_error;
};

#endif  // OPENTPU_INTERNAL_H_
